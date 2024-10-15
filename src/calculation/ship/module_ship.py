import logging
import numpy as np
import pandas as pd
import sys
import traceback

from itertools import product
from typing import Any, Dict

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_shape, get_seeds, get_skims
from calculation.common.vrt import draw_choice_mcs
from .support_ship import (
    get_coeffs_distance_decay, get_nstr_to_ls, get_commodity_matrix,
    get_urban_density, get_make_use_distribution, get_flow_type_shares,
    get_cum_shares_vt_ucc, get_cep_shares,
    draw_delivery_times, draw_vehicle_type_and_shipment_size,
    generate_shipment_seeds,
    get_zonal_prod_attr, write_shipments_to_shp)

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Shipment Synthesis module.
    """
    try:

        chooseNearestDC = (varDict['NEAREST_DC'].upper() == 'TRUE')

        # Create shipments to/from external areas or not
        doExtArea = True

        # ----------------------- Import data --------------------------------

        logger.debug("\tImporting and preparing data...")

        if root is not None:
            root.update_statusbar("Shipment Synthesizer: Importing and preparing data")
            root.progressBar['value'] = 0

        nNSTR = len(dims.nstr) - 1
        nLS = len(dims.logistic_segment) - 1
        nVT = len([row for row in dims.vehicle_type.values() if row['IsRefTypeFreight'] == 1])

        id_parcel_consolidated = dims.get_id_from_label("logistic_segment", "Parcel (consolidated flows)")
        id_dangerous = dims.get_id_from_label("logistic_segment", "Dangerous")

        nFlowTypesInternal = len([row for row in dims.flow_type.values() if row['IsExternal'] == 0])
        nFlowTypesExternal = len([row for row in dims.flow_type.values() if row['IsExternal'] == 1])

        absoluteShipmentSizes = np.array([row['Median'] for row in dims.shipment_size.values()])

        seeds = get_seeds(varDict)

        dayToWeekFactor = float(varDict["DAY_TO_WEEK_FACTOR"]) if varDict["DAY_TO_WEEK_FACTOR"] != "" else 1.0

        # Distance decay parameters
        alpha, beta = get_coeffs_distance_decay(varDict)

        # Which NSTR belongs to which logistic segments (and vice versa)
        nstrToLS, lsToNstr = get_nstr_to_ls(varDict, nNSTR, nLS)

        if root is not None:
            root.progressBar['value'] = 0.2

        # Import make/use distribution tables
        makeDistribution, useDistribution = get_make_use_distribution(varDict, nNSTR, dims)

        if root is not None:
            root.progressBar['value'] = 0.4

        # Locations of the external zones
        superCoordinates = pd.read_csv(varDict['SUP_COORDINATES_ID'], sep='\t')
        superZoneX = np.array(superCoordinates['x_coord'])
        superZoneY = np.array(superCoordinates['y_coord'])
        nSuperZones = len(superCoordinates)

        # Import external zones demand
        if varDict['SHIPMENTS_REF'] == "":
            commodityMatrix = get_commodity_matrix(nstrToLS, nSuperZones, varDict, id_parcel_consolidated)

        if root is not None:
            root.progressBar['value'] = 1

        # Socio-economic data of zones
        segs = pd.read_csv(varDict['SEGS'], sep=',')
        segs.index = segs['zone']

        # Import internal zones data
        zonesShape = read_shape(varDict['ZONES']).sort_values('AREANR')
        zonesShape.index = zonesShape['AREANR']

        zoneID = np.array(zonesShape['AREANR'])
        zoneX = np.array(zonesShape['X'])
        zoneY = np.array(zonesShape['Y'])
        zoneLognode = np.array(zonesShape['LOGNODE'])
        zoneSurface = np.array(zonesShape['area'])

        nInternalZones = len(zonesShape)
        zoneDict = dict(np.transpose(np.vstack((np.arange(nInternalZones), zoneID))))
        for i in range(nSuperZones):
            zoneDict[nInternalZones + i] = superCoordinates.at[i, 'zone_mrdh']
        invZoneDict = dict((v, k) for k, v in zoneDict.items())
        zoneID = np.arange(nInternalZones)

        # Calculate urban density of zones
        urbanDensityCat = get_urban_density(
            segs, zonesShape, zoneDict, nInternalZones, nSuperZones)

        # Is a zone a DC, a TT or Producer/Consumer zone
        isDC = {}
        isTT = {}
        isPC = {}
        for i in zonesShape.index:
            isDC[i] = int(zonesShape.at[i, 'LOGNODE'] == 2)
            isTT[i] = int(zonesShape.at[i, 'LOGNODE'] == 1)
            isPC[i] = int(zonesShape.at[i, 'LOGNODE'] == 0)
        for i in range(nSuperZones):
            isDC[99999901 + i] = 0
            isTT[99999901 + i] = 0
            isPC[99999901 + i] = 0

        if root is not None:
            root.progressBar['value'] = 1.2

        # Import firm data
        if varDict['SHIPMENTS_REF'] == "":
            firms = (
                pd.read_csv(varDict['OUTPUTFOLDER'] + 'Firms.csv', sep=',')
                if varDict['FIRMS_REF'] == "" else
                pd.read_csv(varDict['FIRMS_REF'], sep=',')
            )

            firmID = np.array(firms['firm_id'])
            firmZone = np.array([invZoneDict[firms['zone_mrdh'][i]] for i in firms.index])
            firmSize = np.array(firms['employment'])
            firmX = np.array(firms['x_coord'])
            firmY = np.array(firms['y_coord'])
            firmSector = firms['employment_sector'].astype(int).values

            nFirms = len(firms)

        if root is not None:
            root.progressBar['value'] = 1.5

        # Import logistic nodes data
        distributionCenters = pd.read_csv(varDict['DISTRIBUTIECENTRA'], sep='\t')
        distributionCenters = distributionCenters[~pd.isna(distributionCenters['zone_mrdh'])]
        distributionCentersZone = np.array([invZoneDict[x] for x in distributionCenters['zone_mrdh']], dtype=int)
        distributionCentersX = np.array(distributionCenters['x_coord'])
        distributionCentersY = np.array(distributionCenters['y_coord'])

        # List of zone numbers Transshipment Terminals and Logistic Nodes
        ttZones = np.where(zoneLognode == dims.get_id_from_label("logistic_node", "Transshipment terminal"))[0]
        dcZones = distributionCentersZone

        # Flowtype distribution (10 NSTRs and 12 flowtypes)
        ftShares = get_flow_type_shares(varDict, dims)

        # Corrections
        if varDict['CORRECTIONS_TONNES'] != '':
            corrections = pd.read_csv(varDict['CORRECTIONS_TONNES'], sep='\t')
            nCorrections = len(corrections)

        if root is not None:
            root.progressBar['value'] = 1.5

        skimTravTime, skimDistance, nZones = get_skims(varDict)

        if (nSuperZones + nInternalZones) != nZones:
            raise Exception(
                f"The number of internal zones in the ZONES file ({nInternalZones}) " +
                f"and the number of external zones in the SUP_COORDINATES_ID file ({nSuperZones}) "
                f"don't match with the number of zones in the skim files: ({nZones})."
            )

        if root is not None:
            root.progressBar['value'] = 2.5

        # Cost parameters by vehicle type with size (small/medium/large)
        costPerKm = dict(
            (int(row['vehicle_type']), float(row['cost_per_km']))
            for row in pd.read_csv(varDict['COST_VEHTYPE'], sep='\t').to_dict('records')
        )
        costPerHour = dict(
            (int(row['vehicle_type']), float(row['cost_per_hour']))
            for row in pd.read_csv(varDict['COST_VEHTYPE'], sep='\t').to_dict('records')
        )

        # Cost parameters generic for sourcing (vehicle type is now known yet then)
        costPerKmSourcing = pd.read_csv(varDict['COST_SOURCING'], sep='\t').to_dict('records')[0]['cost_per_km']
        costPerHourSourcing = pd.read_csv(varDict['COST_SOURCING'], sep='\t').to_dict('records')[0]['cost_per_hour']

        # Estimated parameters MNL for combined shipment size and vehicle type
        paramsShipSizeVehType = dict(
            ((int(row['nstr']), str(row['parameter'])), float(row['value']))
            for row in pd.read_csv(varDict['PARAMS_SSVT'], sep='\t').to_dict('records')
        )

        if root is not None:
            root.progressBar['value'] = 2.6

        # Consolidation potential per logistic segment (for UCC scenario)
        cumProbsConsolidation = dict(
            (
                int(row['logistic_segment']),
                np.array([1.0 - float(row['probability']), 1.0]),
            )
            for row in pd.read_csv(varDict['ZEZ_CONSOLIDATION'], sep='\t').to_dict('records')
        )

        # Vehicle/combustion shares (for UCC scenario)
        cumSharesVehUCC = get_cum_shares_vt_ucc(varDict, dims)

        # Depots for parcel deliveries
        parcelNodes = read_shape(varDict['PARCELNODES'])
        parcelNodes.index = parcelNodes['id'].astype(int)
        parcelNodes = parcelNodes.sort_index()

        # Remove parcel nodes in external zones
        parcelNodes = parcelNodes[parcelNodes['AREANR']<99999900]

        # Convert zone numbers to index in skim
        parcelNodes['AREANR'] = [invZoneDict[x] for x in parcelNodes['AREANR']]
        nParcelNodes = len(parcelNodes)
        parcelNodes.index = np.arange(nParcelNodes)

        # Market shares of the different parcel couriers
        cepSharesTotal, cepSharesInternal, cepDict = get_cep_shares(varDict, parcelNodes)

        cepDepotZones = [
            np.array(parcelNodes.loc[parcelNodes['CEP'] == str(cep), 'AREANR'], dtype=int)
            for cep in cepDict.values()]
        cepDepotShares = [
            np.array(parcelNodes.loc[parcelNodes['CEP'] == str(cep), 'Surface'])
            for cep in cepDict.values()]
        cepDepotShares = [
            np.array(np.cumsum(cepDepotShares[i]) / np.sum(cepDepotShares[i]), dtype=float)
            for i in range(len(cepDict))]

        cepDepotX = [
            np.array(zonesShape['X'][[zoneDict[x] for x in cepDepotZones[i]]])
            for i in range(len(cepDict))]
        cepDepotY = [
            np.array(zonesShape['Y'][[zoneDict[x] for x in cepDepotZones[i]]])
            for i in range(len(cepDict))]

        truckCapacities = dict(
            (int(row['vehicle_type']), float(row['capacity_kg']) / 1000)
            for row in pd.read_csv(varDict['VEHICLE_CAPACITY'], sep='\t').to_dict('records')
        )

        if root is not None:
            root.progressBar['value'] = 2.8

        # ----------- Cumulative probability functions for allocation ---------

        if varDict['SHIPMENTS_REF'] == "":
            # Cumulative probability function of firms being receiver or sender
            probReceive = np.zeros((nFirms, nNSTR))
            probSend = np.zeros((nFirms, nNSTR))
            cumProbReceive = np.zeros((nFirms, nNSTR))
            cumProbSend = np.zeros((nFirms, nNSTR))

            # Per goods type, determine probability based on
            # firm size and make/use share
            for nstr in range(nNSTR):
                probReceive[:, nstr] = firmSize * np.array(
                    [useDistribution[nstr, sector] for sector in firmSector])
                probSend[:, nstr] = firmSize * np.array(
                    [makeDistribution[nstr, sector] for sector in firmSector])
                cumProbReceive[:, nstr] = np.cumsum(probReceive[:, nstr])
                cumProbReceive[:, nstr] /= cumProbReceive[-1, nstr]
                cumProbSend[:, nstr] = np.cumsum(probSend[:, nstr])
                cumProbSend[:, nstr] /= cumProbSend[-1, nstr]

            # Cumulative probability function of a shipment being allocated
            # to a particular DC/TT (based on surface)
            probDC = np.array(distributionCenters['surface_m2'])
            cumProbDC = np.cumsum(probDC)
            cumProbDC = cumProbDC / cumProbDC[-1]

            probTT = zoneSurface[ttZones]
            cumProbTT = np.cumsum(probTT)
            cumProbTT = cumProbTT / cumProbTT[-1]

        if root is not None:
            root.progressBar['value'] = 2.9

        # ----------------------- Demand by flowtype -------------------------

        if varDict['SHIPMENTS_REF'] == "":
            # Split demand by internal/export/import and goods type
            demandInternal = np.array(commodityMatrix['WeightDay'][:nLS], dtype=float)
            demandExport = np.zeros((nSuperZones, nLS), dtype=float)
            demandImport = np.zeros((nSuperZones, nLS), dtype=float)

            for superZone in range(nSuperZones):
                for ls in range(nLS):
                    exportIndex = (superZone + 1) * (nSuperZones + 1) * nLS + ls
                    demandExport[superZone][ls] = commodityMatrix['WeightDay'][exportIndex]

                    importIndex = (superZone + 1) * nLS + ls
                    demandImport[superZone][ls] = commodityMatrix['WeightDay'][importIndex]

            # Then split demand by flowtype
            demandInternalByFT = [None for i in range(nFlowTypesInternal)]
            demandExportByFT = [None for i in range(nFlowTypesExternal)]
            demandImportByFT = [None for i in range(nFlowTypesExternal)]

            for ft in range(nFlowTypesInternal):
                demandInternalByFT[ft] = demandInternal * np.array(
                    [ftShares[ls][ft] for ls in ftShares.keys()])

            for ft in range(nFlowTypesExternal):
                demandExportByFT[ft] = demandExport * np.array(
                    [ftShares[ls][nFlowTypesInternal + ft] for ls in ftShares.keys()])
                demandImportByFT[ft] = demandImport * np.array(
                    [ftShares[ls][nFlowTypesInternal + ft] for ls in ftShares.keys()])

        if root is not None:
            root.progressBar['value'] = 3

        # ------------------- Shipment synthesizer procedure ------------------

        if varDict['SHIPMENTS_REF'] == "":

            # Initialize a counter for the procedure
            count = 0

            # Initialize shipment attributes as dictionaries
            fromFirm = {}
            toFirm = {}
            flowType = {}
            goodsType = {}
            logisticSegment = {}
            shipmentSize = {}
            shipmentSizeCat = {}
            vehicleType = {}
            destZone = {}
            origZone = {}
            origX = {}
            origY = {}
            destX = {}
            destY = {}

            logger.debug("\tSynthesizing shipments within study area...")

            percStart = 3
            percEnd = 40

            if root is not None:
                root.update_statusbar("Shipment Synthesizer: Synthesizing shipments within study area")
                root.progressBar['value'] = percStart

            # For progress bar
            totalWeightInternal = dayToWeekFactor * np.sum([
                np.sum(demandInternalByFT[ft])
                for ft in range(nFlowTypesInternal)])
            allocatedWeightInternal = 0

            for ls, nstr in product(range(nLS), range(nNSTR)):

                if lsToNstr[nstr, ls] <= 0:
                    continue

                logger.debug(f"\t\tFor logistic segment {ls} (NSTR{nstr})",)

                if root is not None:
                    root.update_statusbar(
                        "Shipment Synthesizer: " +
                        f"Synthesizing shipments within study area (LS {ls} and NSTR {nstr})")
                    

                for ft in range(nFlowTypesInternal):
                    tmpShipmentNumber = 0
                    allocatedWeight = 0
                    totalWeight = dayToWeekFactor * demandInternalByFT[ft][ls] * lsToNstr[nstr, ls]

                    tmpMaxNumShipments = int(np.ceil(totalWeight / min(absoluteShipmentSizes)))
                    tmpSeedsReceiver = generate_shipment_seeds(
                        seeds['shipment_internal_receiver'], tmpMaxNumShipments, ls, nstr, ft)
                    tmpSeedsSender = generate_shipment_seeds(
                        seeds['shipment_internal_sender'], tmpMaxNumShipments, ls, nstr, ft)
                    tmpSeedsVehicleType = generate_shipment_seeds(
                        seeds['shipment_internal_vehicle_type'], tmpMaxNumShipments, ls, nstr, ft)

                    # While the weight of all synthesized shipments for this segment so far
                    # does not exceed the total weight for this segment
                    while allocatedWeight < totalWeight:
                        flowType[count] = ft + 1
                        goodsType[count] = nstr
                        logisticSegment[count] = ls

                        # Flows between parcel depots
                        if ls == id_parcel_consolidated:
                            cep = draw_choice_mcs(cepSharesInternal, tmpSeedsReceiver[tmpShipmentNumber])
                            depot = draw_choice_mcs(cepDepotShares[cep], 2 * tmpSeedsReceiver[tmpShipmentNumber])
                            toFirm[count] = 0
                            destZone[count] = cepDepotZones[cep][depot]
                            destX[count] = cepDepotX[cep][depot]
                            destY[count] = cepDepotY[cep][depot]
                            toDC = 1

                        # Determine receiving firm for flows to consumer
                        elif (flowType[count] in (1, 3, 6)):
                            toFirm[count] = draw_choice_mcs(
                                cumProbReceive[:, nstr], tmpSeedsReceiver[tmpShipmentNumber])
                            destZone[count] = firmZone[toFirm[count]]
                            destX[count] = firmX[toFirm[count]]
                            destY[count] = firmY[toFirm[count]]
                            toDC = 0
                        # Determine receiving DC for flows to DC
                        elif (flowType[count] in (2, 5, 8)):
                            toFirm[count] = draw_choice_mcs(
                                cumProbDC, tmpSeedsReceiver[tmpShipmentNumber])
                            destZone[count] = dcZones[toFirm[count]]
                            destX[count] = distributionCentersX[toFirm[count]]
                            destY[count] = distributionCentersY[toFirm[count]]
                            toDC = 1
                        # Determine receiving Transshipment Terminal for flows to TT
                        elif (flowType[count] in (4, 7, 9)):
                            toFirm[count] = ttZones[draw_choice_mcs(
                                cumProbTT, tmpSeedsReceiver[tmpShipmentNumber])]
                            destZone[count] = toFirm[count]
                            destX[count] = zoneX[toFirm[count]]
                            destY[count] = zoneY[toFirm[count]]
                            toDC = 0

                        tmpCostTime = (
                            costPerHourSourcing *
                            skimTravTime[destZone[count]::nZones] / 3600)
                        tmpCostDist = (
                            costPerKmSourcing *
                            skimDistance[destZone[count]::nZones] / 1000)
                        tmpCost = tmpCostTime + tmpCostDist

                        distanceDecay = 1 / (1 + np.exp(alpha + beta * np.log(tmpCost)))

                        if (flowType[count] in (1, 2, 4)):
                            distanceDecay = distanceDecay[firmZone]
                        elif (flowType[count] in (3, 5, 7)):
                            distanceDecay = distanceDecay[dcZones]
                        elif (flowType[count] in (6, 8, 9)):
                            distanceDecay = distanceDecay[ttZones]

                        distanceDecay /= np.sum(distanceDecay)

                        if ls == id_parcel_consolidated:
                            origDepot = draw_choice_mcs(
                                cepDepotShares[cep], tmpSeedsSender[tmpShipmentNumber])

                            if origDepot == depot:
                                if depot == 0:
                                    origDepot += 1
                                elif depot == len(cepDepotShares[cep]) - 1:
                                    origDepot -= 1
                                else:
                                    origDepot += [-1, 1][draw_choice_mcs(
                                        np.array([0.5, 1.0]), 2 * tmpSeedsSender[tmpShipmentNumber])]
                            depot = origDepot

                            fromFirm[count] = 0
                            origZone[count] = cepDepotZones[cep][depot]
                            origX[count] = cepDepotX[cep][depot]
                            origY[count] = cepDepotY[cep][depot]
                            fromDC = 1

                        # Determine sending firm for flows from consumer
                        elif (flowType[count] in (1, 2, 4)):
                            prob = probSend[:, nstr] * distanceDecay
                            prob = np.cumsum(prob)
                            prob /= prob[-1]
                            fromFirm[count] = draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])
                            origZone[count] = firmZone[fromFirm[count]]
                            origX[count] = firmX[fromFirm[count]]
                            origY[count] = firmY[fromFirm[count]]
                            fromDC = 0

                        # Determine sending DCfor flows from DC
                        elif (flowType[count] in (3, 5 ,7)):
                            if chooseNearestDC and flowType[count] == 3:
                                dists = skimDistance[distributionCentersZone * nZones + destZone[count]]
                                fromFirm[count] = np.argmin(dists)
                            else:
                                prob = probDC * distanceDecay
                                prob = np.cumsum(prob)
                                prob /= prob[-1]
                                fromFirm[count] = draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])
                            origZone[count] = dcZones[fromFirm[count]]
                            origX[count] = distributionCentersX[fromFirm[count]]
                            origY[count] = distributionCentersY[fromFirm[count]]
                            fromDC = 1

                        # Determine sending Transshipment Terminal for flows from TT
                        elif (flowType[count] in (6, 8, 9)):
                            prob = probTT * distanceDecay
                            prob = np.cumsum(prob)
                            prob /= prob[-1]
                            fromFirm[count] = ttZones[draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])]
                            origZone[count] = fromFirm[count]
                            origX[count] = zoneX[fromFirm[count]]
                            origY[count] = zoneY[fromFirm[count]]
                            fromDC = 0

                        ssChosen, vtChosen = draw_vehicle_type_and_shipment_size(
                            paramsShipSizeVehType, nstr,
                            skimTravTime[(origZone[count]) * nZones + (destZone[count])] / 3600,
                            skimDistance[(origZone[count]) * nZones + (destZone[count])] / 1000,
                            fromDC, toDC,
                            costPerHour, costPerKm, truckCapacities,
                            nVT, absoluteShipmentSizes, tmpSeedsVehicleType[tmpShipmentNumber]
                        )

                        shipmentSizeCat[count] = ssChosen
                        shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                        vehicleType[count] = vtChosen

                        # Update weight and counter
                        allocatedWeight += shipmentSize[count]
                        allocatedWeightInternal += shipmentSize[count]
                        count += 1
                        tmpShipmentNumber += 1

                        if root is not None:
                            if count % 300 == 0:
                                root.progressBar['value'] = (
                                    percStart +
                                    (percEnd - percStart) *
                                    (allocatedWeightInternal / totalWeightInternal))

            if doExtArea:

                logger.debug("\tSynthesizing shipments leaving study area...")

                percStart = 40
                percEnd = 65

                if root is not None:
                    root.update_statusbar(
                        "Shipment Synthesizer: Synthesizing shipments leaving study area")
                    root.progressBar['value'] = percStart

                # For progress bar
                totalWeightExport = dayToWeekFactor * np.sum([
                    np.sum(np.sum(demandExportByFT[ft]))
                    for ft in range(nFlowTypesExternal)])
                allocatedWeightExport = 0

                for ls, nstr in product(range(nLS), range(nNSTR)):

                    if lsToNstr[nstr, ls] <= 0:
                        continue

                    logger.debug(f"\t\tFor logistic segment {ls} (NSTR{nstr})")

                    if root is not None:
                        root.update_statusbar(
                            "Shipment Synthesizer: " +
                            f"Synthesizing shipments leaving study area (LS {ls} and NSTR {nstr})")

                    for ft in range(nFlowTypesExternal):

                        for dest in range(nSuperZones):
                            tmpShipmentNumber = 0
                            allocatedWeight = 0
                            totalWeight = dayToWeekFactor * demandExportByFT[ft][dest, ls] * lsToNstr[nstr, ls]

                            tmpMaxNumShipments = int(np.ceil(totalWeight / min(absoluteShipmentSizes)))
                            tmpSeedsSender = generate_shipment_seeds(
                                seeds['shipment_export_sender'], tmpMaxNumShipments, ls, nstr, ft, dest)
                            tmpSeedsVehicleType = generate_shipment_seeds(
                                seeds['shipment_export_vehicle_type'], tmpMaxNumShipments, ls, nstr, ft, dest)

                            tmpCostTime = (
                                costPerHourSourcing *
                                skimTravTime[(nInternalZones + dest)::nZones] / 3600)
                            tmpCostDist = (
                                costPerKmSourcing *
                                skimDistance[(nInternalZones + dest)::nZones] / 1000)
                            tmpCost = tmpCostTime + tmpCostDist

                            distanceDecay = 1 / (1 + np.exp(alpha + beta * np.log(tmpCost)))

                            if ft + 1 + nFlowTypesInternal == 10:
                                distanceDecay = distanceDecay[firmZone]
                            elif ft + 1 + nFlowTypesInternal == 11:
                                distanceDecay = distanceDecay[dcZones]
                            elif ft + 1 + nFlowTypesInternal == 12:
                                distanceDecay = distanceDecay[ttZones]

                            distanceDecay /= np.sum(distanceDecay)

                            while allocatedWeight < totalWeight:
                                flowType[count] = ft + 1 + nFlowTypesInternal
                                logisticSegment[count] = ls
                                goodsType[count] = nstr
                                toFirm[count] = 0
                                destZone[count] = nInternalZones + dest
                                destX[count] = superZoneX[dest]
                                destY[count] = superZoneY[dest]

                                if ls == id_parcel_consolidated:
                                    cep = draw_choice_mcs(cepSharesTotal, tmpSeedsSender[tmpShipmentNumber])
                                    depot = draw_choice_mcs(cepDepotShares[cep], 2 * tmpSeedsSender[tmpShipmentNumber])
                                    fromFirm[count] = 0
                                    origZone[count] = cepDepotZones[cep][depot]
                                    origX[count] = cepDepotX[cep][depot]
                                    origY[count] = cepDepotY[cep][depot]
                                    fromDC = 1

                                # From consumer
                                elif flowType[count] == 10:
                                    prob = np.cumsum(probSend[:, nstr] * distanceDecay)
                                    prob /= prob[-1]
                                    fromFirm[count] = draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])
                                    origZone[count] = firmZone[fromFirm[count]]
                                    origX[count] = firmX[fromFirm[count]]
                                    origY[count] = firmY[fromFirm[count]]
                                    fromDC = 0

                                # From distribution center
                                elif flowType[count] == 11:
                                    prob = np.cumsum(probDC * distanceDecay)
                                    prob /= prob[-1]
                                    fromFirm[count] = draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])
                                    origZone[count] = dcZones[fromFirm[count]]
                                    origX[count] = distributionCentersX[fromFirm[count]]
                                    origY[count] = distributionCentersY[fromFirm[count]]
                                    fromDC = 1

                                # From transshipment terminal
                                elif flowType[count] == 12:
                                    prob = np.cumsum(probTT * distanceDecay)
                                    prob /= prob[-1]
                                    fromFirm[count] = ttZones[draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])]
                                    origZone[count] = fromFirm[count]
                                    origX[count] = zoneX[fromFirm[count]]
                                    origY[count] = zoneY[fromFirm[count]]
                                    fromDC = 0

                                ssChosen, vtChosen = draw_vehicle_type_and_shipment_size(
                                    paramsShipSizeVehType, nstr,
                                    skimTravTime[(origZone[count]) * nZones + (destZone[count])] / 3600,
                                    skimDistance[(origZone[count]) * nZones + (destZone[count])] / 1000,
                                    fromDC, toDC,
                                    costPerHour, costPerKm, truckCapacities,
                                    nVT, absoluteShipmentSizes, tmpSeedsVehicleType[tmpShipmentNumber]
                                )

                                shipmentSizeCat[count] = ssChosen
                                shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                                vehicleType[count] = vtChosen

                                # Update weight and counter
                                allocatedWeight += shipmentSize[count]
                                allocatedWeightExport += shipmentSize[count]
                                count += 1
                                tmpShipmentNumber += 1

                                if root is not None:
                                    if count % 300 == 0:
                                        root.progressBar['value'] = (
                                            percStart +
                                            (percEnd - percStart) *
                                            (allocatedWeightExport/ totalWeightExport))

                fromDC = 0

                logger.debug("\tSynthesizing shipments entering study area...")

                percStart = 65
                percEnd = 90

                if root is not None:
                    root.progressBar['value'] = percStart
                    root.update_statusbar(
                        "Shipment Synthesizer: Synthesizing shipments entering study area")

                totalWeightImport = dayToWeekFactor * np.sum([
                    np.sum(np.sum(demandImportByFT[ft]))
                    for ft in range(nFlowTypesExternal)])
                allocatedWeightImport  = 0

                for ls, nstr in product(range(nLS), range(nNSTR)):

                    if lsToNstr[nstr, ls] <= 0:
                        continue

                    logger.debug(f"\t\tFor logistic segment {ls} (NSTR{nstr})")

                    if root is not None:
                        root.update_statusbar(
                            "Shipment Synthesizer: " +
                            f"Synthesizing shipments entering study area (LS {ls}) and NSTR {nstr})")

                    for ft in range(nFlowTypesExternal):

                        for orig in range(nSuperZones):
                            tmpShipmentNumber = 0
                            allocatedWeight = 0
                            totalWeight = dayToWeekFactor * demandImportByFT[ft][orig, ls] * lsToNstr[nstr, ls]

                            tmpMaxNumShipments = int(np.ceil(totalWeight / min(absoluteShipmentSizes)))
                            tmpSeedsReceiver = generate_shipment_seeds(
                                seeds['shipment_import_receiver'], tmpMaxNumShipments, ls, nstr, ft, dest)
                            tmpSeedsVehicleType = generate_shipment_seeds(
                                seeds['shipment_import_vehicle_type'], tmpMaxNumShipments, ls, nstr, ft, dest)

                            tmpCostTime = (
                                costPerHourSourcing *
                                skimTravTime[(nInternalZones + orig) * nZones + np.arange(nZones)] /
                                3600)
                            tmpCostDist = (
                                costPerKmSourcing *
                                skimDistance[(nInternalZones + orig) * nZones + np.arange(nZones)] /
                                1000)
                            tmpCost = tmpCostTime + tmpCostDist

                            distanceDecay = 1 / (1 + np.exp(alpha + beta * np.log(tmpCost)))

                            if ft + 1 + nFlowTypesInternal == 10:
                                distanceDecay = distanceDecay[firmZone]
                            elif ft + 1 + nFlowTypesInternal == 11:
                                distanceDecay = distanceDecay[dcZones]
                            elif ft + 1 + nFlowTypesInternal == 12:
                                distanceDecay = distanceDecay[ttZones]

                            distanceDecay /= np.sum(distanceDecay)

                            while allocatedWeight < totalWeight:
                                flowType[count] = ft + 1 + nFlowTypesInternal
                                logisticSegment[count] = ls
                                goodsType[count] = nstr
                                fromFirm[count] = 0
                                origZone[count] = nInternalZones + orig
                                origX[count] = superZoneX[orig]
                                origY[count] = superZoneY[orig]

                                if ls == id_parcel_consolidated:
                                    cep = draw_choice_mcs(cepSharesTotal, tmpSeedsReceiver[tmpShipmentNumber])
                                    depot = draw_choice_mcs(cepDepotShares[cep], 2 * tmpSeedsReceiver[tmpShipmentNumber])
                                    toFirm[count] = 0
                                    destZone[count] = cepDepotZones[cep][depot]
                                    destX[count] = cepDepotX[cep][depot]
                                    destY[count] = cepDepotY[cep][depot]
                                    toDC = 1

                                # To consumer
                                elif flowType[count] == 10:
                                    prob = np.cumsum(probReceive[:, nstr] * distanceDecay)
                                    prob /= prob[-1]
                                    toFirm[count] = draw_choice_mcs(prob, tmpSeedsReceiver[tmpShipmentNumber])
                                    destZone[count] = firmZone[toFirm[count]]
                                    destX[count] = firmX[toFirm[count]]
                                    destY[count] = firmY[toFirm[count]]
                                    toDC = 0

                                # To distribution center
                                elif flowType[count] == 11:
                                    prob = np.cumsum(probDC * distanceDecay)
                                    prob /= prob[-1]
                                    toFirm[count] = draw_choice_mcs(prob, tmpSeedsReceiver[tmpShipmentNumber])
                                    destZone[count] = dcZones[toFirm[count]]
                                    destX[count] = distributionCentersX[toFirm[count]]
                                    destY[count] = distributionCentersY[toFirm[count]]
                                    toDC = 1

                                # To transshipment terminal
                                elif flowType[count] == 12:
                                    prob = np.cumsum(probTT * distanceDecay)
                                    prob /= prob[-1]
                                    toFirm[count] = ttZones[draw_choice_mcs(prob, tmpSeedsReceiver[tmpShipmentNumber])]
                                    destZone[count] = toFirm[count]
                                    destX[count] = zoneX[toFirm[count]]
                                    destY[count] = zoneY[toFirm[count]]
                                    toDC = 0

                                ssChosen, vtChosen = draw_vehicle_type_and_shipment_size(
                                    paramsShipSizeVehType, nstr,
                                    skimTravTime[(origZone[count]) * nZones + (destZone[count])] / 3600,
                                    skimDistance[(origZone[count]) * nZones + (destZone[count])] / 1000,
                                    fromDC, toDC,
                                    costPerHour, costPerKm, truckCapacities,
                                    nVT, absoluteShipmentSizes, tmpSeedsVehicleType[tmpShipmentNumber]
                                )

                                shipmentSizeCat[count] = ssChosen
                                shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                                vehicleType[count] = vtChosen

                                # Update weight and counter
                                allocatedWeight += shipmentSize[count]
                                allocatedWeightImport += shipmentSize[count]
                                count += 1
                                tmpShipmentNumber += 1

                                if root is not None:
                                    if count % 300 == 0:
                                        root.progressBar['value'] = (
                                            percStart +
                                            (percEnd - percStart) *
                                            (allocatedWeightImport / totalWeightImport))

            if varDict['CORRECTIONS_TONNES'] != '':
                logger.debug("\tSynthesizing additional shipments (corrections)...")

                if root is not None:
                    root.update_statusbar(
                        "Shipment Synthesizer: " +
                        "Synthesizing additional shipments (corrections)")

                percStart = 90
                percEnd = 92

                if root is not None:
                    root.progressBar['value'] = percStart
                    root.update_statusbar(
                        "Shipment Synthesizer: " +
                        "Synthesizing additional shipments (corrections)")

                for cor in range(nCorrections):
                    orig = int(corrections.at[cor, 'orig_mrdh'])
                    dest = int(corrections.at[cor, 'dest_mrdh'])

                    logger.debug(f"\t\tAdditional shipments (correction {cor+1})")

                    ls = int(corrections.at[cor, 'logistic_segment'])

                    for nstr in range(nNSTR):

                        if lsToNstr[nstr, ls] <= 0:
                            continue

                        tmpShipmentNumber = 0
                        totalWeight = dayToWeekFactor * float(corrections.at[cor, 'tonnes_day']) * lsToNstr[nstr, ls]
                        allocatedWeight = 0

                        tmpMaxNumShipments = int(np.ceil(totalWeight / min(absoluteShipmentSizes)))
                        tmpSeedsReceiver = generate_shipment_seeds(
                            seeds['shipment_corrections_receiver'], tmpMaxNumShipments, ls, nstr, cor)
                        tmpSeedsSender = generate_shipment_seeds(
                            seeds['shipment_corrections_sender'], tmpMaxNumShipments, ls, nstr, cor)
                        tmpSeedsVehicleType = generate_shipment_seeds(
                            seeds['shipment_corrections_vehicle_type'], tmpMaxNumShipments, ls, nstr, cor)

                        # While the weight of all synthesized shipments for this segment so far
                        # does not exceed the total weight for this segment
                        while allocatedWeight < totalWeight:
                            flowType[count] = 1
                            goodsType[count] = nstr
                            logisticSegment[count] = ls

                            # Determine receiving firm
                            if dest == -1:
                                toFirm[count] = draw_choice_mcs(cumProbReceive[:, nstr], tmpSeedsReceiver[tmpShipmentNumber])
                                destZone[count] = firmZone[toFirm[count]]
                                destX[count] = firmX[toFirm[count]]
                                destY[count] = firmY[toFirm[count]]
                            else:
                                toFirm[count] = -99999
                                destZone[count] = invZoneDict[dest]
                                destX[count] = zoneX[destZone[count]]
                                destY[count] = zoneY[destZone[count]]

                            toDC = 0

                            tmpCostTime = (
                                costPerHourSourcing *
                                skimTravTime[destZone[count]::nZones] / 3600) 
                            tmpCostDist = (
                                costPerKmSourcing *
                                skimDistance[destZone[count]::nZones] / 1000)
                            tmpCost = tmpCostTime + tmpCostDist

                            distanceDecay = 1 / (1 + np.exp(alpha + beta * np.log(tmpCost)))
                            distanceDecay = distanceDecay[firmZone]
                            distanceDecay /= np.sum(distanceDecay)

                            # Determine sending firm
                            if orig == -1:
                                prob = probSend[:, nstr].copy()
                                prob *= distanceDecay
                                prob = np.cumsum(prob)
                                prob /= prob[-1]
                                fromFirm[count] = draw_choice_mcs(prob, tmpSeedsSender[tmpShipmentNumber])
                                origZone[count] = firmZone[fromFirm[count]]
                                origX[count] = firmX[fromFirm[count]]
                                origY[count] = firmY[fromFirm[count]]
                            else:
                                fromFirm[count] = -99999
                                origZone[count] = invZoneDict[orig]
                                origX[count]    = zoneX[origZone[count]]
                                origY[count]    = zoneY[origZone[count]]

                            fromDC = 0

                            ssChosen, vtChosen = draw_vehicle_type_and_shipment_size(
                                paramsShipSizeVehType, nstr,
                                skimTravTime[(origZone[count]) * nZones + (destZone[count])] / 3600,
                                skimDistance[(origZone[count]) * nZones + (destZone[count])] / 1000,
                                fromDC, toDC,
                                costPerHour, costPerKm, truckCapacities,
                                nVT, absoluteShipmentSizes, tmpSeedsVehicleType[tmpShipmentNumber]
                            )

                            shipmentSizeCat[count] = ssChosen
                            shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                            vehicleType[count] = vtChosen

                            # Update weight and counter
                            allocatedWeight += shipmentSize[count]
                            count += 1
                            tmpShipmentNumber += 1

                    if root is not None:
                        root.progressBar['value'] = (
                            percStart +
                            (percEnd - percStart) * (cor / nCorrections))

            nShips = count

            # ------------------------ Delivery time choice -------------------

            logger.debug("\tDelivery time choice...")

            if root is not None:
                root.progressBar['value'] = 92
                root.update_statusbar("Shipment Synthesizer: Delivery time choice")
                
            # Determine delivery time period for each shipment
            deliveryTimePeriod, lowerTOD, upperTOD = draw_delivery_times(
                origZone, destZone, logisticSegment, vehicleType, isTT, isPC,
                urbanDensityCat, zoneDict, nLS, varDict, dims, seeds['shipment_delivery_time'],
            )

            # --------------------- Creating shipments CSV --------------------
            
            shipCols  = [
                "SHIP_ID",
                "ORIG", "DEST",
                "NSTR",
                "WEIGHT", "WEIGHT_CAT", 
                "FLOWTYPE", "LS", 
                "VEHTYPE",
                "SEND_FIRM", "RECEIVE_FIRM",
                "SEND_DC", "RECEIVE_DC",
                "TOD_PERIOD", "TOD_LOWER", "TOD_UPPER"]

            shipments = pd.DataFrame(np.zeros((nShips, len(shipCols))), columns=shipCols)
    
            shipments['SHIP_ID'     ] = np.arange(nShips)
            shipments['ORIG'        ] = [zoneDict[x] for x in origZone.values()]
            shipments['DEST'        ] = [zoneDict[x] for x in destZone.values()]
            shipments['NSTR'        ] = list(goodsType.values())
            shipments['WEIGHT'      ] = list(shipmentSize.values())
            shipments['WEIGHT_CAT'  ] = list(shipmentSizeCat.values())
            shipments['FLOWTYPE'    ] = list(flowType.values())
            shipments['LS'          ] = list(logisticSegment.values())
            shipments['VEHTYPE'     ] = list(vehicleType.values())
            shipments['SEND_FIRM'   ] = [firmID[x] if x != -99999 else -99999 for x in fromFirm.values()]
            shipments['RECEIVE_FIRM'] = [firmID[x] if x != -99999 else -99999 for x in toFirm.values()]
            shipments['SEND_DC'     ] = -99999
            shipments['RECEIVE_DC'  ] = -99999
            shipments['TOD_PERIOD'  ] = deliveryTimePeriod.values()
            shipments['TOD_LOWER'   ] = lowerTOD.values()
            shipments['TOD_UPPER'   ] = upperTOD.values()

            # For the external zones and logistical nodes there is no firm, hence firm ID -99999
            shipments.loc[shipments['ORIG'] > 99999900, 'SEND_FIRM'] = -99999
            shipments.loc[shipments['DEST'] > 99999900, 'RECEIVE_FIRM'] = -99999
            shipments.loc[shipments['LS'] == id_parcel_consolidated, ['SEND_FIRM','RECEIVE_FIRM']] = -99999
            shipments.loc[shipments['FLOWTYPE'] > 10, ['SEND_FIRM','RECEIVE_FIRM']] = -99999
            shipments.loc[shipments['FLOWTYPE'].isin([2, 5, 8]), 'RECEIVE_FIRM'] = -99999
            shipments.loc[shipments['FLOWTYPE'].isin([4, 7, 9]), 'RECEIVE_FIRM'] = -99999
            shipments.loc[shipments['FLOWTYPE'].isin([3, 5, 7]), 'SEND_FIRM'] = -99999
            shipments.loc[shipments['FLOWTYPE'].isin([6, 8, 9]), 'SEND_FIRM'] = -99999

            # Only fill in DC ID for shipments to and from DC
            whereToDC = shipments['FLOWTYPE'].isin([2, 5, 8]) | ((shipments['FLOWTYPE'] == 11) & (shipments['ORIG'] > 99999900))
            whereFromDC = shipments['FLOWTYPE'].isin([3, 5, 7]) | ((shipments['FLOWTYPE'] == 11) & (shipments['DEST'] > 99999900))
            shipments.loc[whereToDC,  'RECEIVE_DC'] = np.array(list(toFirm.values()))[whereToDC]
            shipments.loc[whereFromDC,'SEND_DC'   ] = np.array(list(fromFirm.values()))[whereFromDC]

        else:
            # Import the reference shipments
            shipments = pd.read_csv(varDict['SHIPMENTS_REF'])

        # Get the datatypes right
        intCols = [
            "SHIP_ID",
            "ORIG", "DEST",
            "NSTR",
            "WEIGHT_CAT",
            "FLOWTYPE",
            "LS", 
            "VEHTYPE",
            "SEND_FIRM", "RECEIVE_FIRM",
            "SEND_DC", "RECEIVE_DC",
            "TOD_PERIOD", "TOD_LOWER", "TOD_UPPER"]
        floatCols = ['WEIGHT']
        shipments[intCols  ] = shipments[intCols].astype(int)
        shipments[floatCols] = shipments[floatCols].astype(float)

        # Redirect shipments via UCCs and change vehicle type
        if varDict['LABEL'] == 'UCC':
            if varDict['SHIPMENTS_REF'] == "":

                logger.debug(f"\tExporting REF shipments to {varDict['OUTPUTFOLDER']}Shipments_REF.csv")
                
                if root is not None:
                    root.progressBar['value'] = 93
                    root.update_statusbar("Shipment Synthesizer: Exporting REF shipments to CSV")

                shipments.to_csv(varDict['OUTPUTFOLDER'] + 'Shipments_REF.csv')

            logger.debug("\tRedirecting shipments via UCC...")

            if root is not None:
                root.progressBar['value'] = 94
                root.update_statusbar("Shipment Synthesizer: Redirecting shipments via UCC")

            shipments['FROM_UCC'] = 0
            shipments['TO_UCC'  ] = 0

            whereOrigZEZ = np.array([
                i for i in shipments[shipments['ORIG'] < 99999900].index
                if zonesShape['ZEZ'][shipments['ORIG'][i]] >= 1], dtype=int)
            whereDestZEZ = np.array([
                i for i in shipments[shipments['DEST'] < 99999900].index
                if zonesShape['ZEZ'][shipments['DEST'][i]] >= 1], dtype=int)
            setWhereOrigZEZ = set(whereOrigZEZ)
            setWhereDestZEZ = set(whereDestZEZ)

            whereBothZEZ = [
                i for i in shipments.index
                if i in setWhereOrigZEZ and i in setWhereDestZEZ]

            newShipments = pd.DataFrame(np.zeros(shipments.shape))
            newShipments.columns = shipments.columns
            newShipments[intCols  ] = newShipments[intCols].astype(int)
            newShipments[floatCols] = newShipments[floatCols].astype(float)

            count = 0

            for i in whereOrigZEZ:

                if i in setWhereDestZEZ:
                    continue

                ls = int(shipments['LS'][i])

                if draw_choice_mcs(cumProbsConsolidation[ls], seeds['shipment_zez_consolidation'] + i) == 0:
                    continue

                trueOrigin = int(shipments['ORIG'][i])
                newOrigin = zonesShape['UCC_zone'][trueOrigin]

                # Redirect to UCC
                shipments.at[i,'ORIG'    ] = newOrigin
                shipments.at[i,'FROM_UCC'] = 1
                if varDict['SHIPMENTS_REF'] == "":
                    origX[i] = zoneX[invZoneDict[newOrigin]]
                    origY[i] = zoneY[invZoneDict[newOrigin]]

                # Add shipment from ZEZ to UCC
                newShipments.loc[count, :] = list(shipments.loc[i, :].copy())
                newShipments.at[count,'ORIG'    ] = trueOrigin
                newShipments.at[count,'DEST'    ] = newOrigin
                newShipments.at[count,'FROM_UCC'] = 0
                newShipments.at[count,'TO_UCC'  ] = 1
                newShipments.at[count,'VEHTYPE' ] = draw_choice_mcs(
                    cumSharesVehUCC[ls], seeds['shipment_zez_vehicle_type'] + i)

                if varDict['SHIPMENTS_REF'] == "":
                    origX[nShips+count] = zoneX[invZoneDict[trueOrigin]]
                    origY[nShips+count] = zoneY[invZoneDict[trueOrigin]]
                    destX[nShips+count] = zoneX[invZoneDict[newOrigin]]
                    destY[nShips+count] = zoneY[invZoneDict[newOrigin]]

                count += 1

            for i in whereDestZEZ:

                if i in setWhereOrigZEZ:
                    continue

                ls = int(shipments['LS'][i])

                if draw_choice_mcs(cumProbsConsolidation[ls], seeds['shipment_zez_consolidation'] + i) == 0:
                    continue

                trueDest = int(shipments['DEST'][i])
                newDest = zonesShape['UCC_zone'][trueDest]

                # Redirect to UCC
                shipments.at[i,'DEST'  ] = newDest
                shipments.at[i,'TO_UCC'] = 1
                if varDict['SHIPMENTS_REF'] == "":
                    destX[i] = zoneX[invZoneDict[newDest]]
                    destY[i] = zoneY[invZoneDict[newDest]]   

                # Add shipment to ZEZ from UCC
                newShipments.loc[count, :] = list(shipments.loc[i, :].copy())
                newShipments.at[count,'ORIG'    ] = newDest
                newShipments.at[count,'DEST'    ] = trueDest
                newShipments.at[count,'FROM_UCC'] = 1
                newShipments.at[count,'TO_UCC'  ] = 0
                newShipments.at[count,'VEHTYPE' ] = draw_choice_mcs(
                    cumSharesVehUCC[ls], seeds['shipment_zez_vehicle_type'] + i)

                if varDict['SHIPMENTS_REF'] == "":
                    origX[nShips+count] = zoneX[invZoneDict[newDest]]
                    origY[nShips+count] = zoneY[invZoneDict[newDest]]
                    destX[nShips+count] = zoneX[invZoneDict[trueDest]]
                    destY[nShips+count] = zoneY[invZoneDict[trueDest]]

                count += 1

            # Also change vehicle type and rerouting
            # for shipments that go from a ZEZ area to a ZEZ area
            for i in whereBothZEZ:
                ls = int(shipments['LS'][i])

                # Als het binnen dezelfde gemeente (i.e. dezelfde ZEZ) blijft,
                # dan hoeven we alleen maar het voertuigtype aan te passen

                # Assume dangerous goods keep the same vehicle type
                gemeenteOrig = zonesShape['Gemeentena'][shipments['ORIG'][i]]
                gemeenteDest = zonesShape['Gemeentena'][shipments['DEST'][i]]
                if gemeenteOrig == gemeenteDest:
                    if ls != id_dangerous:
                        shipments.at[i,'VEHTYPE'] = draw_choice_mcs(
                            cumSharesVehUCC[ls], seeds['shipment_zez_vehicle_type'] + i)

                # Als het van de ene ZEZ naar de andere ZEZ gaat,
                # maken we 3 legs: ZEZ1--> UCC1, UCC1-->UCC2, UCC2-->ZEZ2
                elif draw_choice_mcs(cumProbsConsolidation[ls], seeds['shipment_zez_consolidation'] + i) == 1:
                    trueOrigin = int(shipments['ORIG'][i])
                    trueDest   = int(shipments['DEST'][i])
                    newOrigin  = zonesShape['UCC_zone'][trueOrigin]
                    newDest    = zonesShape['UCC_zone'][trueDest]

                    # Redirect to UCC
                    shipments.at[i,'ORIG'    ] = newOrigin
                    shipments.at[i,'FROM_UCC'] = 1
                    if varDict['SHIPMENTS_REF'] == "":
                        origX[i] = zoneX[invZoneDict[newOrigin]]
                        origY[i] = zoneY[invZoneDict[newOrigin]]

                    # Add shipment from ZEZ1 to UCC1
                    newShipments.loc[count, :] = list(shipments.loc[i, :].copy())
                    newShipments.at[count,'ORIG'    ] = trueOrigin
                    newShipments.at[count,'DEST'    ] = newOrigin
                    newShipments.at[count,'FROM_UCC'] = 0
                    newShipments.at[count,'TO_UCC'  ] = 1
                    newShipments.at[count,'VEHTYPE' ] = draw_choice_mcs(
                        cumSharesVehUCC[ls], seeds['shipment_zez_vehicle_type'] + i)
                    if varDict['SHIPMENTS_REF'] == "":
                        origX[nShips + count] = zoneX[invZoneDict[trueOrigin]]
                        origY[nShips + count] = zoneY[invZoneDict[trueOrigin]]
                        destX[nShips + count] = zoneX[invZoneDict[newOrigin]]
                        destY[nShips + count] = zoneY[invZoneDict[newOrigin]]

                    count += 1

                    # Redirect to UCC
                    shipments.at[i,'DEST'  ] = newDest
                    shipments.at[i,'TO_UCC'] = 1
                    if varDict['SHIPMENTS_REF'] == "":
                        destX[i] = zoneX[invZoneDict[newDest]]
                        destY[i] = zoneY[invZoneDict[newDest]]

                    # Add shipment from UCC2 to ZEZ2
                    newShipments.loc[count, :] = list(shipments.loc[i, :].copy())
                    newShipments.at[count,'ORIG'    ] = newDest
                    newShipments.at[count,'DEST'    ] = trueDest
                    newShipments.at[count,'FROM_UCC'] = 1
                    newShipments.at[count,'TO_UCC'  ] = 0
                    newShipments.at[count,'VEHTYPE' ] = draw_choice_mcs(
                        cumSharesVehUCC[ls], 2 * seeds['shipment_zez_vehicle_type'] + i)
                    if varDict['SHIPMENTS_REF'] == "":
                        origX[nShips+count] = zoneX[invZoneDict[newDest]]
                        origY[nShips+count] = zoneY[invZoneDict[newDest]]
                        destX[nShips+count] = zoneX[invZoneDict[trueDest]]
                        destY[nShips+count] = zoneY[invZoneDict[trueDest]]

                    count += 1

            newShipments = newShipments.iloc[np.arange(count), :]

            shipments = shipments.append(newShipments)
            nShips = len(shipments)
            shipments['SHIP_ID'] = np.arange(nShips)
            shipments.index      = np.arange(nShips)

        logger.debug(
            f"\tExporting {varDict['LABEL']} shipments to " +
            f"{varDict['OUTPUTFOLDER']}Shipments_{varDict['LABEL']}.csv")

        if root is not None:
            root.progressBar['value'] = 95
            root.update_statusbar("Shipment Synthesizer: Exporting shipments to CSV")

        shipments[intCols  ] = shipments[intCols].astype(int)
        shipments[floatCols] = shipments[floatCols].astype(float)

        shipments.to_csv(
            varDict['OUTPUTFOLDER'] + f"Shipments_{varDict['LABEL']}.csv",
            index=False)  

        if varDict['SHIPMENTS_REF'] == "":

            # ---------------- Zonal productions and attractions --------------
            logger.debug("\tWriting zonal productions/attractions...")

            if root is not None:
                root.progressBar['value'] = 97
                root.update_statusbar(
                    "Shipment Synthesizer: " +
                    "Writing zonal productions/attractions")
                
            zonalProductions, zonalAttractions = get_zonal_prod_attr(
                shipments, zoneDict, invZoneDict, nInternalZones, nSuperZones, nLS)

            # Export to csv
            zonalProductions.to_csv(
                varDict['OUTPUTFOLDER'] + f"zonal_productions_{varDict['LABEL']}.csv",
                index=False)
            zonalAttractions.to_csv(
                varDict['OUTPUTFOLDER'] + f"zonal_attractions_{varDict['LABEL']}.csv",
                index=False)

            # --------------------- Creating shipments SHP --------------------
            
            logger.debug("\tWriting Shapefile...")

            percStart = 97
            percEnd = 100

            if root is not None:
                root.progressBar['value'] = percStart
                root.update_statusbar("Shipment Synthesizer: Writing Shapefile")

            write_shipments_to_shp(
                shipments, origX, origY, destX, destY, varDict, root, percStart, percEnd)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
