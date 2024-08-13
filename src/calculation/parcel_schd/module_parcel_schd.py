import logging
import numpy as np
import pandas as pd
import sys
import traceback

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_mtx, read_shape, get_seeds, get_skims
from .support_parcel_schd import (
    cluster_parcels, create_schedules, do_crowdshipping,
    write_schedules_to_geojson, export_trip_matrices)

from typing import Any, Dict

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Parcel Demand module.
    """
    try:

        if root is not None:
            root.progressBar['value'] = 0

        maxVehicleLoad = int(varDict['PARCELS_MAXLOAD'])
        doCrowdShipping = (str(varDict['CROWDSHIPPING']).upper() == 'TRUE')

        exportTripMatrix = True

        # ------------------------- Import data--------------------------------

        logger.debug("\tImporting data...")

        seeds = get_seeds(varDict)

        parcels = pd.read_csv(
            f"{varDict['OUTPUTFOLDER']}ParcelDemand_{varDict['LABEL']}.csv", sep=',')

        parcelNodes, coords = read_shape(varDict['PARCELNODES'], returnGeometry=True)
        parcelNodes['X'] = [coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y'] = [coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes['id'] = parcelNodes['id'].astype(int)
        parcelNodes.index = parcelNodes['id']
        parcelNodes = parcelNodes.sort_index()
        parcelNodesCEP = {}
        for i in parcelNodes.index:
            parcelNodesCEP[parcelNodes.at[i, 'id']] = parcelNodes.at[i, 'CEP']

        zones = read_shape(varDict['ZONES'])
        zones = zones.sort_values('AREANR')
        zones.index = zones['AREANR']
        supCoordinates = pd.read_csv(varDict['SUP_COORDINATES_ID'], sep='\t')
        supCoordinates.index = supCoordinates['zone_mrdh']

        zonesX = {}
        zonesY = {}
        for areanr in zones.index:
            zonesX[areanr] = zones.at[areanr, 'X']
            zonesY[areanr] = zones.at[areanr, 'Y']
        for areanr in supCoordinates.index:
            zonesX[areanr] = supCoordinates.at[areanr, 'x_coord']
            zonesY[areanr] = supCoordinates.at[areanr, 'y_coord']

        nIntZones = len(zones)
        nSupZones = len(supCoordinates)
        zoneDict = dict(np.transpose(np.vstack(
            (np.arange(1, nIntZones + 1),
             zones['AREANR']))))
        zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones + i + 1] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        # Change zoning to skim zones which run continuously from 0
        parcels['X'] = [zonesX[x] for x in parcels['D_zone'].values]
        parcels['Y'] = [zonesY[x] for x in parcels['D_zone'].values]
        parcels['D_zone'] = [invZoneDict[x] for x in parcels['D_zone']]
        parcels['O_zone'] = [invZoneDict[x] for x in parcels['O_zone']]
        parcelNodes['skim_zone'] = [invZoneDict[x] for x in parcelNodes['AREANR']]

        if root is not None:
            root.progressBar['value'] = 0.3

        # System input for scheduling
        parcelDepTime = np.array(
            pd.read_csv(varDict['DEPTIME_PARCELS'], sep='\t')['cumulative_share'])
        dropOffTime = varDict['PARCELS_DROPTIME'] / 3600

        # Skims
        skimTravTime, skimDistance, nZones = get_skims(varDict)

        if root is not None:
            root.progressBar['value'] = 1.0

        # ---------------- Crowdshipping use case -----------------------------
        if doCrowdShipping:

            # The first N zones for which to consider
            # crowdshipping for parcel deliveries
            nFirstZonesCS = 5925

            # Percentage of parcels eligible for crowdshipping
            parcelShareCRW = varDict['CRW_PARCELSHARE']

            # Input data and parameters for the crowdshipping use case
            modeParamsCRW = pd.read_csv(varDict['CRW_MODEPARAMS'], index_col=0, sep=',')

            modes = {'fiets': {}, 'auto': {}, }

            modes['fiets']['willingness'] = modeParamsCRW.at['BIKE', 'WILLINGNESS']
            modes['fiets']['dropoff_time'] = modeParamsCRW.at['BIKE', 'DROPOFFTIME']
            modes['fiets']['VoT'] = modeParamsCRW.at['BIKE', 'VOT']

            modes['auto']['willingness'] = modeParamsCRW.at['CAR', 'WILLINGNESS']
            modes['auto']['dropoff_time'] = modeParamsCRW.at['CAR', 'DROPOFFTIME']
            modes['auto']['VoT'] = modeParamsCRW.at['CAR', 'VOT']

            modes['fiets']['relative_extra_parcel_dist_threshold'] = (
                modeParamsCRW.at['BIKE', 'RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD'])
            modes['auto']['relative_extra_parcel_dist_threshold'] = (
                modeParamsCRW.at['CAR', 'RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD'])

            modes['fiets']['OD_path'] = varDict['CRW_PDEMAND_BIKE']
            modes['fiets']['skim_time'] = skimDistance / 1000 / 12 * 3600
            modes['fiets']['n_trav'] = 0
            modes['auto']['OD_path'] = varDict['CRW_PDEMAND_CAR']
            modes['auto']['skim_time'] = skimTravTime
            modes['auto']['n_trav'] = 0

            for mode in modes:
                modes[mode]['OD_array'] = read_mtx(
                    modes[mode]['OD_path']).reshape(nIntZones, nIntZones)
                modes[mode]['OD_array'] *= modes[mode]['willingness']
                modes[mode]['OD_array'] = np.array(
                    np.round(modes[mode]['OD_array']), dtype=int)
                modes[mode]['OD_array'] = modes[mode]['OD_array'][:nFirstZonesCS, :]
                modes[mode]['OD_array'] = modes[mode]['OD_array'][:, :nFirstZonesCS]

            # Which zones are located in which municipality
            zone_gemeente_dict = dict(np.transpose(np.vstack((
                np.arange(1, nIntZones + 1),
                zones['Gemeentena']))))

            # Get retail jobs per zone from socio-economic data
            segs = pd.read_csv(varDict['SEGS'], sep=',')
            segs.index = segs['zone']
            segsDetail = np.array(segs['6: detail'])

            # Perform the crowdshipping calculations
            do_crowdshipping(
                parcels, zones, nIntZones, nZones, zoneDict,
                zonesX, zonesY,
                skimDistance, skimTravTime,
                nFirstZonesCS, parcelShareCRW, modes,
                zone_gemeente_dict, segsDetail,
                varDict['OUTPUTFOLDER'], varDict['LABEL'],
                seeds['parcel_crowdshipping'],
                root)

        # -------------------- Forming spatial clusters of parcels ------------

        logger.debug("\tForming spatial clusters of parcels...")

        # A measure of euclidean distance based on the coordinates
        skimEuclidean = (
            np.array(list(zonesX.values())).repeat(nZones).reshape(nZones, nZones) -
            np.array(list(zonesX.values())).repeat(nZones).reshape(nZones, nZones).transpose())**2
        skimEuclidean += (
            np.array(list(zonesY.values())).repeat(nZones).reshape(nZones, nZones) -
            np.array(list(zonesY.values())).repeat(nZones).reshape(nZones, nZones).transpose())**2
        skimEuclidean = skimEuclidean**0.5
        skimEuclidean = skimEuclidean.flatten()
        skimEuclidean /= np.sum(skimEuclidean)

        # To prevent instability related to possible mistakes in skim,
        # use average of skim and euclidean distance
        # (both normalized to a sum of 1)
        skimClustering = skimDistance.copy()
        skimClustering /= np.sum(skimClustering)
        skimClustering += skimEuclidean

        del skimEuclidean

        if varDict['LABEL'] == 'UCC':

            # Divide parcels into the 4 tour types, namely:
            # 0: Depots to households
            # 1: Depots to UCCs
            # 2: From UCCs, by van
            # 3: From UCCs, by LEVV
            parcelsUCC = {
                0: pd.DataFrame(parcels[(parcels['FROM_UCC'] == 0) & (parcels['TO_UCC'] == 0)]),
                1: pd.DataFrame(parcels[(parcels['FROM_UCC'] == 0) & (parcels['TO_UCC'] == 1)]),
                2: pd.DataFrame(parcels[(parcels['FROM_UCC'] == 1) & (parcels['VEHTYPE'] == 7)]),
                3: pd.DataFrame(parcels[(parcels['FROM_UCC'] == 1) & (parcels['VEHTYPE'] == 8)]),
            }

            # Cluster parcels based on proximity and constrained
            # by vehicle capacity
            for i in range(3):
                if doCrowdShipping:
                    startValueProgress = 56.0 +       i / 3 * (70.0 - 56.0)
                    endValueProgress   = 56.0 + (i + 1) / 3 * (70.0 - 56.0)
                else:
                    startValueProgress = 2.0 +       i / 3 * (55.0 - 2.0)
                    endValueProgress   = 2.0 + (i + 1) / 3 * (55.0 - 2.0)

                logger.debug(f"\tTour type {i + 1}...")

                parcelsUCC[i].index = np.arange(len(parcelsUCC[i]))
                parcelsUCC[i] = cluster_parcels(
                    parcelsUCC[i],
                    maxVehicleLoad, skimClustering,
                    root, startValueProgress, endValueProgress)

            # LEVV have smaller capacity
            startValueProgress = 70.0 if doCrowdShipping else 55.0
            startValueProgress = 75.0 if doCrowdShipping else 60.0

            logger.debug("\tTour type 4...")

            parcelsUCC[3] = cluster_parcels(
                parcelsUCC[3],
                int(round(maxVehicleLoad / 5)), skimClustering,
                root, startValueProgress, endValueProgress)

            # Aggregate parcels based on depot, cluster and destination
            for i in range(4):

                if i <= 1:
                    parcelsUCC[i] = pd.pivot_table(
                        parcelsUCC[i],
                        values=['Parcel_ID'],
                        index=['DepotNumber', 'Cluster', 'O_zone', 'D_zone'],
                        aggfunc={'Parcel_ID': 'count'})
                    parcelsUCC[i] = parcelsUCC[i].rename(
                        columns={'Parcel_ID': 'Parcels'})

                    parcelsUCC[i]['Depot'] = [x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Cluster'] = [x[1] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Orig'] = [x[2] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Dest'] = [x[3] for x in parcelsUCC[i].index]

                else:
                    parcelsUCC[i] = pd.pivot_table(
                        parcelsUCC[i],
                        values=['Parcel_ID'],
                        index=['O_zone', 'Cluster', 'D_zone'],
                        aggfunc={'Parcel_ID': 'count'})
                    parcelsUCC[i] = parcelsUCC[i].rename(
                        columns={'Parcel_ID': 'Parcels'})

                    parcelsUCC[i]['Depot'] = [x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Cluster'] = [x[1] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Orig'] = [x[0] for x in parcelsUCC[i].index]
                    parcelsUCC[i]['Dest'] = [x[2] for x in parcelsUCC[i].index]

                parcelsUCC[i].index = np.arange(len(parcelsUCC[i]))

        if varDict['LABEL'] != 'UCC':
            # Cluster parcels based on proximity and constrained
            # by vehicle capacity
            startValueProgress = 56.0 if doCrowdShipping else 2.0
            endValueProgress = 75.0 if doCrowdShipping else 60.0
            parcels = cluster_parcels(
                parcels,
                maxVehicleLoad, skimClustering,
                root, startValueProgress, endValueProgress)

            # Aggregate parcels based on depot, cluster and destination
            parcels = pd.pivot_table(
                parcels,
                values=['Parcel_ID'],
                index=['DepotNumber', 'Cluster', 'O_zone', 'D_zone'],
                aggfunc={'Parcel_ID': 'count'})
            parcels = parcels.rename(columns={'Parcel_ID': 'Parcels'})

            parcels['Depot'] = [x[0] for x in parcels.index]
            parcels['Cluster'] = [x[1] for x in parcels.index]
            parcels['Orig'] = [x[2] for x in parcels.index]
            parcels['Dest'] = [x[3] for x in parcels.index]

            parcels.index = np.arange(len(parcels))

        del skimClustering

        # ----------- Scheduling of trips (UCC scenario) ----------------------

        if varDict['LABEL'] == 'UCC':

            # Depots to households
            logger.debug("\tStarting scheduling procedure for parcels from depots to households...")

            startValueProgress = 75.0 if doCrowdShipping else 60.0
            endValueProgress = 80.0
            tourType = 0
            deliveries = create_schedules(
                parcelsUCC[0],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                seeds['parcel_departure_time'] * tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs
            logger.debug("\tStarting scheduling procedure for parcels from depots to UCC...")

            startValueProgress = 80.0
            endValueProgress = 83.0
            tourType = 1
            deliveries1 = create_schedules(
                parcelsUCC[1],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                seeds['parcel_departure_time'] * tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs (van)
            logger.debug("\tStarting scheduling procedure for parcels from UCCs (by van)...")

            startValueProgress = 83.0
            endValueProgress = 86.0
            tourType = 2
            deliveries2 = create_schedules(
                parcelsUCC[2],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                seeds['parcel_departure_time'] * tourType,
                root, startValueProgress, endValueProgress)

            # Depots to UCCs (LEVV)
            logger.debug("\tStarting scheduling procedure for parcels from UCCs (by LEVV)...")

            startValueProgress = 86.0
            endValueProgress = 89.0
            tourType = 3
            deliveries3 = create_schedules(
                parcelsUCC[3],
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                seeds['parcel_departure_time'] * tourType,
                root, startValueProgress, endValueProgress)

            # Combine deliveries of all tour types
            deliveries = pd.concat([deliveries, deliveries1, deliveries2, deliveries3])
            deliveries.index = np.arange(len(deliveries))

        # ----------- Scheduling of trips (REF scenario) ---------------------

        if varDict['LABEL'] != 'UCC':

            logger.debug("\tStarting scheduling procedure for parcels...")

            startValueProgress = 75.0 if doCrowdShipping else 60.0
            endValueProgress = 90.0
            tourType = 0

            deliveries = create_schedules(
                parcels,
                dropOffTime,
                skimTravTime, skimDistance,
                parcelNodesCEP,
                parcelDepTime,
                tourType,
                seeds['parcel_departure_time'] * tourType,
                root, startValueProgress, endValueProgress)

        # ---------------- Export output table to CSV and SHP -----------------

        # Transform to MRDH zone numbers and export
        deliveries['O_zone'] = [zoneDict[x] for x in deliveries['O_zone']]
        deliveries['D_zone'] = [zoneDict[x] for x in deliveries['D_zone']]
        deliveries['TripDepTime'] = deliveries['TripDepTime'].round(3)
        deliveries['TripEndTime'] = deliveries['TripEndTime'].round(3)

        logger.debug(
            "\tWriting scheduled trips to " +
            varDict['OUTPUTFOLDER'] + f"ParcelSchedule_{varDict['LABEL']}.csv")

        deliveries.to_csv(
            varDict['OUTPUTFOLDER'] + f"ParcelSchedule_{varDict['LABEL']}.csv",
            index=False)

        if root is not None:
            root.progressBar['value'] = 91.0

        logger.debug("\tWriting GeoJSON...")

        write_schedules_to_geojson(
            deliveries, parcelNodes, zonesX, zonesY, varDict, root)

        logger.debug(
            "\tParcel schedules written to " +
            varDict['OUTPUTFOLDER'] +
            f"ParcelSchedule_{varDict['LABEL']}.geojson")

        # ---------------------- Create and export trip matrices --------------

        if exportTripMatrix:

            logger.debug("\tGenerating trip matrix...")

            export_trip_matrices(deliveries, varDict)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
