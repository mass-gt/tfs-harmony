import functools
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import shapefile as shp
import sys
import traceback

from typing import Any, Dict

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_shape, get_num_cpu, get_seeds, get_skims
from calculation.common.vrt import draw_choice_mcs
from .support_tour import (
    assign_shipments_to_carriers, form_tours,
    get_traveltime, max_nstr, sum_weight, get_cum_shares_comb_ucc)

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Tour Formation module.
    """
    try:

        if root is not None:
            root.progressBar['value'] = 0

        shiftFreightToComb1 = varDict['SHIFT_FREIGHT_TO_COMB1']
        shiftFreightToComb2 = varDict['SHIFT_FREIGHT_TO_COMB2']
        doShiftToElectric = (shiftFreightToComb1 != "")
        doShiftToHydrogen = (shiftFreightToComb2 != "")

        nNSTR = len(dims.nstr) - 1
        nLS = len(dims.logistic_segment) - 1
        nVT = len(dims.vehicle_type)

        nCPU = get_num_cpu(varDict, 8)

        # The number of carriers that transport the shipments not going to or from a DC
        nCarriersNonDC = 100

        # Maximum number of shipments in tour
        maxNumShips = 10

        # Average dwell time at a stop (in hours)
        avgDwellTime = 0.25

        if root is not None:
            root.progressBar['value'] = 0.1

        logger.debug("\tImporting shipments and zones...")

        seeds = get_seeds(varDict)

        # Carrying capacity for each vehicle type
        carryingCapacity = dict(
            (int(row['vehicle_type']), float(row['capacity_kg']))
            for row in pd.read_csv(varDict['VEHICLE_CAPACITY'], sep='\t').to_dict('records')
        )

        # Import zones
        zones = read_shape(varDict['ZONES'])
        zones = zones.sort_values('AREANR')

        UCCzones = np.unique(zones['UCC_zone'])[1:]

        # Add level of urbanization based on segs (# houses / jobs) and surface (square m)
        segs = pd.read_csv(varDict['SEGS'])
        segs.index = segs['zone']
        zones.index = zones['AREANR']
        zones['HOUSEDENSITY'] = segs['1: woningen'] / zones['area']
        zones['JOBDENSITY'] = segs['9: arbeidspl_totaal'] / zones['area']

        # Approximately highest 10% percentile
        zones['STED'] = (
            (zones['HOUSEDENSITY'] > 0.0075) |
            (zones['JOBDENSITY'] > 0.0050))

        # Clean up zones
        # (remove unnecessary columns and replace NaNs in LOGNODE with 0)
        zones = zones[['AREANR', 'X', 'Y', 'LOGNODE', 'STED', 'ZEZ']]
        zones.loc[pd.isna(zones['LOGNODE']), 'LOGNODE'] = 0
        zones = np.array(zones)

        if root is not None:
            root.progressBar['value'] = 0.5

        # Get coordinates of the external zones
        coropCoordinates = pd.read_csv(varDict['SUP_COORDINATES_ID'], sep='\t')
        coropCoordinates.index = np.arange(len(coropCoordinates))

        nSupZones = len(coropCoordinates)
        nIntZones = len(zones)

        zoneDict = dict(np.transpose(np.vstack((
            np.arange(1, nIntZones + 1),
            zones[:, 0]))))
        zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones + i + 1] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        # Add external zones to zone matrix
        zones = np.append(zones, np.zeros((nSupZones, zones.shape[1])), axis=0)

        for i in range(nSupZones):
            # ZoneID, X coordinate, Y coordinate
            zones[nIntZones + i][0] = nIntZones + i
            zones[nIntZones + i][1] = coropCoordinates['x_coord'][i]
            zones[nIntZones + i][2] = coropCoordinates['y_coord'][i]

            # No logistic node
            zones[nIntZones + i][3] = 0

            # No urbanized zone
            zones[nIntZones + i][4] = False

        # Import logistic nodes data
        distributionCenters = pd.read_csv(varDict['DISTRIBUTIECENTRA'], sep='\t')
        distributionCenters = distributionCenters[~pd.isna(distributionCenters['zone_mrdh'])]
        distributionCenters['zone_mrdh'] = [invZoneDict[x] for x in distributionCenters['zone_mrdh']]
        nDC = len(distributionCenters)
        distributionCenters.index = np.arange(nDC)
        dcZones = np.array(distributionCenters['zone_mrdh'])

        # Import ZEZ scenario input
        if varDict['LABEL'] == 'UCC':
            cumProbCombustion = get_cum_shares_comb_ucc(varDict, dims)

        # Import shipments data
        shipments = pd.read_csv(varDict['OUTPUTFOLDER'] + f"Shipments_{varDict['LABEL']}.csv")

        if root is not None:
            root.progressBar['value'] = 1.0

        logger.debug("\tPreparing shipments for tour formation...")

        shipments['ORIG'] = [invZoneDict[x] for x in shipments['ORIG']]
        shipments['DEST'] = [invZoneDict[x] for x in shipments['DEST']]
        shipments['WEIGHT'] *= 1000

        # Is the shipment loaded at a distribution center?
        shipments['LOGNODE_LOADING'] = 0
        isLoadDC = (shipments['SEND_DC'] != -99999)
        isLoadTT = ((zones[shipments['ORIG'] - 1][:, 3] == 1) & (shipments['SEND_DC'] == -99999))
        shipments.loc[isLoadDC, 'LOGNODE_LOADING'] = 2
        shipments.loc[isLoadTT, 'LOGNODE_LOADING'] = 1

        # Is the shipment UNloaded at a distribution center?
        shipments['LOGNODE_UNLOADING'] = 0
        isReceiveDC = (shipments['RECEIVE_DC'] != -99999)
        isReceiveTT = ((zones[shipments['DEST'] - 1][:, 3] == 1) & (shipments['RECEIVE_DC'] == -99999))
        shipments.loc[isReceiveDC, 'LOGNODE_UNLOADING'] = 2
        shipments.loc[isReceiveTT, 'LOGNODE_LOADING'] = 1

        # Is the loading or unloading point in an urbanized region?
        shipments['URBAN'] = (
            (zones[shipments['ORIG'] - 1][:, 4]) |
            (zones[shipments['DEST'] - 1][:, 4]))

        # Which carrier transports which shipments, keep track of field 'CARRIER' in shipments
        shipments = assign_shipments_to_carriers(shipments, nDC, nCarriersNonDC, seeds['tour_carrier_allocation'])

        if root is not None:
            root.progressBar['value'] = 2.0

        # Extra carrierIDs for shipments transported from
        # Urban Consolidation Centers
        if varDict['LABEL'] == 'UCC':
            whereToUCC = np.where(shipments['TO_UCC'] == 1)[0]
            whereFromUCC = np.where(shipments['FROM_UCC'] == 1)[0]

            ZEZzones = set(zones[zones[:, 5] >= 1, 0])

            for i in whereToUCC:
                if zoneDict[shipments['ORIG'][i]] in ZEZzones:
                    shipments.loc[i, 'CARRIER'] = (
                        nDC +
                        nCarriersNonDC +
                        np.where(UCCzones == zoneDict[shipments['DEST'][i]])[0]
                    )

            for i in whereFromUCC:
                if zoneDict[shipments['DEST'][i]] in ZEZzones:
                    shipments.loc[i, 'CARRIER'] = (
                        nDC +
                        nCarriersNonDC +
                        np.where(UCCzones == zoneDict[shipments['ORIG'][i]])[0]
                    )

        shipments = shipments[['SHIP_ID', 'ORIG', 'DEST', 'CARRIER',
                               'VEHTYPE', 'NSTR', 'WEIGHT',
                               'LOGNODE_LOADING', 'LOGNODE_UNLOADING',
                               'URBAN', 'LS',
                               'SEND_FIRM', 'RECEIVE_FIRM',
                               'TOD_PERIOD', 'TOD_LOWER', 'TOD_UPPER']]
        shipments = np.array(shipments)

        # Divide shipments that are larger than the vehicle capacity
        # into multiple shipments
        weight = shipments[:, 6]
        vehicleType = shipments[:, 4]
        capacityExceedence = np.array([
            weight[i] / carryingCapacity[int(vehicleType[i])]
            for i in range(len(shipments))])
        whereCapacityExceeded = np.where(capacityExceedence > 1)[0]
        nNewShipmentsTotal = int(np.sum(np.ceil(capacityExceedence[whereCapacityExceeded])))

        newShipments = np.zeros((nNewShipmentsTotal, shipments.shape[1]))

        count = 0
        for i in whereCapacityExceeded:
            nNewShipments = int(np.ceil(capacityExceedence[i]))
            newWeight = weight[i] / nNewShipments
            newShipment = shipments[i, :].copy()
            newShipment[6] = newWeight

            for n in range(nNewShipments):
                newShipments[count, :] = newShipment
                count += 1

        shipments = np.append(shipments, newShipments, axis=0)
        shipments = np.delete(shipments, whereCapacityExceeded, axis=0)
        shipments[:, 0] = np.arange(len(shipments))
        shipments = shipments.astype(int)

        # Sort shipments by carrierID
        shipments = shipments[shipments[:, 3].argsort()]
        shipmentDict = dict(np.transpose(np.vstack((
            np.arange(len(shipments)),
            shipments[:, 0].copy()))))

        # Give the shipments a new shipmentID after orderingthe array
        # by carrierID
        shipments[:, 0] = np.arange(len(shipments))

        if root is not None:
            root.progressBar['value'] = 2.5

        logger.debug("\tImporting skims...")

        # Import binary skim files
        skimTravTime, skimDistance, nZones = get_skims(varDict)

        # Concatenate time and distance arrays as a 2-column matrix
        skim = np.array(np.c_[skimTravTime, skimDistance], dtype=int)

        if root is not None:
            root.progressBar['value'] = 4.0

        logger.debug("\tImporting logit parameters...")

        logitParams_ETfirst = dict(
            (str(row['parameter']), float(row['value']))
            for row in pd.read_csv(varDict['PARAMS_ET_FIRST'], sep='\t').to_dict('records')
        )
        logitParams_ETlater = dict(
            (str(row['parameter']), float(row['value']))
            for row in pd.read_csv(varDict['PARAMS_ET_LATER'], sep='\t').to_dict('records')
        )

        paramsTOD = pd.read_csv(varDict['PARAMS_TOD'], sep=',', index_col=0)
        nTOD = len([
            x for x in paramsTOD.index
            if x.split('_')[0] == 'Interval'])

        logger.debug("\tObtaining other information required before tour formation...")

        # Total number of shipments and carriers
        nShipments = len(shipments)
        nCarriers = len(np.unique(shipments[:, 3]))

        # At which index is the first shipment of each carrier
        carmarkers = [0]
        for i in range(1, nShipments):
            if shipments[i, 3] != shipments[i - 1, 3]:
                carmarkers.append(i)
        carmarkers.append(nShipments)

        # How many shipments does each carrier have?
        nShipmentsPerCarrier = [
            carmarkers[car + 1] - carmarkers[car] for car in range(nCarriers)]

        logger.debug(f"\tStart tour formation (parallelized over {nCPU} cores)")

        # Initialization of lists with information regarding tours
        # tours: here we store the shipmentIDs of each tour
        # tourSequences: here we store the order of (un)loading locations of each tour
        tours = [[] for car in range(nCarriers)]
        tourSequences = [[] for car in range(nCarriers)]

        if root is not None:
            root.progressBar['value'] = 5.0

        # Bepaal het aantal CPUs dat we gebruiken en welke CPU
        # welke carriers doet
        chunks = [np.arange(nCarriers)[i::nCPU] for i in range(nCPU)]

        # Voer de tourformatie uit
        with mp.Pool(nCPU) as p:
            tourformationResult = p.map(functools.partial(
                form_tours,
                carmarkers,
                shipments,
                skim,
                nZones,
                maxNumShips,
                carryingCapacity,
                dcZones,
                nShipmentsPerCarrier,
                nCarriers,
                nTOD,
                nNSTR,
                logitParams_ETfirst,
                logitParams_ETlater,
                seeds['tour_formation'],
            ), chunks)

        if root is not None:
            root.progressBar['value'] = 80.0

        # Pak de tourformatieresultaten uit
        for cpu in range(nCPU):
            for car in chunks[cpu]:
                tours[car] = tourformationResult[cpu][0][car]
                tourSequences[car] = tourformationResult[cpu][1][car]

        nTours = dict((car, len(tours[car])) for car in range(nCarriers))

        logger.debug("\t\tTour formation completed for all carriers")

        if root is not None:
            root.progressBar['value'] = 81.0

        logger.debug("\tAdding empty trips...")

        # Add empty trips
        emptytripadded = [
            [None for tour in range(nTours[car])]
            for car in range(nCarriers)]

        for car in range(nCarriers):
            for tour in range(nTours[car]):
                tourSequences[car][tour] = [
                    x for x in tourSequences[car][tour] if x != 0]

                # If tour does not end at start location
                if tourSequences[car][tour][0] != tourSequences[car][tour][-1]:
                    tourSequences[car][tour].append(tourSequences[car][tour][0])
                    emptytripadded[car][tour] = True
                else:
                    emptytripadded[car][tour] = False

        if root is not None:
            root.progressBar['value'] = 82.0

        logger.debug("\tObtaining number of shipments and trip weights...")

        # Number of shipments and trips of each tour
        nShipmentsPerTour = [
            [len(tours[car][tour]) for tour in range(nTours[car])]
            for car in range(nCarriers)]
        nTripsPerTour = [
            [len(tourSequences[car][tour]) - 1 for tour in range(nTours[car])]
            for car in range(nCarriers)]
        nTripsTotal = np.sum(np.sum(np.array(nTripsPerTour, dtype=object)))

        # Weight per trip
        tripWeights = [
            [np.zeros(nTripsPerTour[car][tour], dtype=int) for tour in range(nTours[car])]
            for car in range(nCarriers)]

        # Time windows of arrival at the destination of each trip
        tripLowerWindow = [
            [-1 * np.ones(nTripsPerTour[car][tour], dtype=int) for tour in range(nTours[car])]
            for car in range(nCarriers)]
        tripUpperWindow = [
            [1000 * np.ones(nTripsPerTour[car][tour], dtype=int) for tour in range(nTours[car])]
            for car in range(nCarriers)]

        for car in range(nCarriers):
            for tour in range(nTours[car]):

                nShipsThisTour = nShipmentsPerTour[car][tour]
                shipmentIsLoaded = [False for i in range(nShipsThisTour)]
                shipmentIsUnloaded = [False for i in range(nShipsThisTour)]
                
                timeOfDayPeriodsInTour = np.unique(shipments[
                    tours[car][tour], 13])
                tourPastMidnight = (
                    0 in timeOfDayPeriodsInTour and
                    (nTOD - 1) in timeOfDayPeriodsInTour)

                for trip in range(nTripsPerTour[car][tour]):
                    # Startzone of trip
                    orig = tourSequences[car][tour][trip]

                    # If it's the first trip of the tour,
                    # then initialize a counter to add and subtract weight from
                    if trip == 0:
                        tmpTripWeight = 0

                    for i in range(nShipsThisTour):
                        ship = tours[car][tour][i]

                        # If loading location of the shipment was the
                        # startpoint of the trip and shipment has not been
                        # loaded/unloaded yet
                        addWeightToTrip = (
                            shipments[ship][1] == orig and not
                            (shipmentIsLoaded[i] or shipmentIsUnloaded[i]))

                        # Add weight of shipment to counter
                        if addWeightToTrip:
                            tmpTripWeight += shipments[ship][6]
                            shipmentIsLoaded[i] = True

                        # If unloading location of the shipment was the
                        # startpoint of the trip and shipment has not been
                        # unloaded yet
                        subtractWeightFromTrip = (
                            shipments[ship][2] == orig and
                            shipmentIsLoaded[i] and not
                            shipmentIsUnloaded[i])

                        # Remove weight of shipment from counter
                        # and update delivery time window
                        if subtractWeightFromTrip:
                            tmpTripWeight -= shipments[ship][6]
                            shipmentIsUnloaded[i] = True

                            shipPeriod = shipments[ship][13]
                            shipLower = shipments[ship][14]
                            shipUpper = shipments[ship][15]
                            tripLower = tripLowerWindow[car][tour][trip]
                            tripUpper = tripUpperWindow[car][tour][trip]
                            
                            if shipPeriod == 0 and tourPastMidnight:
                                shipLower += 24
                                shipUpper += 24

                            # Update lower end of delivery time window
                            if tripLower < 0 or shipLower < tripLower:
                                tripLowerWindow[car][tour][trip] = shipLower

                            # Update upper end of delivery time window
                            if tripUpper > 999 or shipUpper > tripUpper:
                                tripUpperWindow[car][tour][trip] = shipUpper

                    tripWeights[car][tour][trip] = tmpTripWeight

        if root is not None:
            root.progressBar['value'] = 84.0

        logger.debug("\tDetermining departure times...")

        # Initialize arrays for departure time of each tour
        depTimeTour = [
            [None for tour in range(nTours[car])]
            for car in range(nCarriers)]

        # Initialize arrays for departure and arrival time of each trip
        depTimeTrip = [
            [np.zeros(nTripsPerTour[car][tour]) for tour in range(nTours[car])]
            for car in range(nCarriers)]
        arrTimeTrip = [
            [np.zeros(nTripsPerTour[car][tour]) for tour in range(nTours[car])]
            for car in range(nCarriers)]

        for car in range(nCarriers):
            for tour in range(nTours[car]):
                tmpTimeEarly = [None for i in range(nTripsPerTour[car][tour])]
                tmpTimeLate = [None for i in range(nTripsPerTour[car][tour])]

                # Get initial deviation from time windows if the tour were to
                # depart at 00:00 midnight
                for trip in range(nTripsPerTour[car][tour]):
                    # Travel time to destination of the current trip
                    travTime = get_traveltime(
                        tourSequences[car][tour][trip],
                        tourSequences[car][tour][trip + 1],
                        skim,
                        nZones)
                    
                    # Time spent at the destination of the current trip
                    np.random.seed(seeds['tour_departure_time'] + 10000 * car + tour)
                    dwellTime = avgDwellTime * 2 * np.random.rand()

                    # Arrival time at the destination of the current trip
                    tmpArrTimeTrip = (
                        depTimeTrip[car][tour][trip] +
                        travTime)
                    arrTimeTrip[car][tour][trip] = tmpArrTimeTrip

                    # Departure time from the destination of the current trip
                    if (trip + 1) < nTripsPerTour[car][tour]:
                        depTimeTrip[car][tour][trip + 1] = (
                            tmpArrTimeTrip + dwellTime)

                    if tripLowerWindow[car][tour][trip] > -1:
                        tmpTimeEarly[trip] = (
                            tripLowerWindow[car][tour][trip] -
                            tmpArrTimeTrip)

                    if tripUpperWindow[car][tour][trip] < 1000:
                        tmpTimeLate[trip] = (
                            tripUpperWindow[car][tour][trip] -
                            tmpArrTimeTrip)

                # Determine the optimal tour departure time
                # (minimal deviation from time windows)
                tmpTimeEarly = [x for x in tmpTimeEarly if x is not None]
                tmpTimeLate = [x for x in tmpTimeLate if x is not None]
                avgTimeEarly = (
                    np.average(tmpTimeEarly) if len(tmpTimeEarly) > 0 else 0.0)
                avgTimeLate = (
                    np.average(tmpTimeLate) if len(tmpTimeLate) > 0 else 0.0)

                depTimeTour[car][tour] = np.average([avgTimeEarly, avgTimeLate])

                if depTimeTour[car][tour] < 0:
                    depTimeTour[car][tour] += 24

                # Update the trip departure and arrival times based on this
                # tour departure time
                depTimeTrip[car][tour] += depTimeTour[car][tour]
                arrTimeTrip[car][tour] += depTimeTour[car][tour]

        if root is not None:
            root.progressBar['value'] = 85.0

        logger.debug("\tDetermining combustion type of tours...")

        combTypeTour = [
            [None for tour in range(nTours[car])]
            for car in range(nCarriers)]

        if varDict['LABEL'] == 'UCC':
            id_fuel = dims.get_id_from_label("combustion_type", "Fuel")
            id_hybrid = dims.get_id_from_label("combustion_type", "Hybrid (electric)")

            ZEZzones = set(zones[zones[:, 5] >= 1, 0])

            for car in range(nCarriers):
                for tour in range(nTours[car]):
                    ls = shipments[tours[car][tour][0], 10]
                    vt = shipments[tours[car][tour][0], 4]

                    tmpSeed = seeds['tour_zez_combustion'] + 10000 * car + tour

                    # Combustion type for tours from/to UCCs
                    if car >= nDC + nCarriersNonDC:
                        combTypeTour[car][tour] = draw_choice_mcs(cumProbCombustion[(ls, vt)], tmpSeed)

                    else:
                        inZEZ = [
                            zoneDict[x] in ZEZzones
                            for x in np.unique(tourSequences[car][tour])
                            if x != 0]

                        # Combustion type for tours within ZEZ (but not from/to UCCs)
                        if np.all(inZEZ):
                            combTypeTour[car][tour] = draw_choice_mcs(cumProbCombustion[(ls, vt)], tmpSeed)

                        # Hybrids for tours directly entering/leaving the ZEZ
                        elif np.any(inZEZ):
                            combTypeTour[car][tour] = id_hybrid

                        # Fuel for all other tours that do not go to the ZEZ
                        else:
                            combTypeTour[car][tour] = id_fuel

        # In the not-UCC scenario everything is assumed to be fuel
        if varDict['LABEL'] == 'REF':
            for car in range(nCarriers):
                for tour in range(nTours[car]):
                    combTypeTour[car][tour] = 0

        # Unless the shift-to-electric or shift-to-hydrogen
        # parameter is set (SHIFT_FREIGHT_TO_COMB1 or _COMB2)
        np.random.seed(seeds['tour_shift_combustion'])

        if doShiftToElectric or doShiftToHydrogen:
            for car in range(nCarriers):
                for tour in range(nTours[car]):
                    if doShiftToElectric:
                        if np.random.rand() <= shiftFreightToComb1:
                            combTypeTour[car][tour] = 1
                    if doShiftToHydrogen:
                        if combTypeTour[car][tour] != 1:
                            if np.random.rand() <= shiftFreightToComb2:
                                combTypeTour[car][tour] = 2

        if root is not None:
            root.progressBar['value'] = 86.0

        # ---------------------------- Create Tours CSV ----------------------

        logger.debug("\tWriting tour data to CSV...")

        outputTours = np.zeros((nTripsTotal, 20), dtype=object)
        tripcount = 0

        for car in range(nCarriers):
            for tour in range(nTours[car]):
                for trip in range(nTripsPerTour[car][tour]):

                    # CarrierID, tourID, tripID
                    outputTours[tripcount][0] = car
                    outputTours[tripcount][1] = tour
                    outputTours[tripcount][2] = trip

                    # Origin and destination zone number
                    outputTours[tripcount][[3, 4]] = (
                        zoneDict[tourSequences[car][tour][trip]],
                        zoneDict[tourSequences[car][tour][trip + 1]])

                    # X coordinates of origin and destination
                    outputTours[tripcount][[5, 6]] = (
                        zones[tourSequences[car][tour][trip] - 1][1],
                        zones[tourSequences[car][tour][trip + 1] - 1][1])

                    # Y coordinates of origin and destination
                    outputTours[tripcount][[7, 8]] = (
                        zones[tourSequences[car][tour][trip] - 1][2],
                        zones[tourSequences[car][tour][trip + 1] - 1][2])

                    # Vehicle type
                    outputTours[tripcount][9] = (
                        shipments[tours[car][tour][0], 4])

                    # Dominant NSTR goods type (by weight)
                    outputTours[tripcount][10] = max_nstr(tours[car][tour], shipments, nNSTR)

                    # Number of shipments transported in tour
                    outputTours[tripcount][11] = nShipmentsPerTour[car][tour]

                    # ID for DC zones
                    outputTours[tripcount][12] = (
                        zoneDict[dcZones[outputTours[tripcount][0]]]
                        if car < nDC else -99999)

                    # Weight: sum of tour and for individual trip
                    outputTours[tripcount][13] = sum_weight(tours[car][tour], shipments)
                    outputTours[tripcount][14] = tripWeights[car][tour][trip] / 1000

                    # Departure time of the tour
                    outputTours[tripcount][15] = depTimeTour[car][tour]

                    # Departure and arrival time of the trip
                    outputTours[tripcount][16] = depTimeTrip[car][tour][trip]
                    outputTours[tripcount][17] = arrTimeTrip[car][tour][trip]

                    # Logistic segment of tour
                    outputTours[tripcount][18] = shipments[tours[car][tour][0], 10]

                    # Combustion type of the vehicle
                    outputTours[tripcount][19] = combTypeTour[car][tour]

                    tripcount += 1

                # NSTR of empty trips is -1
                if emptytripadded[car][tour]:
                    outputTours[tripcount - 1][10] = -1

        nTripsTotal = tripcount

        if root is not None:
            root.progressBar['value'] = 87.0

        # Create DataFrame object for easy formatting and exporting to csv
        columns = [
            "CARRIER_ID", "TOUR_ID", "TRIP_ID",
            "ORIG", "DEST",
            "X_ORIG", "X_DEST", "Y_ORIG", "Y_DEST",
            "VEHTYPE",
            "NSTR",
            "N_SHIP",
            "DC_ID",
            "TOUR_WEIGHT", "TRIP_WEIGHT",
            "TOUR_DEPTIME", "TRIP_DEPTIME", "TRIP_ARRTIME",
            "LS",
            "COMBTYPE"]
        dTypes = [
            str, str, str,
            int, int,
            float, float,  float,  float,
            int,
            int,
            int,
            int,
            float, float,
            int, float, float,
            int,
            int]

        outputTours = pd.DataFrame(outputTours, columns=columns)
        for i in range(len(outputTours.columns)):
            outputTours.iloc[:, i] = outputTours.iloc[:, i].astype(dTypes[i])

        outputTours.to_csv(
            f"{varDict['OUTPUTFOLDER']}Tours_{varDict['LABEL']}.csv",
            index=None,
            header=True)

        logger.debug(f"\tCSV written to {varDict['OUTPUTFOLDER']}Tours_{varDict['LABEL']}.csv")

        if root is not None:
            root.progressBar['value'] = 89.0

        # ----------------------- Enrich Shipments CSV ------------------------

        logger.debug(f"\tEnriching Shipments CSV...")

        shipmentTourID = {}
        for car in range(nCarriers):
            for tour in range(nTours[car]):
                for ship in range(len(tours[car][tour])):
                    shipID = tours[car][tour][ship]
                    shipmentTourID[shipmentDict[shipID]] = f'{car}_{tour}'

        # Add TOUR_ID to shipments csv and export again
        shipments = pd.DataFrame(shipments)
        shipments.columns = [
            'SHIP_ID',
            'ORIG', 'DEST',
            'CARRIER',
            'VEHTYPE',
            'NSTR',
            'WEIGHT',
            'LOGNODE_LOADING', 'LOGNODE_UNLOADING',
            'URBAN',
            'LS',
            'SEND_FIRM', 'RECEIVE_FIRM',
            'TOD_PERIOD', 'TOD_LOWER', 'TOD_UPPER']

        shipments['SHIP_ID'] = [shipmentDict[x] for x in shipments['SHIP_ID']]
        shipments['TOUR_ID'] = [shipmentTourID[x] for x in shipments['SHIP_ID']]
        shipments['ORIG'] = [zoneDict[x] for x in shipments['ORIG']]
        shipments['DEST'] = [zoneDict[x] for x in shipments['DEST']]
        shipments['WEIGHT'] /= 1000

        selectCols = [
            'SHIP_ID',
            'ORIG', 'DEST',
            'CARRIER',
            'VEHTYPE',
            'NSTR',
            'WEIGHT',
            'LS',
            'TOUR_ID',
            'SEND_FIRM', 'RECEIVE_FIRM',
            'TOD_PERIOD']
        shipments = shipments[selectCols]
        shipments = shipments.sort_values('SHIP_ID')
        filename = (
            varDict['OUTPUTFOLDER'] +
            "Shipments_AfterScheduling_" +
            varDict['LABEL'] +
            '.csv')
        shipments.to_csv(filename, index=False)

        if root is not None:
            root.progressBar['value'] = 90.0

        # -------------------------- Write GeoJSON ----------------------------

        logger.debug(f"\tWriting Shapefile...")

        percStart = 90.0
        percEnd = 97.0

        Ax = list(outputTours['X_ORIG'])
        Ay = list(outputTours['Y_ORIG'])
        Bx = list(outputTours['X_DEST'])
        By = list(outputTours['Y_DEST'])

        # Initialize shapefile fields
        w = shp.Writer(
            varDict['OUTPUTFOLDER'] +
            'Tours_' +
            varDict['LABEL'] +
            '.shp')
        w.field('CARRIER_ID',   'N', size=5, decimal=0)
        w.field('TOUR_ID',      'N', size=5, decimal=0)
        w.field('TRIP_ID',      'N', size=3, decimal=0)
        w.field('ORIG',         'N', size=8, decimal=0)
        w.field('DEST',         'N', size=8, decimal=0)
        w.field('VEHTYPE',      'N', size=2, decimal=0)
        w.field('NSTR',         'N', size=2, decimal=0)
        w.field('N_SHIP',       'N', size=3, decimal=0)
        w.field('DC_ID',        'N', size=8, decimal=0)
        w.field('TOUR_WEIGHT',  'N', size=5, decimal=2)
        w.field('TRIP_WEIGHT',  'N', size=5, decimal=2)
        w.field('TOUR_DEPTIME', 'N', size=4, decimal=2)
        w.field('TRIP_DEPTIME', 'N', size=5, decimal=2)
        w.field('TRIP_ARRTIME', 'N', size=5, decimal=2)
        w.field('LS',           'N', size=2, decimal=0)
        w.field('COMBTYPE',     'N', size=2, decimal=0)

        dbfData = np.array(
            outputTours.drop(columns=['X_ORIG', 'Y_ORIG', 'X_DEST', 'Y_DEST']),
            dtype=object)

        for i in range(nTripsTotal):
            # Add geometry
            w.line([[[Ax[i], Ay[i]], [Bx[i], By[i]]]])

            # Add data fields
            w.record(*dbfData[i, :])

            if i % 500 == 0:
                print(f"\t{round(i / nTripsTotal * 100, 1)}%", end='\r')

                if root is not None:
                    root.progressBar['value'] = (
                        percStart +
                        (percEnd - percStart) * i / nTripsTotal)

        w.close()

        logger.debug(f"\tTours written to {varDict['OUTPUTFOLDER']}Tours_{varDict['LABEL']}.shp")

        # ----------- Create trip matrices for traffic assignment -------------

        logger.debug("\tGenerating trip matrix...")

        cols = ['ORIG', 'DEST',
                'N_LS0', 'N_LS1', 'N_LS2', 'N_LS3',
                'N_LS4', 'N_LS5', 'N_LS6', 'N_LS7',
                'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3', 'N_VEH4',
                'N_VEH5', 'N_VEH6', 'N_VEH7', 'N_VEH8', 'N_VEH9',
                'N_TOT']

        # Maak dummies in tours variabele per logistiek segment,
        # voertuigtype en N_TOT (altijd 1 hier)
        for ls in range(nLS):
            outputTours['N_LS' + str(ls)] = (
                outputTours['LS'] == ls).astype(int)
        for vt in range(nVT):
            outputTours['N_VEH' + str(vt)] = (
                outputTours['VEHTYPE'] == vt).astype(int)
        outputTours['N_TOT'] = 1

        # Gebruik deze dummies om het aantal ritten per HB te bepalen
        # voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(
            outputTours,
            values=cols[2:],
            index=['ORIG', 'DEST'],
            aggfunc=np.sum)
        pivotTable['ORIG'] = [x[0] for x in pivotTable.index]
        pivotTable['DEST'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]

        pivotTable.to_csv(
            varDict['OUTPUTFOLDER'] + f"tripmatrix_{varDict['LABEL']}.txt",
            index=False,
            sep='\t')

        logger.debug(f"\tTrip matrix written to {varDict['OUTPUTFOLDER']}tripmatrix_{varDict['LABEL']}.txt")

        if root is not None:
            root.progressBar['value'] = 98.0

        tours = pd.read_csv(
            varDict['OUTPUTFOLDER'] + f"Tours_{varDict['LABEL']}.csv")
        tours.loc[tours['TRIP_DEPTIME'] > 24, 'TRIP_DEPTIME'] -= 24
        tours.loc[tours['TRIP_DEPTIME'] > 24, 'TRIP_DEPTIME'] -= 24

        for tod in range(24):
            logger.debug(f"\t\tAlso generating trip matrix for TOD {tod}...")

            output = tours[
                (tours['TRIP_DEPTIME'] >= tod) &
                (tours['TRIP_DEPTIME'] < tod + 1)].copy()

            # Maak dummies in tours variabele
            # per logistiek segment, voertuigtypeen N_TOT (altijd 1 hier)
            for ls in range(nLS):
                output['N_LS' + str(ls)] = (
                    output['LS'] == ls).astype(int)
            for vt in range(nVT):
                output['N_VEH' + str(vt)] = (
                    output['VEHTYPE'] == vt).astype(int)
            output['N_TOT'] = 1

            # Gebruik deze dummies om het aantal ritten per HB te bepalen
            # voor elk logistiek segment, voertuigtype en totaal
            pivotTable = pd.pivot_table(
                output,
                values=cols[2:],
                index=['ORIG', 'DEST'],
                aggfunc=np.sum)
            pivotTable['ORIG'] = [x[0] for x in pivotTable.index]
            pivotTable['DEST'] = [x[1] for x in pivotTable.index]
            pivotTable = pivotTable[cols]

            filename = (
                str(varDict['OUTPUTFOLDER']) +
                "tripmatrix_" +
                str(varDict['LABEL']) +
                "_TOD" +
                str(tod) +
                ".txt")
            pivotTable.to_csv(filename, index=False, sep='\t')

            if root is not None:
                root.progressBar['value'] = (
                    98.0 +
                    (100.0 - 98.0) * (tod + 1) / 24)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
