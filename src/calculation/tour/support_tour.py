import numpy as np
import pandas as pd

from itertools import product
from typing import Dict, Tuple

from calculation.common.dimensions import ModelDimensions


def get_cum_shares_comb_ucc(varDict: Dict[str, str], dims: ModelDimensions) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Returns the cumulative shares of combustion types for the switch to UCCs.
    """
    cumProbCombustion = dict(
        ((ls, vt), np.zeros(len(dims.combustion_type)))
        for ls, vt in product(dims.logistic_segment.keys(), dims.vehicle_type.keys())
    )

    for row in pd.read_csv(varDict["ZEZ_SCENARIO"], sep='\t').to_dict('records'):
        cumProbCombustion[
            (int(row['logistic_segment']), int(row['vehicle_type']))
        ][int(row['combustion_type'])] = float(row['share_combustion'])

    for ls, vt in product(dims.logistic_segment.keys(), dims.vehicle_type.keys()):
        if np.sum(cumProbCombustion[(ls, vt)]) != 0:
            cumProbCombustion[(ls, vt)] = np.cumsum(cumProbCombustion[(ls, vt)]) / np.sum(cumProbCombustion[(ls, vt)])
        else:
            cumProbCombustion[(ls, vt)] = np.arange(1, len(dims.combustion_type) + 1) / len(dims.combustion_type)

    return cumProbCombustion


def assign_shipments_to_carriers(
    shipments: pd.DataFrame,
    nDC: int,
    nCarriersNonDC: int,
    id_parcel_consolidated: int,
    seed: int
) -> pd.DataFrame:
    """Fills the field 'CARRIER' for the shipments DataFrame."""
    # Determine the carrier ID for each shipment
    # Shipments are first grouped by DC (one carrier per DC assumed, carrierID = 0 to nDC)
    shipments['CARRIER'] = 0
    whereLoadDC = (shipments['SEND_DC'] != -99999) & (shipments['LS'] != id_parcel_consolidated)
    whereUnloadDC = (shipments['RECEIVE_DC'] != -99999) & (shipments['LS'] != id_parcel_consolidated)

    # Shipments not loaded or unloaded at DC are randomly assigned
    # to the other carriers, carrierID = nDC to nDC + nCarriersNonDC]
    whereBothDC = (whereLoadDC) & (whereUnloadDC)
    whereNoDC = ~(whereLoadDC) & ~(whereUnloadDC)

    np.random.seed(seed)

    shipments.loc[whereLoadDC, 'CARRIER'] = shipments['SEND_DC'][whereLoadDC]
    shipments.loc[whereUnloadDC, 'CARRIER'] = shipments['RECEIVE_DC'][whereUnloadDC]
    shipments.loc[whereBothDC, 'CARRIER'] = [
        [shipments['SEND_DC'][i], shipments['RECEIVE_DC'][i]][np.random.randint(0, 2)]
        for i in shipments.loc[whereBothDC, :].index]
    shipments.loc[whereNoDC, 'CARRIER'] = nDC + np.random.randint(0, nCarriersNonDC, np.sum(whereNoDC))
    
    return shipments


def get_traveltime(orig, dest, skim, nZones):
    '''
    Obtain the travel time [h] for orig to a destination zone.
    '''
    return skim[(orig - 1) * nZones + (dest - 1)][0] / 3600


def get_distance(orig, dest, skim, nZones):
    '''
    Obtain the distance [km] for orig to a destination zone.
    '''
    return skim[(orig - 1) * nZones + (dest - 1)][1] / 1000


def nearest_neighbor(ships, TODs, dc, skim, nZones, nTOD):
    '''
    Creates a tour sequence to visit all loading and unloading locations.
    First a nearest-neighbor search is applied,then a 2-opt posterior
    improvement phase.

    ships = array with loading locations of shipments in column 0,
        and unloading locations in column 1
    dc = zone ID of the DC in case the carrier is located in a DC zone
    '''
    # ------------------------ Initialization -------------------------------
    # Arrays of unique loading and unloading locations to be visited
    loading = np.unique(ships[:, 0])
    unloading = np.unique(ships[:, 1])

    # Total number of loading and unloading locations to visit
    nLoad = len(loading)
    nUnload = len(unloading)

    # Here we will store the sequence of locations
    tourSequence = np.zeros(nLoad + nUnload, dtype=int)

    # Codes of the first and last period of a day
    firstCategoryTOD = 0
    lastCategoryTOD = (nTOD - 1)

    # Earliest and latest time-of-day in the tour
    if (firstCategoryTOD in TODs) and (lastCategoryTOD in TODs):
        firstTourTOD = lastCategoryTOD
        lastTourTOD = firstCategoryTOD
    else:
        firstTourTOD = np.min(TODs)
        lastTourTOD = np.max(TODs)
    allSameTOD = (firstTourTOD == lastTourTOD)

    # Which zones need to be visited in the earliest time-of-day of the tour
    unloadingFirstTOD = np.unique([
        ships[i, 1] for i in range(len(ships))
        if TODs[i] == firstTourTOD])
    nUnloadFirstTOD = len(unloadingFirstTOD)

    # Which zones need to be visited in the last time-of-day of the tour
    if not allSameTOD:
        unloadingLastTOD = np.unique([
            ships[i, 1] for i in range(len(ships))
            if TODs[i] == lastTourTOD and ships[i, 1] not in unloadingFirstTOD])
        nUnloadLastTOD = len(unloadingLastTOD)

    # ----------------------- Loading sequence -------------------------------
    # First visited location = first listed location
    tourSequence[0] = ships[0, 0]

    # This location has already been visited,
    # remove from list of remaining loading locations
    loading = loading[loading != ships[0, 0]]

    # For each shipment (except the last), decide the one that is
    # visited for loading
    for currentship in range(nLoad - 1):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(loading))

        # Go over all remaining loading locations
        for remainship in range(len(loading)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = get_traveltime(
                tourSequence[currentship],
                loading[remainship],
                skim,
                nZones)

        # Index of the nearest unvisited loading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[currentship + 1] = loading[nearestShipment]

        # Remove this shipment from list of remaining loading locations
        loading = np.delete(loading, nearestShipment)

    # --------------- Unloading sequence (first time-of-day) ------------------
    for currentship in range(nUnloadFirstTOD):

        # (Re)initialize array with travel times from current to
        # each remaining shipment
        timeCurrentToRemaining = np.zeros(len(unloadingFirstTOD))

        # Go over all remaining unloading locations
        for remainship in range(len(unloadingFirstTOD)):

            # Fill in travel time current location to
            # each remaining loading location
            timeCurrentToRemaining[remainship] = get_traveltime(
                tourSequence[nLoad - 1 + currentship],
                unloadingFirstTOD[remainship],
                skim,
                nZones)

        # Index of the nearest unvisited unloading location
        nearestShipment = np.argmin(timeCurrentToRemaining)

        # Fill in as the next location in tour sequence
        tourSequence[nLoad + currentship] = unloadingFirstTOD[nearestShipment]

        # Remove this shipment from list of remaining unloading locations
        unloadingFirstTOD = np.delete(unloadingFirstTOD, nearestShipment)

    # --------------- Unloading sequence (last time-of-day) -------------------
    if not allSameTOD:
        for currentship in range(nUnloadLastTOD):
            timeCurrentToRemaining = np.zeros(len(unloadingLastTOD))
    
            for remainship in range(len(unloadingLastTOD)):
                timeCurrentToRemaining[remainship] = get_traveltime(
                    tourSequence[nLoad + nUnloadFirstTOD - 1 + currentship],
                    unloadingLastTOD[remainship],
                    skim,
                    nZones)
    
            nearestShipment = np.argmin(timeCurrentToRemaining)
    
            tourSequence[nLoad + nUnloadFirstTOD + currentship] = (
                    unloadingLastTOD[nearestShipment])
    
            unloadingLastTOD = np.delete(unloadingLastTOD, nearestShipment)

    # ---------- Some restructuring of tourSequence variable ------------------

    tourSequence = list(tourSequence)

    # If the carrier is at a DC, start the tour here
    # (not always necessarily the first loading location)
    if dc is not None:
        if tourSequence[0] != dc:
            tourSequence.insert(0, int(dc))

    # Make sure the tour does not visit the homebase in between
    # (otherwise it's not 1 tour but 2 tours)
    nStartLoc = set(np.where(np.array(tourSequence) == tourSequence[0])[0][1:])
    if len(nStartLoc) > 1:
        tourSequence = [
            tourSequence[x]
            for x in range(len(tourSequence))
            if x not in nStartLoc]
        tourSequence.append(tourSequence[0])
    lenTourSequence = len(tourSequence)

    # ------------------ 2-opt posterior improvement ------------------
    # Only do 2-opt if tour has more than 3 stops
    if lenTourSequence > 3:
        startLocations = np.array(tourSequence[:-1]) - 1
        endLocations = np.array(tourSequence[1:]) - 1
        tourDuration = np.sum(
            skim[startLocations * nZones + endLocations, 0]) / 3600

        for shiftLocA in range(1, lenTourSequence - 1):
            for shiftLocB in range(1, lenTourSequence - 1):
                if shiftLocA != shiftLocB:
                    swappedTourSequence = tourSequence.copy()
                    swappedTourSequence[shiftLocA] = tourSequence[shiftLocB]
                    swappedTourSequence[shiftLocB] = tourSequence[shiftLocA]

                    swappedStartLocations = np.array(swappedTourSequence[:-1]) - 1
                    swappedEndLocations = np.array(swappedTourSequence[1:]) - 1
                    swappedTourDuration = np.sum(
                        skim[swappedStartLocations * nZones + swappedEndLocations, 0]) / 3600

                    # Only make the swap definitive
                    # if it reduces the tour duration
                    if swappedTourDuration < tourDuration:

                        # Check if the loading locations are visited
                        # before the unloading locations
                        precedence = [None for i in range(len(ships))]
                        for i in range(len(ships)):
                            load, unload = ships[i, 0], ships[i, 1]
                            precedence[i] = (
                                np.where(swappedTourSequence == unload)[0][-1] >
                                np.where(swappedTourSequence == load)[0][0])

                        if np.all(precedence):
                            tourSequence = swappedTourSequence.copy()
                            tourDuration = swappedTourDuration

    return np.array(tourSequence, dtype=int)


def tourdur(tourSequence, skim, nZones):
    '''
    Calculates the tour duration of the tour (so far)

    tourSequence = array with ordered sequence of visited locations in tour
    '''
    tourSequence = np.array(tourSequence)
    tourSequence = tourSequence[tourSequence != 0]

    startLocations = tourSequence[:-1] - 1
    endLocations = tourSequence[1:] - 1

    return np.sum(skim[startLocations * nZones + endLocations, 0]) / 3600


def tourdist(tourSequence, skim, nZones):
    '''
    Calculates the tour distance of the tour (so far)

    tourSequence = array with ordered sequence of visited locations in tour
    '''
    tourSequence = np.array(tourSequence)
    tourSequence = tourSequence[tourSequence != 0]

    startLocations = tourSequence[:-1] - 1
    endLocations = tourSequence[1:] - 1

    return np.sum(skim[startLocations * nZones + endLocations, 1]) / 1000


def cap_utilization(veh, weights, carryingCapacity):
    '''
    Calculates the capacity utilzation of the tour (so far).
    Assume that vehicle type chosen for 1st shipment in tour
    defines the carrying capacity.
    '''
    return sum(weights) / carryingCapacity[veh]


def proximity(tourlocs, universal, skim, nZones):
    '''
    Reports the proximity value [km] of each shipment in
    the universal choice set

    tourlocs = array with the locations visited in the tour so far
    universal = the universal choice set
    '''
    # Unique locations visited in the tour so far
    # (except for tour starting point)
    if tourlocs[0, 0] == tourlocs[0, 1]:
        tourlocs = [x for x in np.unique(tourlocs) if x != 0]
    else:
        tourlocs = [x for x in np.unique(tourlocs) if x != 0 and x != tourlocs[0, 0]]

    # Loading and unloading locations of the remaining shipments
    otherShipments = universal[:, 1:3].astype(int)
    nOtherShipments = len(otherShipments)

    # Initialization
    distancesLoading = np.zeros((len(tourlocs), nOtherShipments))
    distancesUnloading = np.zeros((len(tourlocs), nOtherShipments))

    for i in range(len(tourlocs)):
        distancesLoading[i, :] = skim[
            (tourlocs[i] - 1) * nZones + (otherShipments[:, 0] - 1), 1] / 1000
        distancesUnloading[i, :] = skim[
            (tourlocs[i] - 1) * nZones + (otherShipments[:, 1] - 1), 1] / 1000

    # Proximity measure = distance to nearest loading and unloading
    # location summed up
    distances = np.min(distancesLoading + distancesUnloading, axis=0)

    return distances


def lognode_loading(tour, shipments):
    '''
    Returns a Boolean that states whether a logistical node is visited in
    the tour for loading
    '''
    if len(tour) == 1:
        return shipments[tour][0][7] == 2
    else:
        return np.any(shipments[tour][0][7] == 2)


def lognode_unloading(tour, shipments):
    '''
    Returns a Boolean that states whether a logistical node is visited in
    the tour for unloading
    '''
    if len(tour) == 1:
        return shipments[tour][0][8] == 2
    else:
        return np.any(shipments[tour][0][8] == 2)


def transship(tour, shipments):
    '''
    Returns a Boolean that states whether a transshipment zone is visited in
    the tour
    '''
    if len(tour) == 1:
        return (
            shipments[tour][0][7] == 1 or
            shipments[tour][0][8] == 1)
    else:
        return (
            np.any(shipments[tour][0][7] == 1) or
            np.any(shipments[tour][0][8] == 1))


def urbanzone(tour, shipments):
    '''
    Returns a Boolean that states whether an urban zone is visited in the tour
    '''
    if len(tour) == 1:
        return shipments[tour][0][9]
    else:
        return np.any(shipments[tour][0][9] == 1)


def is_concrete():
    '''
    Returns a Boolean that states whether concrete is transported in the tour.
    This was part of the estimated ET model but is not modelled in the TFS.
    '''
    return False


def max_nstr(tour, shipments, nNSTR):
    '''
    Returns the NSTR goods type (0-9) of which the highest weight is
    transported in the tour (so far)

    tour = array with the IDs of all shipments in the tour
    '''
    nstrWeight = np.zeros(nNSTR)

    for i in range(len(tour)):
        shipNSTR = shipments[tour[i], 5]
        shipWeight = shipments[tour[i], 6]
        nstrWeight[shipNSTR] += shipWeight

    return np.argmax(nstrWeight)


def sum_weight(tour, shipments):
    '''
    Returns the weight of all goods that are transported in the tour

    tour = array with the IDs of all shipments in the tour
    '''
    sumWeight = 0

    for i in range(len(tour)):
        # The weights of the shipments in the tour are summed up
        # and converted to tonnes
        shipWeight = shipments[tour[i], 6]
        sumWeight += (shipWeight / 1000)

    return sumWeight


def endtour_first(tourDuration, capUt, tour, params, shipments, nNSTR):
    '''
    Returns True if we decide to end the tour, False if we decide
    to add another shipment
    '''
    # Calculate explanatory variables
    vehicleTypeIs0 = (shipments[tour[0], 4] == 0) * 1
    vehicleTypeIs1 = (shipments[tour[0], 4] == 1) * 1
    maxNSTR = max_nstr(tour, shipments, nNSTR)

    # Calculate utility
    etUtility = (
        params['ASC_ET'] +
        params['Tour duration'] * tourDuration**0.5 +
        params['Capacity utilization'] * capUt**2 +
        params['Transshipment'] * transship(tour, shipments) +
        params['DC loading'] * lognode_loading(tour, shipments) +
        params['DC unloading'] * lognode_unloading(tour, shipments) +
        params['Urban'] * urbanzone(tour, shipments) +
        params['VT0'] * vehicleTypeIs0 +
        params['VT1'] * vehicleTypeIs1 +
        params['NSTR0'] * (maxNSTR == 0) +
        params['NSTR1'] * (maxNSTR == 1) +
        params['NSTR2to5'] * (maxNSTR in [2, 3, 4, 5]) +
        params['NSTR6'] * (maxNSTR == 6) +
        params['NSTR7'] * (maxNSTR == 7) +
        params['NSTR8'] * (maxNSTR == 8)
    )

    # Calculate probability
    etProbability = np.exp(etUtility) / (np.exp(etUtility) + np.exp(0))

    # Monte Carlo to simulate choice based on probability
    return np.random.rand() < etProbability


def endtour_later(
    tour, tourlocs, tourSequence, universal, skim,
    nZones, carryingCapacity, params, shipments, nNSTR
):
    '''
    Returns True if we decide to end the tour, False if we decide
    to add another shipment
    '''
    # Calculate explanatory variables
    tourDuration = tourdur(tourSequence, skim, nZones)
    prox = np.min(proximity(tourlocs, universal, skim, nZones))
    capUt = cap_utilization(shipments[tour[0], 4], shipments[tour, 6], carryingCapacity)
    numberOfStops = len(np.unique(tourlocs))
    vehicleTypeIs0 = (shipments[tour[0], 4] == 0) * 1
    vehicleTypeIs1 = (shipments[tour[0], 4] == 1) * 1
    maxNSTR = max_nstr(tour, shipments, nNSTR)

    # Calculate utility
    etUtility = (
        params['ASC_ET'] +
        params['Tour duration'] * (tourDuration) +
        params['Capacity utilization'] * capUt +
        params['Prox'] * prox +
        params['Transshipment'] * transship(tour, shipments) +
        params['Stops'] * np.log(numberOfStops) +
        params['DC loading'] * lognode_loading(tour, shipments) +
        params['DC unloading'] * lognode_unloading(tour, shipments) +
        params['Urban'] * urbanzone(tour, shipments) +
        params['VT0'] * vehicleTypeIs0 +
        params['VT1'] * vehicleTypeIs1 +
        params['NSTR0'] * (maxNSTR == 0) +
        params['NSTR1'] * (maxNSTR == 1) +
        params['NSTR6'] * (maxNSTR == 6) +
        params['NSTR7'] * (maxNSTR == 7) +
        params['NSTR8'] * (maxNSTR == 8)
    )

    # Calculate probability
    etProbability = np.exp(etUtility) / (np.exp(etUtility) + np.exp(0))

    # Monte Carlo to simulate choice based on probability
    return np.random.rand() < etProbability


def selectshipment(
    tour, tourlocs, universal, skim,
    nZones, nTOD, carryingCapacity, shipments
):
    '''
    Returns the chosen shipment based on Select Shipment MNL
    '''
    # Some tour characteristics as input for the constraint checks
    if type(tour) == int:
        tourLS = shipments[tour][10]
        tourVT = shipments[tour][4]
        tourTODs = [shipments[tour][13]]
    else:
        tourLS = shipments[tour[0]][10]
        tourVT = shipments[tour[0]][4]
        tourTODs = [shipments[i][13] for i in tour]

    # Check capacity utilization
    tourWeight = np.sum(shipments[tour, 6])
    shipsCapUt = (tourWeight + universal[:, 6]) / carryingCapacity[tourVT]

    # Check proximity of other shipments to the tour
    shipsProx = proximity(tourlocs, universal, skim, nZones)

    # Which shipments belong to the same logistic segment and have
    # the same vehicle type as the tour
    shipsSameLS = np.array(universal[:, 10] == tourLS)
    shipsSameVT = np.array(universal[:, 4] == tourVT)

    # Which shipments have a delivery time matching with the tour
    shipsTOD = universal[:, 13]

    if len(tourTODs) == 1:

        # Codes of the first and last period of a day
        firstCategoryTOD = 0
        lastCategoryTOD = (nTOD - 1)

        # Only two aligning time periods allowed in a tour
        if tourTODs[0] == firstCategoryTOD:
            shipsFeasibleTOD = (
                (np.abs(shipsTOD - tourTODs[0]) <= 1) |
                (shipsTOD == lastCategoryTOD))
        elif tourTODs[0] == lastCategoryTOD:
            shipsFeasibleTOD = (
                (np.abs(shipsTOD - tourTODs[0]) <= 1) |
                (shipsTOD == firstCategoryTOD))
        else:
            shipsFeasibleTOD = (
                np.abs(shipsTOD - tourTODs[0]) <= 1)

    else:
        shipsFeasibleTOD = (
            (universal[:, 13] == tourTODs[0]) |
            (universal[:, 13] == tourTODs[1]))

    # Initialize feasible choice set, those shipments that
    # comply with constraints
    selectShipConstraints = (
        (shipsCapUt < 1.1) &
        shipsSameLS &
        shipsFeasibleTOD)
    feasibleChoiceSet = universal[selectShipConstraints]

    # If there are no feasible shipments, the return -2 statement
    # is used to end the tour
    if len(feasibleChoiceSet) == 0:
        return -2

    else:
        shipsSameVT = shipsSameVT[selectShipConstraints]
        shipsProx = shipsProx[selectShipConstraints]

        # Make shipments of same vehicle type more likely to be chosen
        # (through lower proximity value)
        shipsProx[shipsSameVT] /= 2

        # Select the shipment with minimum distance to the tour (proximity)
        ssChoice = np.argmin(shipsProx)
        chosenShipment = feasibleChoiceSet[ssChoice]

        return chosenShipment


def form_tours(
    carMarkers, shipments, skim, nZones, maxNumShips,
    carryingCapacity, dcZones,
    nShipmentsPerCarrier, nCarriers, nTOD, nNSTR,
    logitParams_ETfirst, logitParams_ETlater,
    seed,
    cars,
):
    '''
    Run the tour formation procedure for a set of carriers with a set of shipments.
    '''
    tours = [[] for car in range(nCarriers)]
    tourSequences = [[] for car in range(nCarriers)]

    for car in cars:
        print(f'\tForming tours for carrier {car+1} of {nCarriers}...', end='\r')

        np.random.seed(seed + car)
        np.random.seed(np.random.randint(10000000))

        tourCount = 0

        # Universal choice set = all non-allocated shipments per carrier
        universalChoiceSet = (
            shipments[carMarkers[car]:carMarkers[car + 1], :].copy())

        if car < len(dcZones):
            dc = dcZones[car]

            # Sort by shipment distance, this will help in constructing
            # more efficient/realistic tours
            shipmentDistances = skim[
                (universalChoiceSet[:, 1] - 1) * nZones + (universalChoiceSet[:, 1] - 1)]
            universalChoiceSet = np.c_[universalChoiceSet, shipmentDistances]
            universalChoiceSet = universalChoiceSet[universalChoiceSet[:, -1].argsort()]
            universalChoiceSet = universalChoiceSet[:, :-1]

        else:
            dc = None

        while len(universalChoiceSet) != 0:
            shipmentCount = 0

            tourLocations = np.zeros(
                (min(nShipmentsPerCarrier[car], maxNumShips) * 2, 2),
                dtype=int)

            # Loading and unloading location of this shipment
            tourLocations[shipmentCount, 0] = universalChoiceSet[0, 1].copy()
            tourLocations[shipmentCount, 1] = universalChoiceSet[0, 2].copy()

            # First shipment in the tour is the first listed one
            # in universal choice set
            tours[car].append([universalChoiceSet[0, 0].copy()])

            # Tour with only 1 shipment:
            # sequence = go from loading to unloading of this shipment
            tourSequences[car].append(
                np.array([universalChoiceSet[0, 1], universalChoiceSet[0, 2]], dtype=int))

            # Remove shipment from universal choice set
            universalChoiceSet = np.delete(universalChoiceSet, 0, 0)

            # If no shipments left for carrier,
            # break out of while loop and go to next carrier
            if len(universalChoiceSet) == 0:
                tourCount += 1
                break

            else:
                # Current tour, tourlocations and toursequence
                tour = tours[car][tourCount]

                # Duration, proximity value and capacity utilization of the tour
                tourDuration = tourdur(
                    tourSequences[car][tourCount], skim, nZones)
                prox = proximity(
                    tourLocations[0:shipmentCount + 1], universalChoiceSet, skim, nZones)
                capUt = cap_utilization(
                    shipments[tour[0], 4],
                    shipments[tour, 6],
                    carryingCapacity)

                # Check for End Tour constraints
                etConstraintCheck = (
                    (shipmentCount < maxNumShips) and
                    (tourDuration < 9) and
                    (capUt < 1) and
                    (np.min(prox) < 100) and
                    ~is_concrete())

                if etConstraintCheck:
                    # End Tour or not
                    etChoice = endtour_first(
                        tourDuration,
                        capUt,
                        tour,
                        logitParams_ETfirst,
                        shipments,
                        nNSTR)

                    while (not etChoice) and len(universalChoiceSet) != 0:
                        # Current tour, tourlocations and toursequence
                        tour = tours[car][tourCount]

                        # Duration, proximity and capacity utilization of the tour
                        tourDuration = tourdur(
                            tourSequences[car][tourCount], skim, nZones)
                        prox = proximity(
                            tourLocations[0:shipmentCount + 1], universalChoiceSet, skim, nZones)
                        capUt = cap_utilization(
                            shipments[tour[0], 4],
                            shipments[tour, 6],
                            carryingCapacity)

                        # Check for End Tour constraints
                        etConstraintCheck = (
                            (shipmentCount < maxNumShips) and
                            (tourDuration < 9) and
                            (capUt < 1) and
                            (np.min(prox) < 100) and
                            ~is_concrete())

                        if etConstraintCheck:
                            # Choose which shipment to add
                            chosenShipment = selectshipment(
                                tour, tourLocations[0:shipmentCount + 1], universalChoiceSet, skim,
                                nZones, nTOD,
                                carryingCapacity,
                                shipments)

                            # If no feasible shipments left
                            if np.any(chosenShipment == -2):
                                tourCount += 1
                                break

                            else:
                                shipmentCount += 1

                                # Add the chosen shipment to the tour
                                # and update tour sequence
                                tours[car][tourCount] = np.append(
                                    tours[car][tourCount],
                                    int(chosenShipment[0]))
                                tourLocations[shipmentCount, 0] = int(chosenShipment[1])
                                tourLocations[shipmentCount, 1] = int(chosenShipment[2])
                                
                                tourTODs = shipments[tours[car][tourCount], 13]
                                tourSequences[car][tourCount] = nearest_neighbor(
                                    tourLocations[0:shipmentCount + 1],
                                    tourTODs,
                                    dc,
                                    skim,
                                    nZones, nTOD)

                                # Remove chosen shipment from
                                # universal choice set
                                shipmentToDelete = np.where(
                                    universalChoiceSet[:, 0] == chosenShipment[0])[0][0]
                                universalChoiceSet = np.delete(
                                    universalChoiceSet,
                                    shipmentToDelete,
                                    0)

                                if len(universalChoiceSet) != 0:
                                    # Current tour, tourlocations
                                    # and toursequence
                                    tour = tours[car][tourCount]

                                    # End tour or not
                                    etChoice = endtour_later(
                                        tour,
                                        tourLocations[0:shipmentCount + 1],
                                        [tourSequences[car][tourCount][0]],
                                        universalChoiceSet, skim,
                                        nZones,
                                        carryingCapacity,
                                        logitParams_ETlater,
                                        shipments,
                                        nNSTR)

                                else:
                                    tourCount += 1
                                    break

                        else:
                            tourCount += 1
                            break

                    else:
                        tourCount += 1

                else:
                    tourCount += 1

    return [tours, tourSequences]
