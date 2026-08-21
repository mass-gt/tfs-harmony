import numpy as np
import pandas as pd
import logging
import time

from typing import Any, Dict

logger = logging.getLogger("tfs")


def create_schedules(
    parcelsAgg: pd.DataFrame,
    dropOffTime: float,
    skimTravTime: np.ndarray,
    skimDistance: np.ndarray,
    parcelNodesCEP: Dict[int, str],
    parcelDepTime: np.ndarray,
    tourType: int,
    seed: int,
    root: Any,
    startValueProgress: float,
    endValueProgress: float,
):
    '''
    Create the parcel schedules and store them in a DataFrame
    '''
    nZones = int(len(skimTravTime)**0.5)
    depots = np.unique(parcelsAgg['Depot'])
    nDepots = len(depots)

    print('\t0%', end='\r')

    tours = {}
    parcelsDelivered = {}
    departureTimes = {}
    depotCount = 0
    nTrips = 0

    for depot in np.unique(parcelsAgg['Depot']):
        depotParcels = parcelsAgg[parcelsAgg['Depot'] == depot]

        tours[depot] = {}
        parcelsDelivered[depot] = {}
        departureTimes[depot] = {}

        for cluster in np.unique(depotParcels['Cluster']):
            tour = []

            clusterParcels = depotParcels[depotParcels['Cluster'] == cluster]
            depotZone = list(clusterParcels['Orig'])[0]
            destZones = list(clusterParcels['Dest'])
            nParcelsPerZone = dict(zip(destZones, clusterParcels['Parcels']))

            # Nearest neighbor
            tour.append(depotZone)
            for i in range(len(destZones)):
                distances = [skimDistance[
                    tour[i] * nZones + dest] for dest in destZones]
                nextIndex = np.argmin(distances)
                tour.append(destZones[nextIndex])
                destZones.pop(nextIndex)
            tour.append(depotZone)

            # Shuffle the order of tour locations and accept
            # the shuffle if it reduces the tour distance
            nStops = len(tour)
            tour = np.array(tour, dtype=int)
            tourDist = np.sum(skimDistance[tour[:-1] * nZones + tour[1:]])

            if nStops > 4:
                for shiftLocA in range(1, nStops - 1):
                    for shiftLocB in range(1, nStops - 1):
                        if shiftLocA != shiftLocB:
                            swappedTour = tour.copy()
                            swappedTour[shiftLocA] = tour[shiftLocB]
                            swappedTour[shiftLocB] = tour[shiftLocA]
                            swappedTourDist = np.sum(skimDistance[
                                swappedTour[:-1] * nZones + swappedTour[1:]])

                            if swappedTourDist < tourDist:
                                tour = swappedTour.copy()
                                tourDist = swappedTourDist

            # Add current tour to dictionary with all formed tours
            tours[depot][cluster] = list(tour.copy())

            # Store the number of parcels delivered at each
            # location in the tour
            nParcelsPerStop = []
            for i in range(1, nStops - 1):
                nParcelsPerStop.append(nParcelsPerZone[tour[i]])
            nParcelsPerStop.append(0)
            parcelsDelivered[depot][cluster] = list(nParcelsPerStop.copy())

            # Determine the departure time of each trip in the tour
            np.random.seed(seed + 10000 * depot + cluster)
            np.random.seed(np.random.randint(10000000))

            departureTimesTour = [
                np.where(parcelDepTime > np.random.rand())[0][0] +
                np.random.rand()]

            for i in range(1, nStops - 1):
                orig = tour[i - 1]
                dest = tour[i]
                travTime = skimTravTime[orig * nZones + dest] / 3600
                departureTimesTour.append(
                    departureTimesTour[i - 1] +
                    dropOffTime * nParcelsPerStop[i - 1] +
                    travTime)
            departureTimes[depot][cluster] = list(departureTimesTour.copy())

            nTrips += (nStops - 1)

        print(
            '\t' + str(round((depotCount + 1) / nDepots * 100, 1)) + '%',
            end='\r')

        if root is not None:
            root.progressBar['value'] = (
                startValueProgress +
                (endValueProgress - startValueProgress - 1) * (depotCount + 1) / nDepots)

        depotCount += 1

    # --------------------------- Create return table -------------------------
    deliveriesCols = [
        'TourType',
        'CEP',
        'Depot_ID', 'Tour_ID', 'Trip_ID', 'Unique_ID',
        'O_zone', 'D_zone',
        'N_parcels',
        'Traveltime',
        'TourDepTime', 'TripDepTime', 'TripEndTime']
    deliveries = np.zeros((nTrips, len(deliveriesCols)), dtype=object)

    tripcount = 0
    for depot in tours.keys():
        for tour in tours[depot].keys():
            for trip in range(len(tours[depot][tour]) - 1):

                orig = tours[depot][tour][trip]
                dest = tours[depot][tour][trip + 1]

                # Depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                deliveries[tripcount, 0] = tourType

                # Name of the couriers
                if tourType <= 1:
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]
                else:
                    deliveries[tripcount, 1] = 'ConsolidatedUCC'

                # Depot_ID, Tour_ID, Trip_ID,
                # Unique ID under consideration of tour type
                deliveries[tripcount, 2] = depot
                deliveries[tripcount, 3] = f'{depot}_{tour}'
                deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'
                deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'

                # Origin and destination
                deliveries[tripcount, 6] = orig
                deliveries[tripcount, 7] = dest

                # Number of parcels
                deliveries[tripcount, 8] = parcelsDelivered[depot][tour][trip]

                # Travel time in hrs
                deliveries[tripcount, 9] = skimTravTime[
                    orig * nZones + dest] / 3600

                # Departure of tour from depot
                deliveries[tripcount, 10] = departureTimes[depot][tour][0]

                # Departure time of trip
                deliveries[tripcount, 11] = departureTimes[depot][tour][trip]

                # End of trip/start of next trip if there is another one
                deliveries[tripcount, 12] = 0.0

                tripcount += 1

    # Place in DataFrame with the right data type per column
    deliveries = pd.DataFrame(deliveries, columns=deliveriesCols)
    for col, dtype in (
        ('TourType', int),
        ('CEP', str),
        ('Depot_ID', int),
        ('Tour_ID', str),
        ('Trip_ID', str),
        ('Unique_ID', str),
        ('O_zone', int),
        ('D_zone', int),
        ('N_parcels', int),
        ('Traveltime', float),
        ('TourDepTime', float),
        ('TripDepTime', float),
        ('TripEndTime', float),
    ):
        deliveries[col] = deliveries[col].astype(dtype)

    vehTypes = ['Van', 'Van', 'Van', 'LEVV']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC']
    destTypes = ['HH', 'UCC', 'HH', 'HH']

    deliveries['VehType'] = vehTypes[tourType]
    deliveries['OrigType'] = origTypes[tourType]
    deliveries['DestType'] = destTypes[tourType]

    if root is not None:
        root.progressBar['value'] = endValueProgress

    return deliveries


def cluster_parcels(
    parcels, maxVehicleLoad, skimDistance,
    root, startValueProgress, endValueProgress
):
    '''
    Assign parcels to clusters based on spatial proximity with
    cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.
    '''
    parcels.index = np.arange(len(parcels))

    depotNumbers = np.unique(parcels['DepotNumber'])
    nParcels = len(parcels)
    nParcelsAssigned = 0
    firstClusterID = 0
    nZones = int(len(skimDistance)**0.5)

    parcels['Cluster'] = -1

    print('\t0%', end='\r')

    # First check for depot/destination combination with more than
    # {maxVehicleLoad} parcels.
    # These we don't need to use the clustering algorithm for
    counts = pd.pivot_table(
        parcels,
        values=['VEHTYPE'],
        index=['DepotNumber', 'D_zone'],
        aggfunc=len)

    whereLargeCluster = list(counts.index[
        np.where(counts >= maxVehicleLoad)[0]])

    for x in whereLargeCluster:
        depotNumber = x[0]
        destZone = x[1]

        indices = np.where(
            (parcels['DepotNumber'] == depotNumber) &
            (parcels['D_zone'] == destZone))[0]

        for i in range(int(np.floor(len(indices) / maxVehicleLoad))):
            parcels.loc[indices[:maxVehicleLoad], 'Cluster'] = firstClusterID
            indices = indices[maxVehicleLoad:]

            firstClusterID += 1
            nParcelsAssigned += maxVehicleLoad

            print('\t' + str(round(nParcelsAssigned / nParcels * 100, 1)) + '%',
                  end='\r')

            if root is not None:
                root.progressBar['value'] = (
                    startValueProgress +
                    (endValueProgress - startValueProgress - 1) * nParcelsAssigned / nParcels)

    # For each depot, cluster remaining parcels into batches of
    # {maxVehicleLoad} parcels
    for depotNumber in depotNumbers:

        # Select parcels of the depot that are not assigned a cluster yet
        parcelsToFit = parcels[
            (
                (parcels['DepotNumber'] == depotNumber) &
                (parcels['Cluster'] == -1)
            )].copy()

        # Sort parcels descending based on distance to depot
        # so that at the end of the loop the remaining parcels
        # are all nearby the depot and form a somewhat reasonable
        # parcels cluster
        parcelsToFit['Distance'] = skimDistance[
            (parcelsToFit['O_zone'] - 1) * nZones +
            (parcelsToFit['D_zone'] - 1)]
        parcelsToFit = parcelsToFit.sort_values('Distance', ascending=False)
        parcelsToFitIndex = list(parcelsToFit.index)
        parcelsToFit.index = np.arange(len(parcelsToFit))
        dests = np.array(parcelsToFit['D_zone'])

        # How many tours are needed to deliver these parcels
        nTours = int(np.ceil(len(parcelsToFit) / maxVehicleLoad))

        # In the case of 1 tour it's simple,
        # all parcels belong to the same cluster
        if nTours == 1:
            parcels.loc[parcelsToFitIndex, 'Cluster'] = firstClusterID
            firstClusterID += 1
            nParcelsAssigned += len(parcelsToFit)

        # When there are multiple tours needed, the heuristic is a
        # little bit more complex
        else:
            clusters = np.ones(len(parcelsToFit), dtype=int) * -1

            for tour in range(nTours):
                # Select the first parcel for the new cluster
                # that is now initialized
                yetAssigned = (clusters != -1)
                notYetAssigned = np.where(~yetAssigned)[0]
                firstParcelIndex = notYetAssigned[0]
                clusters[firstParcelIndex] = firstClusterID

                # Find the nearest {maxVehicleLoad-1} parcels
                # to this first parcel that are not in a cluster yet
                distances = skimDistance[
                    (dests[firstParcelIndex] - 1) * nZones + (dests - 1)]
                distances[notYetAssigned[0]] = 99999
                distances[yetAssigned] = 99999
                clusters[np.argsort(distances)[
                    :(maxVehicleLoad - 1)]] = firstClusterID

                firstClusterID += 1

            # Group together remaining parcels, these are all nearby the depot
            yetAssigned = (clusters != -1)
            notYetAssigned = np.where(~yetAssigned)[0]
            clusters[notYetAssigned] = firstClusterID
            firstClusterID += 1

            parcels.loc[parcelsToFitIndex, 'Cluster'] = clusters
            nParcelsAssigned += len(parcelsToFit)

            print('\t' + str(round(nParcelsAssigned / nParcels * 100, 1)) + '%',
                  end='\r')

            if root is not None:
                root.progressBar['value'] = (
                    startValueProgress +
                    (endValueProgress - startValueProgress - 1) * nParcelsAssigned / nParcels)

    parcels['Cluster'] = parcels['Cluster'].astype(int)

    return parcels


def write_schedules_to_geojson(
    deliveries: pd.DataFrame,
    parcelNodes: pd.DataFrame,
    zonesX: Dict[int, float],
    zonesY: Dict[int, float],
    varDict: Dict[str, str],
    root: Any,
) -> None:
    """Writes the parcel schedules to a geojson file with coordinates."""
    # Initialize arrays with coordinates
    Ax = np.zeros(len(deliveries), dtype=int)
    Ay = np.zeros(len(deliveries), dtype=int)
    Bx = np.zeros(len(deliveries), dtype=int)
    By = np.zeros(len(deliveries), dtype=int)

    # Determine coordinates of LineString for each trip
    tripIDs = [x.split('_')[-1] for x in deliveries['Trip_ID']]
    tourTypes = np.array(deliveries['TourType'], dtype=int)
    depotIDs = np.array(deliveries['Depot_ID'])

    for i in deliveries.index[:-1]:

        # First trip of tour
        if tripIDs[i] == '0' and tourTypes[i] <= 1:
            Ax[i] = parcelNodes['X'][depotIDs[i]]
            Ay[i] = parcelNodes['Y'][depotIDs[i]]
            Bx[i] = zonesX[deliveries['D_zone'][i]]
            By[i] = zonesY[deliveries['D_zone'][i]]

        # Last trip of tour
        elif tripIDs[i + 1] == '0' and tourTypes[i] <= 1:
            Ax[i] = zonesX[deliveries['O_zone'][i]]
            Ay[i] = zonesY[deliveries['O_zone'][i]]
            Bx[i] = parcelNodes['X'][depotIDs[i]]
            By[i] = parcelNodes['Y'][depotIDs[i]]

        # Intermediate trips of tour
        else:
            Ax[i] = zonesX[deliveries['O_zone'][i]]
            Ay[i] = zonesY[deliveries['O_zone'][i]]
            Bx[i] = zonesX[deliveries['D_zone'][i]]
            By[i] = zonesY[deliveries['D_zone'][i]]

    # Last trip of last tour
    i += 1
    if tourTypes[i] <= 1:
        Ax[i] = zonesX[deliveries['O_zone'][i]]
        Ay[i] = zonesY[deliveries['O_zone'][i]]
        Bx[i] = parcelNodes['X'][depotIDs[i]]
        By[i] = parcelNodes['Y'][depotIDs[i]]
    else:
        Ax[i] = zonesX[deliveries['O_zone'][i]]
        Ay[i] = zonesY[deliveries['O_zone'][i]]
        Bx[i] = zonesX[deliveries['D_zone'][i]]
        By[i] = zonesY[deliveries['D_zone'][i]]

    Ax = np.array(Ax, dtype=str)
    Ay = np.array(Ay, dtype=str)
    Bx = np.array(Bx, dtype=str)
    By = np.array(By, dtype=str)
    nTrips = len(deliveries)

    filename = (
        varDict['OUTPUTFOLDER'] +
        f"ParcelSchedule_{varDict['LABEL']}.geojson")
    with open(filename, 'w') as geoFile:
        geoFile.write(
            '{\n' + '"type": "FeatureCollection",\n' +
            '"features": [\n')

        for i in range(nTrips - 1):
            outputStr = (
                '{ "type": "Feature", "properties": ' +
                str(deliveries.loc[i, :].to_dict()).replace("'", '"') +
                ', "geometry": ' +
                '{ "type": "LineString", "coordinates": [ [ ' +
                Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                Bx[i] + ', ' + By[i] + ' ] ] } },\n')
            geoFile.write(outputStr)

            if i % int(nTrips / 20) == 0:
                print(
                    '\t' + str(round(i / nTrips * 100, 1)) + '%',
                    end='\r')

                if root is not None:
                    root.progressBar['value'] = (
                        91.0 +
                        (98.0 - 91.0) * (i / nTrips))

        # Bij de laatste feature moet er geen komma aan het einde
        i += 1
        outputStr = (
            '{ "type": "Feature", "properties": ' +
            str(deliveries.loc[i, :].to_dict()).replace("'", '"') +
            ', "geometry": ' +
            '{ "type": "LineString", "coordinates": [ [ ' +
            Ax[i] + ', ' + Ay[i] + ' ], [ ' +
            Bx[i] + ', ' + By[i] + ' ] ] } }\n')
        geoFile.write(outputStr)
        geoFile.write(']\n')
        geoFile.write('}')


def export_trip_matrices(
    deliveries: pd.DataFrame,
    varDict: Dict[str, str],
) -> None:
    """Constructs OD matrices with numbers of trips and exports these to the output folder."""
    cols = ['ORIG', 'DEST', 'N_TOT']
    deliveries['N_TOT'] = 1

    # Gebruik N_TOT om het aantal ritten per HB te bepalen,
    # voor elk logistiek segment, voertuigtype en totaal
    pivotTable = pd.pivot_table(
        deliveries,
        values=['N_TOT'],
        index=['O_zone', 'D_zone'],
        aggfunc=np.sum)
    pivotTable['ORIG'] = [x[0] for x in pivotTable.index]
    pivotTable['DEST'] = [x[1] for x in pivotTable.index]
    pivotTable = pivotTable[cols]

    # Assume one intrazonal trip for each zone with
    # multiple deliveries visited in a tour
    intrazonalTrips = {}

    for i in deliveries[deliveries['N_parcels'] > 1].index:
        zone = deliveries.at[i, 'D_zone']
        if zone in intrazonalTrips.keys():
            intrazonalTrips[zone] += 1
        else:
            intrazonalTrips[zone] = 1

    intrazonalKeys = list(intrazonalTrips.keys())

    for zone in intrazonalKeys:

        if (zone, zone) in pivotTable.index:
            pivotTable.at[(zone, zone), 'N_TOT'] += (
                intrazonalTrips[zone])

            del intrazonalTrips[zone]

    intrazonalTripsDF = pd.DataFrame(np.zeros((len(intrazonalTrips), 3)), columns=cols)
    intrazonalTripsDF['ORIG'] = intrazonalTrips.keys()
    intrazonalTripsDF['DEST'] = intrazonalTrips.keys()
    intrazonalTripsDF['N_TOT'] = intrazonalTrips.values()

    pivotTable = pivotTable.append(intrazonalTripsDF)
    pivotTable = pivotTable.sort_values(['ORIG', 'DEST'])

    pivotTable.to_csv(
        (
            varDict['OUTPUTFOLDER'] +
            f"tripmatrix_parcels_{varDict['LABEL']}.txt"
        ),
        index=False,
        sep='\t')

    logger.debug(
        "\tTrip matrix written to " +
        f"{varDict['OUTPUTFOLDER']}tripmatrix_parcels_{varDict['LABEL']}.txt")

    # Departure time after 24:00 get 24 subtracted
    deliveries.loc[deliveries['TripDepTime'] >= 24, 'TripDepTime'] -= 24
    deliveries.loc[deliveries['TripDepTime'] >= 24, 'TripDepTime'] -= 24

    for tod in range(24):

        logger.debug(f"\t\tAlso generating trip matrix for TOD {tod}...")

        output = deliveries[
            (deliveries['TripDepTime'] >= tod) &
            (deliveries['TripDepTime'] < (tod + 1))].copy()
        output['N_TOT'] = 1

        if len(output) > 0:
            # Gebruik deze dummies om het aantal ritten
            # per HB te bepalen, voor elk logistiek segment,
            # voertuigtype en totaal
            pivotTable = pd.pivot_table(
                output,
                values=['N_TOT'],
                index=['O_zone', 'D_zone'],
                aggfunc=np.sum)
            pivotTable['ORIG'] = [x[0] for x in pivotTable.index]
            pivotTable['DEST'] = [x[1] for x in pivotTable.index]
            pivotTable = pivotTable[cols]

            # Assume one intrazonal trip for each zone with
            # multiple deliveries visited in a tour
            intrazonalTrips = {}

            for i in output[output['N_parcels'] > 1].index:
                zone = output.at[i, 'D_zone']
                if zone in intrazonalTrips.keys():
                    intrazonalTrips[zone] += 1
                else:
                    intrazonalTrips[zone] = 1

            intrazonalKeys = list(intrazonalTrips.keys())

            for zone in intrazonalKeys:

                if (zone, zone) in pivotTable.index:
                    pivotTable.at[(zone, zone), 'N_TOT'] += (
                        intrazonalTrips[zone])

                    del intrazonalTrips[zone]

            intrazonalTripsDF = pd.DataFrame(
                np.zeros((len(intrazonalTrips), 3)),
                columns=cols)
            intrazonalTripsDF['ORIG'] = intrazonalTrips.keys()
            intrazonalTripsDF['DEST'] = intrazonalTrips.keys()
            intrazonalTripsDF['N_TOT'] = intrazonalTrips.values()
            pivotTable = pivotTable.append(intrazonalTripsDF)
            pivotTable = pivotTable.sort_values(['ORIG', 'DEST'])

        else:
            pivotTable = pd.DataFrame(columns=cols)

        pivotTable.to_csv(
            f"{varDict['OUTPUTFOLDER']}tripmatrix_parcels_{varDict['LABEL']}_TOD{tod}.txt",
            index=False,
            sep='\t')


def do_crowdshipping(
    parcels, zones, nIntZones, nZones, zoneDict,
    zonesX, zonesY,
    skimDistance, skimTravTime,
    nFirstZonesCS, parcelShareCRW, modes,
    zone_gemeente_dict, segsDetail,
    datapathO, label,
    seed,
    root,
):
    '''
    Do all crowdshipping calculations and export files for
    the crowdshipping use case.
    '''
    start_time_cs = time.time()

    logger.debug("\tCrowdshipping use case...")

    logger.debug("\t\tGet parcel demand...")

    nParcels = len(parcels)

    np.random.seed(seed)

    randSelectionCRW = (
        (np.random.rand(nParcels) < parcelShareCRW) &
        (parcels['D_zone'] < nFirstZonesCS))
    indicesCRW = np.where(randSelectionCRW)[0]
    indicesREF = np.where(~randSelectionCRW)[0]

    # The parcels to use for crowdshipping
    parcelsCRW = parcels.loc[indicesCRW, :]
    parcelsCRW.index = parcelsCRW['Parcel_ID']

    # The parcels to ship regularly by parcel couriers
    parcels = parcels.loc[indicesREF, :]
    parcels.index = parcels['Parcel_ID']

    # Add number of parcels to zonal data
    zones['parcels'] = [
        np.sum(parcels['D_zone'] == (i + 1))
        for i in range(nIntZones)]
    zones['parcelsCS'] = [
        np.sum(parcelsCRW['D_zone'] == (i + 1))
        for i in range(nIntZones)]
    nParcels = int(zones[:nFirstZonesCS]["parcels"].sum())
    nParcelsCS = int(zones[:nFirstZonesCS]["parcelsCS"].sum())

    # Dictionary of all zones per municipality
    gemeente_zone_dict = {}
    gemeente_id_dict = {}
    id_gemeente_dict = {}
    count = 0

    for gemeente in np.unique(zones[:nFirstZonesCS]['Gemeentena']):
        gemeente_zone_dict[gemeente] = np.where(
            zones['Gemeentena'] == gemeente)[0]
        gemeente_id_dict[gemeente] = count
        id_gemeente_dict[count] = gemeente
        count += 1

    # Initialize an array for the crowdshipping parcels
    parcelsCS_cols = {
        'id': 0,
        'orig': 1,
        'dest': 2,
        'orig_skim': 3,
        'dest_skim': 4,
        'gemeente': 5,
        'X_ORIG': 6,
        'Y_ORIG': 7,
        'X_DEST': 8,
        'Y_DEST': 9,
        'TravelTime_car': 10,
        'TravelDistance': 11,
        'vector': 12,
        'status': 13,
        'traveller': 14,
        'modal choice': 15,
        'detour_time': 16,
        'detour_dist': 17,
        'compensation': 18}

    parcelsCS_array = np.zeros(
        (nParcelsCS, len(parcelsCS_cols)),
        dtype=object)

    # Dictionary from (key) parcel ID as used in loop below
    # to (value) parcel ID as used in input parcel demand file
    parc_id_dict = {}

    # Create object for crowdshipping parcels
    count = 0
    for i in range(nFirstZonesCS):
        nParcelsZone = int(zones.at[zoneDict[i + 1], 'parcelsCS'])
        parc_ids = np.array(parcelsCRW.loc[
            parcelsCRW['D_zone'] == (i + 1), 'Parcel_ID'])

        if nParcelsZone > 0:
            dest_skim = i + 1
            dest = zoneDict[dest_skim]
            gemeente = zones.at[dest, 'Gemeentena']
            x_dest = zonesX[dest]
            y_dest = zonesY[dest]
            status = "ordered"
            possible_origins = gemeente_zone_dict[gemeente]

            ratio_origins = np.zeros(nIntZones)
            ratio_origins[possible_origins] = segsDetail[possible_origins]
            ratio_origins = np.cumsum(ratio_origins)
            ratio_origins /= ratio_origins[-1]

            for n in range(nParcelsZone):
                parc_id = n + count
                orig_skim = np.where(
                    ratio_origins >= np.random.rand())[0][0] + 1
                orig = zoneDict[orig_skim]
                x_orig = zonesX[orig]
                y_orig = zonesY[orig]
                trav_time = skimTravTime[
                    (orig_skim - 1) * nZones + (dest_skim - 1)] / 3600
                trav_dist = skimDistance[
                    (orig_skim - 1) * nZones + (dest_skim - 1)] / 1000
                vector = [
                    x_dest - x_orig,
                    y_dest - y_orig]
                parcelsCS_array[n + count] = [
                    parc_id,
                    orig, dest,
                    orig_skim, dest_skim,
                    gemeente,
                    x_orig, y_orig, x_dest, y_dest,
                    trav_time, trav_dist,
                    vector, status,
                    0, 0, 0, 0, 0]
                parc_id_dict[parc_id] = parc_ids[n]

        count += nParcelsZone

    # Recode municipalities to numberic IDs for faster checking
    # in parcel assignment loop
    parcelsCS_array[:, 5] = np.array(
        [gemeente_id_dict[x] for x in parcelsCS_array[:, 5]],
        dtype=int)

    # Place crowdshipping parcel in DataFrame with headers
    parcelsCS_df = pd.DataFrame(parcelsCS_array, columns=parcelsCS_cols)

    if root is not None:
        root.progressBar['value'] = 2.0

    logger.debug("\t\tGet potential crowdshippers...")

    # Editing OD-matrices of passenger travellers that are willing to crowdship
    trav_array_cols = {
        'id': 0,
        'orig': 1,
        'dest': 2,
        'orig_skim': 3,
        'dest_skim': 4,
        'vector': 5,
        'gemeenten': 6,
        'parcel': 7,
        'status': 8}

    for mode in modes:
        OD_array = modes[mode]['OD_array']
        trav_array = np.zeros(
            (OD_array.sum().sum(), len(trav_array_cols)),
            dtype=object)

        start_id = sum(d['n_trav'] for d in modes.values() if d)
        count = 0

        for i, row in enumerate(OD_array):
            for j in np.where(row > 0)[0]:

                if i != j:
                    n = row[j]

                    orig_skim = i + 1
                    dest_skim = j + 1
                    orig = int(zoneDict[orig_skim])
                    dest = int(zoneDict[dest_skim])

                    vector = [
                        zonesX[dest] - zonesX[orig],
                        zonesY[dest] - zonesY[orig]]

                    gemeente = [
                        zone_gemeente_dict[orig_skim],
                        zone_gemeente_dict[dest_skim]]

                    for N in range(n):
                        trav_id = N + count + start_id
                        trav_array[N + count] = [
                            trav_id,
                            orig, dest,
                            orig_skim, dest_skim,
                            vector, gemeente,
                            0, 0]

                    count += n

        trav_array = trav_array[~np.all(trav_array == 0, axis=1)]

        # Recode municipalities to numberic IDs for
        # faster checking in parcel assignment loop
        trav_array[:, 6] = [
            [gemeente_id_dict[x[0]], gemeente_id_dict[x[1]]]
            for x in trav_array[:, 6]]

        modes[mode]['n_trav'] = int(len(trav_array))
        modes[mode]['trav_array'] = trav_array

    if root is not None:
        root.progressBar['value'] = 4.0

    # Assign parcels to crowdshippers
    parcelsToBeAssigned = np.array([True for i in range(nParcelsCS)])
    nParcelsAssigned = 0

    # Variables to keep track of progress
    nTravellersTotal = sum([len(modes[mode]['trav_array']) for mode in modes])
    travellerCount = 0

    for mode in modes:

        logger.debug(f"\t\tAssigning {mode} travellers to parcels...")

        skimTravTime = modes[mode]['skim_time']
        dropoff_time = modes[mode]['dropoff_time']
        VoT = modes[mode]['VoT']

        nTravellers = len(modes[mode]['trav_array'])

        # Initialize variable with orig/dest municipality of
        # previously checked traveller
        prevMunicipality = [-1, -1]

        for i, traveller in enumerate(modes[mode]['trav_array']):

            # Stop in the case all crowdshipping-eligible parcels
            # are assigned to a bringer
            if nParcelsAssigned == nParcelsCS:
                break

            offers_dict = {}
            offers2_dict = {}

            trav_orig = traveller[3]
            trav_dest = traveller[4]
            trip_dist = skimDistance[
                (trav_orig - 1) * nZones + (trav_dest - 1)] / 1000
            trip_time = skimTravTime[
                (trav_orig - 1) * nZones + (trav_dest - 1)] / 3600

            # Boolean: Parcels for which no carrier has been found yet
            checkUnassigned = parcelsToBeAssigned

            # Boolean: Parcels with a reasonable distance in relation
            # to the traveller's trip distance
            checkDistance = (
                (trip_dist / parcelsCS_array[:, 11] < 4) &
                (trip_dist / parcelsCS_array[:, 11] > 0.5))

            # Boolean: Parcels within the municipality of traveller's
            # origin / destination
            # (Only needs to be recalculated if the traveller has
            # different orig/dest from previous traveller)
            if prevMunicipality != traveller[6]:
                checkMunicipality = (
                    (parcelsCS_array[:, 5] == traveller[6][0]) |
                    (parcelsCS_array[:, 5] == traveller[6][1]))

            # Now select the parcels that comply to the above
            # three boolean checks
            parcelsToConsider = parcelsCS_array[
                (checkUnassigned & checkMunicipality & checkDistance)]

            # Determine detour due to delivering parcel
            parc_orig = np.array(parcelsToConsider[:, 3], dtype=int)
            parc_dest = np.array(parcelsToConsider[:, 4], dtype=int)
            dist_traveller_parcel = skimDistance[
                (trav_orig - 1) * nZones + (parc_orig - 1)] / 1000
            dist_parcel_trip = skimDistance[
                (parc_orig - 1) * nZones + (parc_dest - 1)] / 1000
            dist_customer_end = skimDistance[
                (parc_dest - 1) * nZones + (trav_dest - 1)] / 1000
            CS_trip_dist = (
                dist_traveller_parcel +
                dist_parcel_trip +
                dist_customer_end)
            traveller_detour = CS_trip_dist - trip_dist
            extra_parcel_dist = traveller_detour - dist_parcel_trip
            relative_extra_parcel_dist = extra_parcel_dist / dist_parcel_trip

            # Determine compensation offered to traveller
            whereTripsWithinThreshold = np.where(
                relative_extra_parcel_dist <
                modes[mode]['relative_extra_parcel_dist_threshold'])[0]

            for trip in whereTripsWithinThreshold:
                CS_compensation = np.log((dist_parcel_trip[trip]) + 5)
                offers_dict[parcelsToConsider[trip, 0]] = {
                    'distance': dist_parcel_trip[trip],
                    'rel_detour': relative_extra_parcel_dist[trip],
                    'compensation': CS_compensation}

            # Traveller chooses the parcel to ship
            if offers_dict:
                offered_parcels = sorted(
                    offers_dict,
                    key=lambda x: (offers_dict[x]['rel_detour']))[:3]

                # Search for best parcel
                for parcel in offered_parcels:
                    parc_orig = parcelsCS_array[parcel, 3]
                    parc_dest = parcelsCS_array[parcel, 4]
                    traveller_detour_time = (
                        skimTravTime[
                            (trav_orig - 1) * nZones + (parc_orig - 1)] +
                        skimTravTime[
                            (parc_orig - 1) * nZones + (parc_dest - 1)] +
                        skimTravTime[
                            (parc_dest - 1) * nZones + (trav_dest - 1)])
                    traveller_detour_time /= 3600
                    traveller_detour_time -= trip_time
                    CS_utility = (
                        offers_dict[parcel]['compensation'] /
                        (traveller_detour_time + 2 * dropoff_time))
                    offers2_dict[parcel] = {'utility': CS_utility}

                best_parcel = offered_parcels[0]

                # Traveller chooses whether to ship this 'best' parcel
                # based on value of time
                if offers2_dict[best_parcel]['utility'] > VoT:
                    modes[mode]['trav_array'][i, 7] = int(best_parcel)
                    modes[mode]['trav_array'][i, 8] = str('shipping')
                    parcelsToBeAssigned[best_parcel] = False
                    nParcelsAssigned += 1

                    parcelsCS_array[best_parcel, 13] = 'carrier found'
                    parc_orig = parcelsCS_array[best_parcel, 3]
                    parc_dest = parcelsCS_array[best_parcel, 4]
                    traveller_detour_time = (
                        skimTravTime[
                            (trav_orig - 1) * nZones + (parc_orig - 1)] +
                        skimTravTime[
                            (parc_orig - 1) * nZones + (parc_dest - 1)] +
                        skimTravTime[
                            (parc_dest - 1) * nZones + (trav_dest - 1)] -
                        skimTravTime[
                            (trav_orig - 1) * nZones + (trav_dest - 1)]) / 3600
                    traveller_detour_distance = (
                        skimDistance[
                            (trav_orig - 1) * nZones + (parc_orig - 1)] +
                        skimDistance[
                            (parc_orig - 1) * nZones + (parc_dest - 1)] +
                        skimDistance[
                            (parc_dest - 1) * nZones + (trav_dest - 1)] -
                        skimDistance[
                            (trav_orig - 1) * nZones + (trav_dest - 1)]) / 1000
                    parcelsCS_array[best_parcel, 14] = traveller[0]
                    parcelsCS_array[best_parcel, 15] = mode
                    parcelsCS_array[best_parcel, 16] = traveller_detour_time
                    parcelsCS_array[best_parcel, 17] = traveller_detour_distance
                    parcelsCS_array[best_parcel, 18] = offers_dict[best_parcel]['compensation']

            travellerCount += 1
            prevMunicipality = traveller[6]

            if i % int(nTravellers / 100) == 0:
                print('\t\t' + str(round(i / nTravellers * 100, 1)) + "% ", end='\r')
                if root is not None:
                    root.progressBar['value'] = (
                        4.0 +
                        (52.0 - 4.0) * (travellerCount / nTravellersTotal))

    # Recode municipalities back to string
    parcelsCS_array[:, 5] = [
        id_gemeente_dict[x] for x in parcelsCS_array[:, 5]]

    # Put parcels back in DataFrame again
    parcelsCS_df = pd.DataFrame(parcelsCS_array, columns=parcelsCS_cols)

    # Parcels that are not assigned to an occassional carrier
    # will need to be scheduled in the regular scheduling procedure
    unassignedParcelIDs = np.array(parcelsCS_df.loc[
        parcelsCS_df['status'] != 'carrier found', 'id'])
    unassignedParcelIDs = [parc_id_dict[x] for x in unassignedParcelIDs]
    parcels = parcels.append(parcelsCRW.loc[unassignedParcelIDs, :])

    logger.debug("\t\tWriting crowdshipping output to CSV and GeoJSON...")

    # Write CSV
    parcelsCS_df.to_csv(
        datapathO + f'ParcelDemand_{label}_Crowdshipping.csv',
        index=False)

    # Write GeoJSON
    for mode in modes:
        trav_array = modes[mode]['trav_array']
        tours = pd.DataFrame()

        for i, traveller in enumerate(trav_array[trav_array[:, 8] == 'shipping']):
            trav_ORIG = traveller[1]
            trav_DEST = traveller[2]
            parc_ORIG = parcelsCS_array[int(traveller[7]), 1]
            parc_DEST = parcelsCS_array[int(traveller[7]), 2]
            trav_orig = traveller[3]
            trav_dest = traveller[4]
            parc_orig = parcelsCS_array[int(traveller[7]), 3]
            parc_dest = parcelsCS_array[int(traveller[7]), 4]

            for j in range(3):
                tours.at[i * 3 + j, 'TOUR_ID'] = i
                tours.at[i * 3 + j, 'TRIP_ID'] = str(i) + "_" + str(j)
                tours.at[i * 3 + j, 'traveller_ID'] = traveller[0]
                tours.at[i * 3 + j, 'parcel_ID'] = traveller[7]
                tours.at[i * 3 + j, 'mode'] = mode
                if j == 0:
                    tours.at[i * 3 + j, 'skim_dist'] = skimDistance[
                        (trav_orig - 1) * nZones + (parc_orig - 1)] / 1000
                    tours.at[i * 3 + j, 'ORIG'] = trav_ORIG
                    tours.at[i * 3 + j, 'DEST'] = parc_ORIG
                if j == 1:
                    tours.at[i * 3 + j, 'skim_dist'] = skimDistance[
                        (parc_orig - 1) * nZones + (parc_dest - 1)] / 1000
                    tours.at[i * 3 + j, 'ORIG'] = parc_ORIG
                    tours.at[i * 3 + j, 'DEST'] = parc_DEST
                if j == 2:
                    tours.at[i * 3 + j, 'skim_dist'] = skimDistance[
                        (parc_dest - 1) * nZones + (trav_dest - 1)] / 1000
                    tours.at[i * 3 + j, 'ORIG'] = parc_DEST
                    tours.at[i * 3 + j, 'DEST'] = trav_DEST

        if not tours.empty:
            for i, ORIG in enumerate(tours['ORIG']):
                tours.at[i, 'X_ORIG'] = zones.loc[ORIG]['X']
                tours.at[i, 'Y_ORIG'] = zones.loc[ORIG]['Y']
            for i, DEST in enumerate(tours['DEST']):
                tours.at[i, 'X_DEST'] = zones.loc[DEST]['X']
                tours.at[i, 'Y_DEST'] = zones.loc[DEST]['Y']

            # ----- GeoJSON ---
            Ax = np.array(tours['X_ORIG'], dtype=str)
            Ay = np.array(tours['Y_ORIG'], dtype=str)
            Bx = np.array(tours['X_DEST'], dtype=str)
            By = np.array(tours['Y_DEST'], dtype=str)
            nTrips = len(tours)

            filename = (
                datapathO +
                f'ParcelSchedule_{label}_Crowdshipping_{mode}.geojson')

            with open(filename, 'w') as geoFile:
                geoFile.write(
                    '{\n' +
                    '"type": "FeatureCollection",\n' +
                    '"features": [\n')

                for i in range(nTrips - 1):
                    outputStr = (
                        '{ "type": "Feature", "properties": ' +
                        str(tours.loc[i, :].to_dict()).replace("'", '"') +
                        ', "geometry": ' +
                        '{ "type": "LineString", "coordinates": [ [ ' +
                        Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                        Bx[i] + ', ' + By[i] + ' ] ] } },\n')
                    geoFile.write(outputStr)

                # Bij de laatste feature moet er geen komma aan het einde
                i += 1
                outputStr = (
                    '{ "type": "Feature", "properties": ' +
                    str(tours.loc[i, :].to_dict()).replace("'", '"') +
                    ', "geometry": ' +
                    '{ "type": "LineString", "coordinates": [ [ ' +
                    Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                    Bx[i] + ', ' + By[i] + ' ] ] } }\n')
                geoFile.write(outputStr)
                geoFile.write(']\n')
                geoFile.write('}')

    # Print summary of crowdshipping results
    n_parcels_total = nParcels
    n_parcels_CS = len(parcelsCS_array)
    n_parcels_CS_delivered = (parcelsCS_array[:, 13] == 'carrier found').sum()
    delivered_percentage = round(
        n_parcels_CS_delivered / n_parcels_CS * 100, 2)

    avg_dist = round(
        parcelsCS_array[:, 11].mean(), 2)
    avg_detour = round(
        parcelsCS_array[:, 17].sum() / n_parcels_CS_delivered, 2)
    total_detour = int(round(
        parcelsCS_array[:, 17].sum(), -1))

    bike_parcels = len(
        parcelsCS_array[parcelsCS_array[:, 15] == 'fiets'][:, 17])
    bike_km = int(round(
        parcelsCS_array[parcelsCS_array[:, 15] == 'fiets'][:, 17].sum(), -1))
    bike_km_avg = round(
        parcelsCS_array[parcelsCS_array[:, 15] == 'fiets'][:, 17].mean(), 2)

    car_parcels = len(
        parcelsCS_array[parcelsCS_array[:, 15] == 'auto'][:, 17])
    car_km = int(round(
        parcelsCS_array[parcelsCS_array[:, 15] == 'auto'][:, 17].sum(), -1))
    car_km_avg = round(
        parcelsCS_array[parcelsCS_array[:, 15] == 'auto'][:, 17].mean(), 2)

    avg_compensation = round(
        parcelsCS_array[
            parcelsCS_array[:, 13] == 'carrier found'][:, 18].mean(), 2)

    message = '\tSummary of outcomes:\n' + (
        (
            "\t\tA total of " +
            str(n_parcels_total) +
            " parcels is ordered in the system. " +
            str(n_parcels_CS) +
            " are eligible for CS of which " +
            str(n_parcels_CS_delivered) +
            " have been delivered through CS (=" +
            str(delivered_percentage) + "%)." + "\n") +
        (
            "\t\tThe average distance of CS parcel trips is " +
            str(avg_dist) +
            "km. For the delivered parcels, the average detour is " +
            str(avg_detour) +
            "km." + "\n") +
        (
            "\t\tFor the CS deliveries, " +
            str(total_detour) +
            " extra kilometers are driven." +
            " The detours are distributed to modes as follows:" + "\n") +
        (
            "\t\t\tBike: " +
            str(bike_parcels) +
            " parcels, total of " +
            str(bike_km) +
            "km (" +
            str(bike_km_avg) +
            "km average) \n") +
        (
            "\t\t\tCar:  " +
            str(car_parcels) +
            " parcels, total of " +
            str(car_km) +
            "km (" +
            str(car_km_avg) +
            "km average) \n") +
        (
            "\t\tThe average provided compensation for the occasional" +
            " carriers is " + str(avg_compensation) + " euro."))
    logger.debug(message)

    totaltime_cs = round(time.time() - start_time_cs, 2)
    logger.debug(f"\t\tCrowdshipping calculations took: {totaltime_cs} seconds.")

    if root is not None:
        root.progressBar['value'] = 55.0
