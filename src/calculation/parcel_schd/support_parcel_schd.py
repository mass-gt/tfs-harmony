import numpy as np
import pandas as pd
import logging
import time

from tqdm import tqdm
from typing import Any, Dict, List

logger = logging.getLogger("tfs")


def create_schedules(
    varDict: Dict[Any, Any],
    parcelsAgg: pd.DataFrame,
    dropOffTime: float,
    skimTravTime: np.ndarray,
    skimDistance: np.ndarray,
    parcelNodesCEP: Dict[int, str],
    parcelDepTime: np.ndarray,
    tourType: int,
    tourType_origin: str,
    seed: int,
    root: Any,
    startValueProgress: float,
    endValueProgress: float,
) -> pd.DataFrame:
    """Create the parcel schedules and store them in a DataFrame."""
    nZones = int(len(skimTravTime)**0.5)
    hubs = parcelsAgg[tourType_origin].unique()

    if 'MIC' in varDict['LABEL']:
        microhubs = pd.read_csv(varDict['MICROHUBS'])

    tours, parcelsDelivered, departureTimes = {}, {}, {}

    depotCount = 0
    nDepots = len(hubs)

    for hub in tqdm(hubs, desc="Processing hubs", unit="hub", ncols=80):
        hubParcels = parcelsAgg[parcelsAgg[tourType_origin] == hub]

        tours[hub], parcelsDelivered[hub], departureTimes[hub] = {}, {}, {}

        for cluster in hubParcels['Cluster'].unique():
            clusterParcels = hubParcels[hubParcels['Cluster'] == cluster]
            vehicle = clusterParcels['VEHTYPE'].iloc[0]

            hubZone = clusterParcels['Orig'].iloc[0]
            destZones = clusterParcels['Dest'].tolist()

            nParcelsPerZone = dict(zip(destZones, clusterParcels['Parcels']))

            # Nearest neighbor
            tour = [hubZone]
            while destZones:
                distances = skimDistance[(tour[-1] - 1) * nZones + np.array(destZones) - 1]
                nextIndex = np.argmin(distances)
                tour.append(destZones.pop(nextIndex))
            tour.append(hubZone)

            # Optimize tour ordering (2-opt-like swap)
            tour = np.array(tour, dtype=int)
            tourDist = skimDistance[(tour[:-1] - 1) * nZones + (tour[1:] - 1)].sum()

            if len(tour) > 4:
                for shiftLocA in range(1, len(tour) - 2):
                    for shiftLocB in range(shiftLocA + 1, len(tour) - 1):
                        swappedTour = tour.copy()
                        swappedTour[shiftLocA], swappedTour[shiftLocB] = swappedTour[shiftLocB], swappedTour[shiftLocA]
                        swappedTourDist = skimDistance[(swappedTour[:-1] - 1) * nZones + (swappedTour[1:] - 1)].sum()
                        if swappedTourDist < tourDist:
                            tour, tourDist = swappedTour, swappedTourDist

            # Add current tour to dictionary with all formed tours
            tours[hub][cluster] = [tour.tolist(), vehicle]

            # Store the number of parcels delivered at each
            # location in the tour
            parcelsDelivered[hub][cluster] = [
                nParcelsPerZone.get(t, 0) for t in tour[1:-1]] + [0]

            # Determine the departure time of each trip in the tour
            np.random.seed(seed + 10000 * hub + cluster)
            np.random.seed(np.random.randint(10000000))

            departureTimesTour = [
                np.searchsorted(parcelDepTime, np.random.rand()) + np.random.rand()]

            for i in range(1, len(tour)):
                travTime = skimTravTime[(tour[i - 1] - 1) * nZones + (tour[i] - 1)]
                departureTimesTour.append(
                    departureTimesTour[-1] +
                    dropOffTime * parcelsDelivered[hub][cluster][i - 1] + travTime)

            departureTimes[hub][cluster] = departureTimesTour

        if root is not None:
            root.progressBar['value'] = (
                startValueProgress +
                (endValueProgress - startValueProgress - 1) * (depotCount + 1) / nDepots)

        depotCount += 1

    # --------------------------- Create return table -------------------------
    deliveriesCols = [
        'TourType', 'CEP', 'Depot_ID', 'MH_ID',
        'Tour_ID', 'Trip_ID', 'Unique_ID',
        'O_zone', 'D_zone', 'N_parcels',
        'Traveltime', 'Distance', 'TourDepTime', 'TripDepTime', 'TripEndTime', 'Vehicle']

    deliveries = []

    for hub, tour_data in tqdm(tours.items(), desc="Compiling deliveries", unit="tour", ncols=80):
        for tour, (tourStops, vehicle) in tour_data.items():
            for trip in range(len(tourStops) - 1):
                orig, dest = tourStops[trip], tourStops[trip + 1]
                deliveries.append([
                    tourType,
                    parcelNodesCEP.get(hub, np.nan) if tourType <= 1 else (
                        microhubs.loc[microhubs.ID == hub, 'CEP'].values[0] if 'MIC' in varDict['LABEL'] else 'ConsolidatedUCC'
                    ),
                    hub if tourType <= 1 else np.nan,
                    np.nan if tourType <= 1 else hub,
                    f'{hub}_{tour}',
                    f'{hub}_{tour}_{trip}',
                    f'{hub}_{tour}_{trip}_{tourType}',
                    orig,
                    dest,
                    parcelsDelivered[hub][tour][trip],
                    skimTravTime[(orig - 1) * nZones + (dest - 1)],
                    skimDistance[(orig - 1) * nZones + (dest - 1)] / 1000,
                    departureTimes[hub][tour][0],
                    departureTimes[hub][tour][trip],
                    departureTimes[hub][tour][trip + 1],
                    vehicle
                ])

    deliveries_df = pd.DataFrame(deliveries, columns=deliveriesCols)
    deliveries_df = deliveries_df.astype({
        'TourType': int, 'CEP': str, 'Depot_ID': float, 'MH_ID': float,
        'Tour_ID': str, 'Trip_ID': str, 'Unique_ID': str,
        'O_zone': int, 'D_zone': int, 'N_parcels': int,
        'Traveltime': float, 'Distance': float, 'TourDepTime': float,
        'TripDepTime': float, 'TripEndTime': float, 'Vehicle': int
    })

    # Add OrigType and DestType
    if 'MIC' in varDict['LABEL']:
        OD_tourtype_Dic = {0: ['Depot', 'HH'], 1: ['Depot', 'MH'], 2: ['MH', 'HH']}
    elif varDict['LABEL'] == 'USE_CASE_REF':
        OD_tourtype_Dic = {0: ['Depot', 'HH']}
    else:
        OD_tourtype_Dic = {
            0: ['Depot', 'HH'], 1: ['Depot', 'UCC'],
            2: ['UCC', 'HH'], 3: ['UCC', 'HH']
        }
        deliveries_df['VehType'] = ['Van', 'Van', 'Van', 'LEVV'][tourType]

    deliveries_df['OrigType'] = deliveries_df['TourType'].map(lambda x: OD_tourtype_Dic.get(x, [None, None])[0])
    deliveries_df['DestType'] = deliveries_df['TourType'].map(lambda x: OD_tourtype_Dic.get(x, [None, None])[1])

    return deliveries_df


def cluster_parcels(
    varDict: Dict[str, Any],
    parcels: pd.DataFrame,
    typeoftour: int,
    skimDistance: np.ndarray,
    root: Any,
    startValueProgress: float,
    endValueProgress: float,
) -> pd.DataFrame:
    '''
    Assign parcels to clusters based on spatial proximity with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.
    '''
    nParcels = len(parcels)

    if 'MIC' in varDict['LABEL']:
        typeoftoursdic = {0: 'DepotNumber', 1: 'DepotNumber', 2: 'FROM_MH'}
        microhub_vt_capacity = {
            int(row['Veh_ID']): int(row['Capacity'])
            for row in pd.read_csv(varDict['VEHICLETYPES']).to_dict('records')
        }
    elif varDict['LABEL'] == 'UCC':
        typeoftoursdic = {0: 'DepotNumber', 1: 'DepotNumber', 2: 'FROM_UCC', 3: 'FROM_UCC'}
    else:
        typeoftoursdic = {0: 'DepotNumber'}

    # vehicletype = parcels.VEHTYPE.unique()
    parcels['Cluster'] = -1
    parcels.index = np.arange(len(parcels))
    nParcelsAssigned = 0
    firstClusterID = 0
    nZones = int(len(skimDistance)**0.5)

    for veh in parcels['VEHTYPE'].unique():
        parcels_veh: pd.DataFrame = parcels[parcels['VEHTYPE'] == veh].copy()

        # Determine vehicle capacity
        if 'MIC' in varDict['LABEL']:
            maxVehicleLoad = varDict['PARCELS_MAXLOAD'] if typeoftour == 0 else microhub_vt_capacity[veh]
        elif varDict['LABEL'] == 'UCC':
            maxVehicleLoad = varDict['PARCELS_MAXLOAD'] / 5 if typeoftour == 4 else varDict['PARCELS_MAXLOAD']
        else:
            maxVehicleLoad = varDict['PARCELS_MAXLOAD']
        maxVehicleLoad = int(maxVehicleLoad)

        counts = parcels_veh.groupby([typeoftoursdic[typeoftour], 'D_zone']).size()

        # Large cluster assignment (where parcels exceed max load)
        for (hub, destZone), _ in counts[counts >= maxVehicleLoad].items():
            indices = parcels_veh.index[
                (parcels_veh[typeoftoursdic[typeoftour]] == hub) &
                (parcels_veh['D_zone'] == destZone)
            ]

            while len(indices) >= maxVehicleLoad:
                parcels.loc[indices[:maxVehicleLoad], 'Cluster'] = firstClusterID
                indices = indices[maxVehicleLoad:]

                firstClusterID += 1
                nParcelsAssigned += maxVehicleLoad

            if root is not None:
                root.progressBar['value'] = (
                    startValueProgress +
                    (endValueProgress - startValueProgress - 1) * nParcelsAssigned / nParcels)

        # Cluster remaining parcels
        hubs = np.sort(parcels[typeoftoursdic[typeoftour]].unique())
        for hub in hubs:
            parcelsToFit: pd.DataFrame = parcels.loc[
                (parcels[typeoftoursdic[typeoftour]] == hub) & (parcels['Cluster'] == -1)
            ].copy()

            if parcelsToFit.empty:
                continue

            # Compute distance to depot and sort descending
            parcelsToFit['Distance'] = skimDistance[
                (parcelsToFit['O_zone'] - 1) * nZones + (parcelsToFit['D_zone'] - 1)]

            parcelsToFit = parcelsToFit.sort_values('Distance', ascending=False)

            nTours = int(np.ceil(len(parcelsToFit) / maxVehicleLoad))

            # If only one tour is needed
            if nTours == 1:
                parcels.loc[parcelsToFit.index, 'Cluster'] = firstClusterID
                firstClusterID += 1
                nParcelsAssigned += len(parcelsToFit)
                continue

            unassigned_parcels = set(parcelsToFit.index)

            # Multi-tour case: Cluster by proximity
            with tqdm(
                total=nTours, desc=f"  Assigning parcels to clusters for hub {hub}",
                unit="cluster", leave=False, dynamic_ncols=True, ascii=True
            ) as cluster_pbar:
                for _ in range(nTours):
                    if not unassigned_parcels:
                        break

                    # Select the furthest unassigned parcel as the seed for the new cluster
                    seed_parcel_idx = parcelsToFit.loc[list(unassigned_parcels), 'Distance'].idxmax()
                    cluster_parcels = {seed_parcel_idx}
                    unassigned_parcels.remove(seed_parcel_idx)

                    # Compute distances from the seed parcel to all other unassigned parcels
                    parcelsToFit['Distance_Relative'] = skimDistance[
                        (parcelsToFit.loc[seed_parcel_idx, 'D_zone'] - 1) * nZones +
                        (parcelsToFit['D_zone'] - 1)
                    ]

                    # Find the closest parcels (excluding the seed), up to vehicle capacity
                    nearest_parcels = (
                        parcelsToFit.loc[list(unassigned_parcels), 'Distance_Relative']
                        .nsmallest(n=maxVehicleLoad - 1)
                        .index
                    )

                    cluster_parcels.update(nearest_parcels)
                    unassigned_parcels.difference_update(nearest_parcels)

                    parcels.loc[list(cluster_parcels), 'Cluster'] = firstClusterID
                    firstClusterID += 1
                    cluster_pbar.update(1)

            # Assign any remaining parcels to a final cluster
            if unassigned_parcels:
                parcels.loc[list(unassigned_parcels), 'Cluster'] = firstClusterID
                firstClusterID += 1

            nParcelsAssigned += len(parcelsToFit)

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
    """Write the parcel schedules to a geojson file with coordinates."""
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


def export_trip_matrices(deliveries: pd.DataFrame, varDict: Dict[str, Any]) -> None:
    """Aggregate deliveries to trip matrices and export to output folder."""
    deliveries['N_TOT'] = 1

    for veh in deliveries['Vehicle'].unique():
        df = deliveries[deliveries['Vehicle'] == veh].copy()
        tour_type = df['TourType'].iloc[0]
        df['TripDepTime'] = df['TripDepTime'] % 24  # Wrap times over 24

        is_veh14 = veh == 14

        if is_veh14:
            # Add trip index and classify
            df['trip_index'] = df['Trip_ID'].apply(lambda x: int(x.split('_')[-1]))
            df = df.sort_values(['Tour_ID', 'trip_index'])
            df['first_trip'] = df.groupby('Tour_ID')['trip_index'].transform('min') == df['trip_index']
            df['last_trip'] = df.groupby('Tour_ID')['trip_index'].transform('max') == df['trip_index']
            df['trip_type'] = df.apply(
                lambda row: 'first' if row['first_trip'] else ('last' if row['last_trip'] else 'intermediate'),
                axis=1
            )

        # === LOOP OVER TOD ===
        for tod in range(24):
            df_tod = df[(df['TripDepTime'] >= tod) & (df['TripDepTime'] < tod + 1)].copy()
            if df_tod.empty:
                continue

            if is_veh14:
                pivot = pd.pivot_table(
                    df_tod,
                    values='N_TOT',
                    index=['O_zone', 'D_zone'],
                    columns='trip_type',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
            else:
                pivot = df_tod.groupby(['O_zone', 'D_zone'])['N_TOT'].sum().reset_index()

            # Save TOD matrix
            pivot.to_csv(
                f"{varDict['OUTPUTFOLDER']}tripmatrix_parcels_{varDict['LABEL']}_{veh}_tourtype{tour_type}_TOD{tod}.txt",
                index=False, sep='\t'
            )

        # === FULL-DAY MATRIX ===
        if is_veh14:
            pivot_day = pd.pivot_table(
                df,
                values='N_TOT',
                index=['O_zone', 'D_zone'],
                columns='trip_type',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
        else:
            pivot_day = df.groupby(['O_zone', 'D_zone'])['N_TOT'].sum().reset_index()

        # Save full-day matrix
        pivot_day.to_csv(
            f"{varDict['OUTPUTFOLDER']}tripmatrix_parcels_{varDict['LABEL']}_{veh}_tourtype{tour_type}.txt",
            index=False, sep='\t'
        )


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


def create_summary(
    parcels: pd.DataFrame,
    deliveries: pd.DataFrame,
    microhubs: pd.DataFrame,
    tier: str,
) -> pd.DataFrame:
    """Create a table of summary statistics per microhub."""
    avg_shift_duration = 8

    mh_to_cep = {row['ID']: row['CEP'] for row in microhubs.to_dict('records')}

    pivot_from_mh = parcels['FROM_MH'].value_counts().to_dict()
    pivot_to_mh = parcels['TO_MH'].value_counts().to_dict()

    summary_list: List[List[Any]] = []

    for mc_id in set(list(pivot_from_mh.keys()) + list(pivot_to_mh.keys())):
        if mc_id <= 0:
            continue

        n_parcels_arriving = pivot_to_mh.get(mc_id, 0.0)
        n_parcels_departing = pivot_from_mh.get(mc_id, 0.0)

        subset_tours = deliveries.loc[deliveries['MH_ID'] == mc_id, :].copy()

        n_trips = subset_tours.shape[0]

        where_tour = subset_tours.groupby('Tour_ID').apply(lambda x: x.index.tolist()).to_dict()
        n_tours = len(where_tour)

        tour_durations = [
            subset_tours.at[rows[-1], 'TripEndTime'] - subset_tours.at[rows[0], 'TourDepTime']
            for tour_id, rows in where_tour.items()
        ]
        avg_tour_duration = np.mean(tour_durations)
        avg_num_tours_in_shift = avg_shift_duration / avg_tour_duration
        required_fleet_size = np.ceil(n_tours / avg_num_tours_in_shift)

        summary_list.append([
            mc_id,
            'Shared' if tier == "Horizontal Collaboration" else mh_to_cep[mc_id],
            n_parcels_arriving,
            n_parcels_departing,
            n_parcels_arriving - n_parcels_departing,
            n_trips,
            n_tours,
            required_fleet_size,
            subset_tours["Traveltime"].sum(),
            sum(tour_durations),
        ])

    summary_list = summary_list + [
        [
            'Total',
            '',
            *[
                sum(summary_list[i][j] for i in range(len(summary_list)))
                for j in range(2, len(summary_list[0]))
            ]
        ]
    ]

    return pd.DataFrame(
        summary_list,
        columns=[
            'Microhub',
            'Courier',
            'Total number of parcels handled',
            'Number of parcels out for delivery',
            'Number of parcels in pick-up lockers',
            'Number of trips',
            'Number of tours',
            'Required fleet size',
            'Total time spent driving (h)',
            'Total time spent overall (h)',
        ],
    )
