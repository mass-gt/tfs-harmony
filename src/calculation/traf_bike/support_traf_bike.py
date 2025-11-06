import concurrent.futures
import logging
import numpy as np
import pandas as pd
import psutil
import shapefile
import struct

from numba import njit
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import shape as shapely_shape
from typing import Dict, List, Tuple

from calculation.common.dijkstra import get_prev


logger = logging.getLogger("tfs")


def get_available_memory():
    """ Returns available memory in MB. """
    return psutil.virtual_memory().available / (1024 * 1024)


def read_shape(
    shape_path: str,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    sf = shapefile.Reader(shape_path, encoding=encoding)
    records = sf.records()
    fields = sf.fields
    shapes = sf.shapes()  # Get geometries
    sf.close()

    columns = [x[0] for x in fields[1:]]
    col_types = [x[1] for x in fields[1:]]
    decimals = [x[3] for x in fields[1:]]
    nFeatures = len(records)
    nFields = len(columns)

    shape_data = np.zeros((nFeatures, nFields), dtype=object)
    for i in range(nFeatures):
        shape_data[i, :] = records[i][0:]

    df = pd.DataFrame(shape_data, columns=columns)

    for col in range(nFields):
        if col_types[col] == 'C':  # Character field -> string
            df[columns[col]] = df[columns[col]].astype(str)
        elif col_types[col] == 'N':  # Numeric field
            if decimals[col] > 0:
                df[columns[col]] = df[columns[col]].astype(float)
            else:
                df[columns[col]] = df[columns[col]].astype(int)
        elif col_types[col] == 'D':
            df[columns[col]] = pd.to_datetime(df[columns[col]])

    # Add geometry column
    df['geometry'] = [shapely_shape(s.__geo_interface__) for s in shapes]

    return df


def write_shape(df: pd.DataFrame, output_path: str) -> None:
    with shapefile.Writer(output_path) as shp:
        # Define field types based on DataFrame dtypes
        for col, dtype in df.dtypes.items():
            if col != 'geometry':
                if dtype == 'int64':
                    shp.field(col, 'N', decimal=0)  # Integer
                elif dtype == 'float64':
                    shp.field(col, 'F', decimal=6)  # Float
                else:
                    shp.field(col, 'C')  # Character (string)

        # Write records and geometry
        for _, row in df.iterrows():
            geom = row['geometry']

            # Convert list to appropriate shapely geometry
            if isinstance(geom, list) and len(geom) >= 2:
                if len(geom) >= 3 and geom[0] == geom[-1]:  # Closed, assume Polygon
                    geom = Polygon(geom)
                else:  # Assume LineString
                    geom = LineString(geom)

            if isinstance(geom, (Point, LineString, Polygon)) and not geom.is_empty:
                shp.record(*[row[col] for col in df.columns if col != 'geometry'])
                shp.shape(geom)


def create_parking_heatmap_shape(trip_matrix: pd.DataFrame, nodes: pd.DataFrame, parking_heatmap_path: str) -> None:
    trip_matrix = trip_matrix[['POSITION', 'ORIGIN_ZONE', 'DESTINATION_ZONE', 'ORIGIN_NODE', 'DESTINATION_NODE', 'NTRIPS_CABI']]
    trip_matrix = trip_matrix.loc[~trip_matrix['NTRIPS_CABI'].isnull()]

    trip_matrix = trip_matrix.groupby('DESTINATION_NODE')['NTRIPS_CABI'].sum().reset_index()

    vals_dict = dict(zip(trip_matrix['DESTINATION_NODE'].astype(int), trip_matrix['NTRIPS_CABI']))

    nodes['park_val'] = nodes['NODENR'].map(vals_dict)
    nodes = nodes.loc[~nodes['park_val'].isnull()]

    write_shape(nodes, parking_heatmap_path)


def writeTestMTX(testMatrix, mtxPath: str, nzones: int) -> None:
    """ Writes a binary test OD matrix with a single nonzero value. """
    o, d = testMatrix[0] - 1, testMatrix[1] - 1

    # Create an empty OD matrix and set the test value
    odmat = np.zeros((nzones, nzones), dtype=np.float32)
    odmat[o - 1, d - 1] = 1

    odlist = odmat.flatten()

    with open(mtxPath, 'wb') as file:
        # headerType = np.dtype([('soort',np.uint8),('nZones',np.uint16),('zeros',np.int32)])
        file.write(struct.pack('<BHi', np.uint8(1), np.uint16(nzones), np.int32(0)))
        file.write(struct.pack(f'{len(odlist)}f', *odlist))

    logger.debug(f'\tTest MXT file with [{o},{d}]=1 written to {str(mtxPath)}', extra={'memory_usage': '-'})


def writeMTX(
    output_path: str,
    times: List[str],
    label: str,
    nzones: int,
    micro_vt: int,
) -> None:
    """ Writes a binary OD matrix. """
    time_periods = {
        "OS": [7, 8],
        "RD": [i for i in range(0, 24) if i not in [7, 8, 16, 17]], 
        "AS": [16, 17],
    }

    for time_period in times:
        df_list = []

        for hour in time_periods[time_period]:
            df_list.append(pd.read_csv(
                f"{output_path}tripmatrix_parcels_{label}_{micro_vt}_tourtype2_TOD{hour}.txt",
                sep='\t',
            ))
        df = pd.concat(df_list)
        df = df.groupby(['O_zone', 'D_zone'])[['first', 'intermediate', 'last']].sum().reset_index()

        for position in ['first', 'intermediate', 'last']:
            odmat = np.zeros((nzones, nzones), dtype=np.float32)
            for _, row in df.iterrows():
                odmat[row["O_zone"] - 1, row["D_zone"] - 1] = row[position]

            odlist = odmat.flatten()
            new_path = f"{output_path}cargo_bikes_{time_period}_{label}_{position}.MTX"
            with open(new_path, 'wb') as file:
                file.write(struct.pack('<BHi', np.uint8(1), np.uint16(nzones), np.int32(0)))
                file.write(struct.pack(f'{len(odlist)}f', *odlist))

            logger.debug(f'\tMTX file for cargo bikes written to {str(new_path)}', extra={'memory_usage': '-'})


def process_mat(mtx_path, headerType, zone2idx) -> np.ndarray:
    # Read OD matrix
    header = np.fromfile(mtx_path, dtype=headerType, count=1)[0]
    nZones = header['nZones']

    data = np.fromfile(mtx_path, dtype=np.float32, offset=7)
    mat = data.reshape((nZones, nZones))
    odMat = np.zeros((nZones + 1, nZones + 1), dtype=np.float32)
    odMat[1:, 1:] = mat
    np.fill_diagonal(odMat, 0)

    cMat = compress_mat(odMat, zone2idx)

    return cMat


def compress_mat(
    mat: np.ndarray,
    zone2idx: Dict[int, int],
) -> np.ndarray:
    mappedI = [i for i in zone2idx.keys()]
    newI = [zone2idx[i] for i in mappedI]
    cMat = mat[np.ix_(mappedI, mappedI)]
    cMat = cMat[np.ix_(newI, newI)]
    return cMat


def createLinkDict(
    linkArray: np.ndarray,
    nodeArray: np.ndarray,
    maxConnections: int
) -> np.ndarray:
    nNodes = len(nodeArray)
    linkDict = -1 * np.ones((nNodes, maxConnections * 2), dtype=np.int32)

    for i in range(linkArray.shape[0]):
        a = int(linkArray[i, 0])
        b = int(linkArray[i, 1])

        for col in range(maxConnections):
            if linkDict[a, col] == -1:
                linkDict[a, col] = b
                linkDict[a, col + maxConnections] = i
                break

    return linkDict


@njit
def get_route(
    o: int,
    d: int,
    prev: np.ndarray,
    linkDict: np.ndarray,
    maxConnections: int,
) -> np.ndarray:
    # Deduce sequence of nodes on network
    sequenceNodes: List[int] = []
    destNode = d
    if prev[o][destNode] >= 0:
        while prev[o][destNode] >= 0:
            sequenceNodes.insert(0, destNode)
            destNode = prev[o][destNode]
        else:
            sequenceNodes.insert(0, destNode)

    # Deduce sequence of links on network
    route = []

    if len(sequenceNodes) > 1:
        for i in range(len(sequenceNodes) - 1):
            aNode = sequenceNodes[i]
            bNode = sequenceNodes[i + 1]

            tmp = linkDict[aNode]
            for col in range(maxConnections):
                if tmp[col] == bNode:
                    route.append(tmp[col + maxConnections])
                    break

    return np.array(route, dtype=np.int32)


def calc_loads(
    cMatDict: Dict[str, np.ndarray],
    linksAB: np.ndarray,
    linkDict: np.ndarray,
    prev: np.ndarray,
    prev_nodetonode: np.ndarray,
    maxConnections: int,
    parcel_counts: pd.DataFrame,
    micro_nodes: pd.DataFrame,
    idx2zone: Dict[int, int],
    idx2node: Dict[int, int],
    u: str,
    use_micronodes: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Computes link loads based on shortest paths from an OD matrix."""
    loads = np.zeros(len(linksAB), dtype=float)
    parcel_shares = parcel_counts.pivot_table(index='D_zone', columns='Receiver', values='Share', fill_value=0)
    micronodes_by_zone = dict(tuple(micro_nodes.groupby('AREANR')))
    ntrips_list = []

    for position, mat in cMatDict.items():
        for o, d in list(zip(*np.nonzero(mat))):
            try:
                val = mat[o, d]
                origin_zone = idx2zone[o]
                destination_zone = idx2zone[d]

                if not (u == 'CABI' and use_micronodes):
                    route = get_route(o, d, prev, linkDict, maxConnections)
                    if len(route) == 0:
                        print(f'No route found for O-D {origin_zone}-{destination_zone}!\n')
                        continue
                    loads[route] += val
                    ntrips_list.append((position, origin_zone, destination_zone, idx2node[o], idx2node[d], val))
                    continue

                empl_share_orig = parcel_shares.loc[origin_zone].get('empl', 0) if destination_zone in parcel_shares.index else 0
                inhab_share_orig = parcel_shares.loc[origin_zone].get('inhab', 0) if destination_zone in parcel_shares.index else 0

                origin_node_nrs = micronodes_by_zone.get(origin_zone)

                if origin_node_nrs is not None:
                    inhab_nodes = origin_node_nrs[origin_node_nrs['WOON'] != 0][['NODENR_INDEX', 'WOON']].copy()
                    inhab_nodes['Share'] = inhab_share_orig * (inhab_nodes['WOON'] / inhab_nodes['WOON'].sum())

                    empl_nodes = origin_node_nrs[origin_node_nrs['KANTOOR'] != 0][['NODENR_INDEX', 'KANTOOR']].copy()
                    empl_nodes['Share'] = empl_share_orig * (empl_nodes['KANTOOR'] / empl_nodes['KANTOOR'].sum())

                    origin_indexes = pd.concat([
                        inhab_nodes[['NODENR_INDEX', 'Share']],
                        empl_nodes[['NODENR_INDEX', 'Share']]],
                    ).groupby('NODENR_INDEX', as_index=False)['Share'].sum()

                else:
                    origin_indexes = pd.DataFrame({
                        "NODENR_INDEX": [o], "Share": [1]
                    }).groupby('NODENR_INDEX', as_index=False)['Share'].sum()

                if len(origin_indexes) == 1 and origin_indexes['Share'].sum() != 1:
                    origin_indexes['Share'] = 1

                empl_share_dest = parcel_shares.loc[destination_zone].get('empl', 0) if destination_zone in parcel_shares.index else 0
                inhab_share_dest = parcel_shares.loc[destination_zone].get('inhab', 0) if destination_zone in parcel_shares.index else 0

                destination_node_nrs = micronodes_by_zone.get(destination_zone)

                if destination_node_nrs is not None:
                    inhab_nodes = destination_node_nrs[destination_node_nrs['WOON'] != 0][['NODENR_INDEX', 'WOON']].copy()
                    inhab_nodes['Share'] = inhab_share_dest * (inhab_nodes['WOON'] / inhab_nodes['WOON'].sum())

                    empl_nodes = destination_node_nrs[destination_node_nrs['KANTOOR'] != 0][['NODENR_INDEX', 'KANTOOR']].copy()
                    empl_nodes['Share'] = empl_share_dest * (empl_nodes['KANTOOR'] / empl_nodes['KANTOOR'].sum())

                    destination_indexes = pd.concat([
                        inhab_nodes[['NODENR_INDEX', 'Share']],
                        empl_nodes[['NODENR_INDEX', 'Share']]],
                    ).groupby('NODENR_INDEX', as_index=False)['Share'].sum()

                else:
                    destination_indexes = pd.DataFrame({
                        "NODENR_INDEX": [d], "Share": [1]
                    }).groupby('NODENR_INDEX', as_index=False)['Share'].sum()

                if len(destination_indexes) == 1 and destination_indexes['Share'].sum() != 1:
                    destination_indexes['Share'] = 1

                if position == 'first':
                    origin_node = idx2node[o]
                    orig_node_index = o
                    rows = destination_indexes[['NODENR_INDEX', 'Share']].values

                    summed_val = 0
                    for i, (dest_node_index, share) in enumerate(rows):
                        destination_node = idx2node[int(dest_node_index)]

                        is_last = (i == len(rows) - 1)
                        split_val = val - summed_val if is_last else val * share
                        summed_val += split_val

                        route = get_route(int(orig_node_index), int(dest_node_index), prev, linkDict, maxConnections)
                        if len(route) == 0:
                            print(
                                f"Micronodes: No route found for position {position}, O-D {origin_zone}-{destination_zone}, " +
                                f"origin_node = {origin_node}, destination_node = {destination_node}!\n"
                            )
                            continue
                        loads[route] += split_val

                        ntrips_list.append((position, origin_zone, destination_zone, origin_node, destination_node, split_val))

                elif position == 'last':
                    destination_node = idx2node[d]
                    dest_node_index = d
                    rows = origin_indexes[['NODENR_INDEX', 'Share']].values

                    summed_val = 0
                    for i, (orig_node_index, share) in enumerate(rows):
                        origin_node = idx2node[int(orig_node_index)]

                        is_last = (i == len(rows) - 1)
                        split_val = val - summed_val if is_last else val * share
                        summed_val += split_val

                        try:
                            route = get_route(
                                micro_nodes[micro_nodes['NODENR_INDEX'] == int(orig_node_index)].index[0],
                                dest_node_index, prev_nodetonode, linkDict, maxConnections,
                            )
                        except IndexError:
                            route = get_route(int(orig_node_index), int(dest_node_index), prev, linkDict, maxConnections)

                        if len(route) == 0:
                            print(
                                f"Micronodes: No route found for position {position}, O-D {origin_zone}-{destination_zone}, " +
                                f"origin_node = {origin_node}, destination_node = {destination_node}!\n"
                            )
                            continue

                        loads[route] += split_val

                        ntrips_list.append((position, origin_zone, destination_zone, origin_node, destination_node, split_val))

                else:

                    origin_rows = origin_indexes[['NODENR_INDEX', 'Share']]
                    destination_rows = destination_indexes[['NODENR_INDEX', 'Share']]
                    pairs = [(o, d) for o in origin_rows['NODENR_INDEX'] for d in destination_rows['NODENR_INDEX']]

                    micro_od_df = pd.DataFrame(pairs, columns=['origin', 'destination'])
                    micro_od_df['origin'] = micro_od_df['origin'].astype(int)
                    micro_od_df['destination'] = micro_od_df['destination'].astype(int)

                    micro_od_df['val'] = val * np.outer(origin_rows['Share'], destination_rows['Share']).flatten()

                    same_od = micro_od_df.loc[micro_od_df['origin'] == micro_od_df['destination']]
                    for i, line in same_od.iterrows():
                        same_val = line['val']
                        same_dest = line['destination']
                        same_index = i

                        micro_od_df = micro_od_df.drop(same_index)

                        len_same_dest = len(micro_od_df.loc[micro_od_df['destination'] == same_dest, 'val'])
                        micro_od_df.loc[micro_od_df['destination'] == same_dest, 'val'] += (same_val / len_same_dest)

                    micro_od_df = micro_od_df.reset_index()
                    for i, row in micro_od_df.iterrows():
                        orig_node_index = row['origin']
                        origin_node = idx2node[orig_node_index]

                        dest_node_index = int(row['destination'])
                        destination_node = idx2node[dest_node_index]

                        try:
                            route = get_route(
                                micro_nodes[micro_nodes['NODENR_INDEX'] == int(orig_node_index)].index[0],
                                dest_node_index, prev_nodetonode, linkDict, maxConnections,
                            )
                        except IndexError:
                            route = get_route(int(orig_node_index), int(dest_node_index), prev, linkDict, maxConnections)
                        if len(route) == 0:
                            print(f'Micronodes: No route found for position {position}, O-D {origin_zone}-{destination_zone}, origin_node = {origin_node}, destination_node = {destination_node}!\n')
                            continue
                        loads[route] += row['val']

                        ntrips_list.append((position, origin_zone, destination_zone, origin_node, destination_node, row['val']))

            except Exception as e:
                raise Exception(f"Error occurred at o={o}, d={d}. Error message: {e}") from e

    ntrips = pd.DataFrame(ntrips_list, columns=['POSITION', 'ORIGIN_ZONE', 'DESTINATION_ZONE', 'ORIGIN_NODE', 'DESTINATION_NODE', f'NTRIPS_{u}'])

    return loads, ntrips


def cpu_bound_task(userClassInputx):

    (
        cpu, t, u, zoneArray, linksAB, costArray, maxConnections, cMatDict, linkDict,
        idx_u, idx2node, idx2zone,
        parcel_counts, micro_nodes, use_micronodes, output_path,
    ) = userClassInputx

    print(f'({cpu}) - Running time period {t} for user class {u}')

    print(f'\t({cpu}) - Finding node-to-node routes for {len(micro_nodes)} origins')

    nodetonodeArray = micro_nodes['NODENR_INDEX'].values
    prev_nodetonode = get_prev(nodetonodeArray, linksAB, costArray, False, maxConnections)

    print(f'\t({cpu}) - Finding zone-to-zone routes for {len(zoneArray)} origins')

    # prev_path = output_path / f"prev_{u}_{t}.npy"
    # if os.path.exists(prev_path):
    #     prev = np.load(prev_path)
    # else:
    #     prev = get_prev(zoneArray, linksAB, costArray, False, maxConnections)
    #     np.save(prev_path, prev)
    prev = get_prev(zoneArray, linksAB, costArray, False, maxConnections)

    print(f'\t({cpu}) - Calculating loads')

    loads, ntrips = calc_loads(
        cMatDict, linksAB, linkDict, prev, prev_nodetonode, maxConnections,
        parcel_counts, micro_nodes, idx2zone, idx2node, u, use_micronodes,
    )

    print(f'\t({cpu}) - Converting links to output format')

    loads2d = loads[:, np.newaxis]
    abLoad = np.hstack((linksAB, loads2d))
    ab_with_load = abLoad[abLoad[:, -1] > 0]
    abLoadDF = pd.DataFrame(ab_with_load, columns=['a', 'b', f'LOAD_{t}{idx_u}'])

    abLoadDF['a_map'] = abLoadDF['a'].map(idx2node)
    abLoadDF['b_map'] = abLoadDF['b'].map(idx2node)
    abLoadDF = abLoadDF.drop(columns=['a', 'b'])

    print(f'\t({cpu}) - Done!')

    return [abLoadDF, ntrips]


def run_in_batches(run_inputs, memory_per_run):
    total_runs = len(run_inputs)
    available_memory = get_available_memory()
    max_parallel_runs = min(max(int(available_memory / memory_per_run), 1), psutil.cpu_count(logical=False))
    n_batches = int(np.ceil(total_runs / max_parallel_runs))

    if max_parallel_runs < 1:
        raise MemoryError('Not enough memory available for 1 run')

    logger.debug(f'\tRunning {total_runs} run(s) in {n_batches} batches', extra={'memory_usage': '-'})

    all_results = pd.DataFrame()
    all_trips = pd.DataFrame()

    for batch_no, start in enumerate(range(0, total_runs, max_parallel_runs)):
        batch = range(start, min(start + max_parallel_runs, total_runs))
        logger.debug(f'\tRunning batch {batch_no}: {list(batch)}', extra={'memory_usage': '-'})

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel_runs) as executor:
                results = list(executor.map(cpu_bound_task, run_inputs[start:start + max_parallel_runs]))

            for run_results in results:
                if all_results.empty:
                    all_results = run_results[0]
                    all_trips = run_results[1]
                else:
                    all_results = pd.merge(all_results, run_results[0], on=['a_map', 'b_map'], how='outer')
                    all_trips = pd.merge(
                        all_trips, run_results[1],
                        on=['POSITION', 'ORIGIN_ZONE', 'DESTINATION_ZONE', 'ORIGIN_NODE', 'DESTINATION_NODE'],
                        how='outer',
                    )

        except Exception as e:
            logger.error(f"Error during batch {batch_no}: {e}")

    return (all_results, all_trips)
