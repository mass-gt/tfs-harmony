import array
import logging
import multiprocessing as mp
import numpy as np
import os.path
import pandas as pd
import shapefile as shp

from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger("tfs")


def read_mtx(file_path: str):
    '''
    Read a binary mtx-file (skimTijd and skimAfstand)
    '''
    mtxData = array.array('f')  # f for float
    mtxData.fromfile(
        open(file_path, 'rb'),
        os.path.getsize(file_path) // mtxData.itemsize)

    # The number of zones is in the first byte
    mtxData = np.array(mtxData, dtype=float)[1:]

    return mtxData


def write_mtx(file_path: str, mat: np.ndarray, nZones: int) -> None:
    '''
    Write an array into a binary file
    '''
    mat = np.append(nZones, mat)
    matBin = array.array('f')
    matBin.fromlist(list(mat))
    matBin.tofile(open(file_path, 'wb'))


def read_shape(
    shapePath: str, encoding: str = 'latin1', returnGeometry: bool=False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Any]]]:
    '''
    Read the shapefile with zones (using pyshp --> import shapefile as shp)
    '''
    # Load the shape
    sf = shp.Reader(shapePath, encoding=encoding)
    records = sf.records()
    if returnGeometry:
        geometry = sf.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf.fields
    sf.close()

    # Get information on the fields in the DBF
    columns = [x[0] for x in fields[1:]]
    colTypes = [x[1:] for x in fields[1:]]
    nRecords = len(records)

    # Check for headers that appear twice
    for col in range(len(columns)):
        name = columns[col]
        whereName = [i for i in range(len(columns)) if columns[i] == name]
        if len(whereName) > 1:
            for i in range(1, len(whereName)):
                columns[whereName[i]] = (
                    str(columns[whereName[i]]) + '_' + str(i))

    # Put all the data records into a NumPy array
    # (much faster than Pandas DataFrame)
    shape = np.zeros((nRecords, len(columns)), dtype=object)
    for i in range(nRecords):
        shape[i, :] = records[i][0:]

    # Then put this into a Pandas DataFrame with
    # the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if colTypes[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if colTypes[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)

    if returnGeometry:
        return (shape, geometry)
    else:
        return shape


def get_skims(varDict: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reads the time and distance skim matrix and overwrites values for cells with value zero.
    """
    skimTravTime = read_mtx(varDict['SKIMTIME'])
    skimDistance = read_mtx(varDict['SKIMDISTANCE'])

    if len(skimTravTime) != len(skimDistance):
        raise Exception("Files for 'SKIMTIME' and 'SKIMDISTANCE' do not contain the same number of zones.")

    nZones = int(len(skimTravTime)**0.5)

    skimTravTime[skimTravTime < 0] = 0
    skimDistance[skimDistance < 0] = 0

    # For zero times and distances assume half the value to
    # the nearest (non-zero) zone
    # (otherwise we get problem in the distance decay function)
    for orig in range(nZones):
        whereZero = np.where(skimTravTime[orig * nZones + np.arange(nZones)] == 0)[0]
        whereNonZero = np.where(skimTravTime[orig * nZones + np.arange(nZones)] != 0)[0]
        if len(whereZero) > 0:
            skimTravTime[orig * nZones + whereZero] = 0.5 * np.min(skimTravTime[orig * nZones + whereNonZero])

        whereZero = np.where(skimDistance[orig * nZones + np.arange(nZones)] == 0)[0]
        whereNonZero = np.where(skimDistance[orig * nZones + np.arange(nZones)] != 0)[0]
        if len(whereZero) > 0:
            skimDistance[orig * nZones + whereZero] = 0.5 * np.min(skimDistance[orig * nZones + whereNonZero])
            
    return skimTravTime, skimDistance, nZones


def get_num_cpu(varDict: Dict[str, str], maxCPU: int) -> int:
    """Determines the number of cores over which to parallelize tasks."""
    if varDict['N_CPU'] not in ['', '""', "''"]:
        try:
            nCPU = int(varDict['N_CPU'])
            if nCPU > mp.cpu_count():
                nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
                logger.debug(
                    f"N_CPU parameter too high. Only {mp.cpu_count()} CPUs available. " +
                    f"Hence defaulting to {nCPU} CPUs.")
            if nCPU < 1:
                nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

        except ValueError:
            nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
            logger.debug(
                "Could not convert CPU parameter to an integer. " +
                f"Hence defaulting to {nCPU} CPUs.")
    else:
        nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

    return nCPU


def get_seeds(varDict: Dict[str, str]) -> Dict[str, int]:
    """Reads the file with random seeds."""
    return dict(
        (str(row['step']), int(row['seed']))
        for row in pd.read_csv(varDict['SEEDS'], sep='\t').to_dict('records')
    )
