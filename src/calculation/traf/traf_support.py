
import logging
import numpy as np
import pandas as pd
import shapefile as shp

from itertools import product
from numba import njit, int32
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Any, Dict, List, Tuple

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import get_skims

logger = logging.getLogger("tfs")


def get_emission_factors(
    varDict: Dict[str, str],
    dims: ModelDimensions,
) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
    """
    Reads the file with emission factors and converts them from grams to kilograms.
    """
    emissionFactors = dict(
        ((vt, rt, et), [0.0, 0.0])
        for vt, rt, et in product(
            dims.vehicle_type.keys(), dims.road_type.keys(), dims.emission_type.keys()
        )
    )

    for row in pd.read_csv(varDict['EMISSIONFACS'], sep='\t').to_dict('records'):
        emissionFactors[(
            int(row['vehicle_type']),
            int(row['road_type']),
            int(row['emission_type'])
        )][int(row['is_loaded'])] = float(row['factor_gram_per_km']) / 1000

    return dict((k, tuple(v)) for k, v in emissionFactors.items())


def calc_prev(
    csgraph: lil_matrix,
    nNodes: int,
    args: Tuple[np.ndarray, int],
) -> np.ndarray:
    '''
    For each origin zone and destination node,
    determine the previously visited node on the shortest path.
    '''
    indices, whichCPU = args
    nOrigSelection = len(indices)

    prev = np.zeros((nOrigSelection, nNodes), dtype=int)
    for i in range(nOrigSelection):
        prev[i, :] = dijkstra(
            csgraph, indices=indices[i], return_predecessors=True
        )[1]

        if whichCPU == 0:
            if i % int(round(nOrigSelection / 20)) == 0:
                print(f'{round(i / nOrigSelection * 100, 1)}%', end='\r')

    del csgraph

    return prev


@njit
def get_route(
    orig: int,
    dest: int,
    prev: np.ndarray,
    linkDict: np.ndarray,
    maxNumConnections: int = 8
):
    '''
    Deduce the paths from the prev object.
    Returns path sequence in terms of link IDs.
    '''
    route = []

    if orig != dest:

        # Deduce sequence of nodes on network
        sequenceNodes = []
        destNode = dest
        if prev[orig][destNode] >= 0:
            while prev[orig][destNode] >= 0:
                sequenceNodes.insert(0, destNode)
                destNode = prev[orig][destNode]
            else:
                sequenceNodes.insert(0, destNode)

        # Deduce sequence of links on network
        if len(sequenceNodes) > 1:

            for i in range(len(sequenceNodes) - 1):
                aNode = sequenceNodes[i]
                bNode = sequenceNodes[i + 1]

                tmp = linkDict[aNode]
                for col in range(maxNumConnections):
                    if tmp[col] == bNode:
                        route.append(tmp[col + maxNumConnections])
                        break

    return np.array(route, dtype=int32)


def get_link_dict(
    MRDHlinks: pd.DataFrame,
    maxNumConnections: int = 8,
) -> np.ndarray:
    """
    Returns an array that can be used to map two nodes to its link number.
    """
    linkDict = -1 * np.ones((max(MRDHlinks['A']) + 1, 2 * maxNumConnections), dtype=int)

    for i in MRDHlinks.index:
        aNode = MRDHlinks['A'][i]
        bNode = MRDHlinks['B'][i]

        for col in range(maxNumConnections):
            if linkDict[aNode][col] == -1:
                linkDict[aNode][col] = bNode
                linkDict[aNode][col + maxNumConnections] = i
                break

    return linkDict


def get_applicable_emission_fac(
    emissionsFactors: Dict[Tuple[int, int, int], Tuple[int, int]],
    vt: int, rt: int, et: int,
    capUt: float,
) -> float:
    '''
    Get the applicable emission factor given the
    vehicle type, emission type, road type and capacity utilization
    '''
    emissionFactorEmpty, emissionFactorLoaded = emissionsFactors[(vt, rt, et)]

    return (
        emissionFactorEmpty +
        capUt * (emissionFactorLoaded - emissionFactorEmpty)
    )


def add_zez_to_links(
    MRDHlinks: pd.DataFrame,
    MRDHlinksGeometry: List[Any],
    zones: pd.DataFrame,
    zonesGeometry: List[Any],
) -> pd.DataFrame:
    """
    Adds the field 'ZEZ' to the links DataFrame.
    """
    # Get links as shapely Point objects
    shapelyLinks = [
        [Point(x['coordinates'][0]), Point(x['coordinates'][-1])]
        for x in MRDHlinksGeometry
    ]

    # Get zones as shapely MultiPolygon/Polygon objects
    shapelyZones = []
    for x in zonesGeometry:
        if x['type'] == 'MultiPolygon':
            shapelyZones.append(MultiPolygon([
                Polygon(x['coordinates'][0][i])
                for i in range(len(x['coordinates'][0]))]))
        else:
            shapelyZones.append(Polygon(x['coordinates'][0]))

    shapelyZonesZEZ = np.array(shapelyZones, dtype=object)[np.where(zones['ZEZ'] >= 1)[0]]

    # Check if links are in ZEZ
    zezLinks = np.zeros((len(MRDHlinks)), dtype=int)
    linksToCheck = np.where(
        (MRDHlinks['Gemeentena'] != '') &
        (MRDHlinks['WEGTYPE'] != 'Autosnelweg'))[0]
    nLinksToCheck = len(linksToCheck)
    for i in range(nLinksToCheck):
        linkNo = linksToCheck[i]
        startPoint = shapelyLinks[linkNo][0]
        endPoint = shapelyLinks[linkNo][1]

        # Check if startpoint is in ZEZ
        for shapelyZone in shapelyZonesZEZ:
            if shapelyZone.contains(startPoint):
                zezLinks[linkNo] = 1
                break

        # If startpoint not in ZEZ, check if endpoint is in ZEZ
        if zezLinks[linkNo] == 0:
            for shapelyZone in shapelyZonesZEZ:
                if shapelyZone.contains(endPoint):
                    zezLinks[linkNo] = 1
                    break

        if i % int(nLinksToCheck / 20) == 0:
            print(f"{round(i / nLinksToCheck * 100, 1)}%", end='\r')

    logger.debug(f'\tFound {np.sum(zezLinks)} links located in ZEZ.')

    MRDHlinks['ZEZ'] = zezLinks

    return MRDHlinks


def write_emissions_into_tours(
    filename: str,
    tripsCO2: Dict[int, float]
) -> pd.DataFrame:
    """
    Adds the field 'CO2' in the specified tour file.
    """
    tours = pd.read_csv(filename, sep=',')
    tours['CO2'] = [tripsCO2[i] for i in tours.index]
    tours.to_csv(filename, sep=',', index=False)

    return tours


def write_emissions_into_shipments(
    filename: str,
    tours: pd.DataFrame,
    zoneDict: Dict[int, int],
    varDict: Dict[str, str],
) -> None:
    """
    Adds the field 'CO2' in the specified shipments file.
    """
    # Calculate emissions at the tour level instead of trip level
    tours['TOUR_ID'] = [
        str(tours.at[i, 'CARRIER_ID']) + '_' + str(tours.at[i, 'TOUR_ID'])
        for i in tours.index]
    toursCO2 = pd.pivot_table(
        tours,
        values=['CO2'],
        index=['TOUR_ID'],
        aggfunc=np.sum)
    tourIDDict = dict(np.transpose(np.vstack((
        toursCO2.index,
        np.arange(len(toursCO2))))))
    toursCO2 = np.array(toursCO2['CO2'])

    # Read the shipments
    shipments = pd.read_csv(filename, sep=',')

    invZoneDict = dict((v, k) for k, v in zoneDict.items())
    shipments['ORIG'] = [invZoneDict[x] for x in shipments['ORIG']]
    shipments['DEST'] = [invZoneDict[x] for x in shipments['DEST']]

    shipments = shipments.sort_values('TOUR_ID')
    shipments.index = np.arange(len(shipments))

    # For each tour, which shipments belong to it
    tourIDs = [tourIDDict[x] for x in shipments['TOUR_ID']]
    shipIDs = []
    currentShipIDs = [0]
    for i in range(1, len(shipments)):
        if tourIDs[i - 1] == tourIDs[i]:
            currentShipIDs.append(i)
        else:
            shipIDs.append(currentShipIDs.copy())
            currentShipIDs = [i]
    shipIDs.append(currentShipIDs.copy())

    # Network distance of each shipment
    _, skimDistance, nZones = get_skims(varDict)
    shipDist = skimDistance[(shipments['ORIG'] - 1) * nZones + (shipments['DEST'] - 1)]

    del _, skimDistance

    # Divide CO2 of each tour over its shipments based on distance
    shipCO2 = np.zeros(len(shipments))

    for tourID in np.unique(tourIDs):
        currentDists = shipDist[shipIDs[tourID]]
        currentCO2 = toursCO2[tourID]

        if np.sum(currentDists) == 0:
            shipCO2[shipIDs[tourID]] = 0
        else:
            shipCO2[shipIDs[tourID]] = currentDists / np.sum(currentDists) * currentCO2

    shipments['CO2'] = shipCO2

    # Export enriched shipments with CO2 field
    shipments = shipments.sort_values('SHIP_ID')
    shipments.index = np.arange(len(shipments))
    shipments['ORIG'] = [zoneDict[x] for x in shipments['ORIG']]
    shipments['DEST'] = [zoneDict[x] for x in shipments['DEST']]

    shipments.to_csv(filename, sep=',', index=False)


def write_select_link_analysis(
    selectedLinkTripsArray: np.ndarray,
    MRDHlinks: pd.DataFrame,
    varDict: Dict[str, str],
) -> None:
    """
    Writes the select link analyses to a CSV in the output folder.
    """
    selectedLinks = varDict['SELECTED_LINKS'].split(',')

    selectedLinkTripsDF = pd.DataFrame(
        selectedLinkTripsArray,
        columns=[f'N_{link}' for link in selectedLinks])

    selectedLinkTripsDF['LINKNR'] = MRDHlinks['LINKNR']
    selectedLinkTripsDF['A'] = MRDHlinks['A']
    selectedLinkTripsDF['B'] = MRDHlinks['B']

    colOrder = ['LINKNR', 'A', 'B'] + [f'N_{link}' for link in selectedLinks]

    selectedLinkTripsDF[colOrder].to_csv(
        varDict['OUTPUTFOLDER'] + 'SelectedLinks.csv', sep=',', index=False)


def write_links_to_shp(
    MRDHlinks: pd.DataFrame,
    MRDHlinksGeometry: List[Any],
    intensityFieldsGeojson: List[str],
    varDict: Dict[str, str],
    root: Any,
) -> None:
    """
    Writes the loaded links into a shapefile in the output folder.
    """
    # Sorteren kolommen
    MRDHlinks = MRDHlinks[
        ['LINKNR', 'A', 'B', 'LENGTH', 'WEGTYPE', 'ZEZ', 'Gemeentena'] +
        ['T_FREIGHT', 'T_VAN', 'COST_FREIGHT', 'COST_VAN'] +
        intensityFieldsGeojson]

    MRDHlinks[intensityFieldsGeojson] = np.round(MRDHlinks[intensityFieldsGeojson], 5)
    MRDHlinks['Gemeentena'] = [x.replace("'","") for x in MRDHlinks['Gemeentena'].astype(str).values]
    MRDHlinks.loc[pd.isna(MRDHlinks['ZEZ']), 'ZEZ'] = 0

    # Initialize shapefile fields
    w = shp.Writer(f"{varDict['OUTPUTFOLDER']}links_loaded_{varDict['LABEL']}.shp")
    w.field('LINKNR',      'N', size=8, decimal=0)
    w.field('A',           'N', size=9, decimal=0)
    w.field('B',           'N', size=9, decimal=0)
    w.field('LENGTH'       'N', size=7, decimal=3)
    w.field('WEGTYPE',     'C')
    w.field('ZEZ',         'N', size=1, decimal=0)
    w.field('Gemeentena',  'C')
    w.field('T_FREIGHT',   'N', size=8, decimal=5)
    w.field('T_VAN',       'N', size=8, decimal=5)
    w.field('COST_FREIGHT','N', size=8, decimal=5)
    w.field('COST_VAN',    'N', size=8, decimal=5)

    for field in intensityFieldsGeojson[:4]:
        w.field(field, 'N', size=9, decimal=5)
    for field in intensityFieldsGeojson[4:]:
        w.field(field, 'N', size=6, decimal=1)

    dbfData = np.array(MRDHlinks, dtype=object)
    nLinks = MRDHlinks.shape[0]

    for i in range(nLinks):
        # Add geometry
        geom = MRDHlinksGeometry[i]['coordinates']
        line = []
        for l in range(len(geom) - 1):
            line.append([
                [geom[l][0], geom[l][1]],
                [geom[l + 1][0], geom[l + 1][1]]])
        w.line(line)

        # Add data fields
        w.record(*dbfData[i, :])

        if i % int(round(nLinks / 100)) == 0:
            print(f'{round(i / nLinks * 100, 1)}%', end='\r')

            if root is not None:
                root.progressBar['value'] = (
                    95.0 +
                    (100.0 - 95.0) * i / nLinks)

    w.close()
