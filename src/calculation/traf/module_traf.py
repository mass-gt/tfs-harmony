import functools
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import traceback

from scipy.sparse import lil_matrix
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Any, Dict, List

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import get_num_cpu, read_mtx, read_shape
from .traf_support import (
    add_zez_to_links, calc_prev, get_route, get_link_dict, get_emission_factors,
    get_applicable_emission_fac,
    write_emissions_into_tours, write_emissions_into_shipments,
    write_select_link_analysis, write_links_to_shp)

logger = logging.getLogger("tfs")

LARGE_NUMBER = 10000


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

        doShiftVanToElectric = (varDict['SHIFT_VAN_TO_COMB1'] != '')

        exportShp = True
        addZezToLinks = False

        if varDict['N_MULTIROUTE'] == '':
            varDict['N_MULTIROUTE'] = 1
        else:
            varDict['N_MULTIROUTE'] = int(varDict['N_MULTIROUTE'])

        nLS = len(dims.logistic_segment)  # Number of logistic segments
        nET = len(dims.emission_type)  # Number of emission types
        nVT = len(dims.vehicle_type)  # Number of vehicle types

        dictET: Dict[int, str] = dict(
            (et, str(row['Comment'])) for et, row in dims.emission_type.items()
        )

        id_stad = dims.get_id_from_label("road_type", "Stad")
        id_buitenweg = dims.get_id_from_label("road_type", "Buitenweg")
        id_snelweg = dims.get_id_from_label("road_type", "Snelweg")

        # Which vehicle type can be used in the parcel module
        vehTypesParcels = [
            row['ID'] for row in dims.vehicle_type.values()
            if row['IsAvailableInParcelModule'] == 1]

        # Enumerate the different time periods (i.e. hours) of the day
        nHours = 24
        timeOfDays = np.arange(nHours)

        # Carrying capacity in kg
        truckCapacities = dict(
            (int(row['vehicle_type']), float(row['capacity_kg']))
            for row in pd.read_csv(varDict['VEHICLE_CAPACITY'], sep='\t').to_dict('records')
        )

        # For which LINKNR-values to perform selected link analyses
        doSelectedLink = (varDict['SELECTED_LINKS'] not in ["", "''", '""'])
        if doSelectedLink:
            selectedLinks = varDict['SELECTED_LINKS'].split(',')
            nSelectedLinks = len(selectedLinks)
            try:
                selectedLinks = [int(x) for x in selectedLinks]
            except ValueError:
                logger.warning(f"Could not convert SELECTED_LINKS to integers: {selectedLinks}")

        # Number of CPUs over which the route search procedure is parallelized
        nCPU = get_num_cpu(varDict, 8)

        if root is not None:
            root.progressBar['value'] = 0.2

        # -------------- Importing and preprocessing network ------------------

        logger.debug("\tImporting and preprocessing network...")

        # Import links
        MRDHlinks, MRDHlinksGeometry = read_shape(varDict['LINKS'], returnGeometry=True)
        nLinks = len(MRDHlinks)

        if root is not None:
            root.progressBar['value'] = 2.0

        # Import nodes and zones
        MRDHnodes: pd.DataFrame = read_shape(varDict['NODES'])
        zones, zonesGeometry = read_shape(varDict['ZONES'], returnGeometry=True)
        nNodes = len(MRDHnodes)

        if root is not None:
            root.progressBar['value'] = 2.5

        # Cost parameters freight
        costPerKmFreight = pd.read_csv(varDict['COST_SOURCING'], sep='\t').to_dict('records')[0]['cost_per_km']
        costPerHourFreight = pd.read_csv(varDict['COST_SOURCING'], sep='\t').to_dict('records')[0]['cost_per_hour']

        # Cost parameters vans
        id_van = dims.get_id_from_label("vehicle_type", "Van")
        costPerKmVan = dict(
            (int(row['vehicle_type']), float(row['cost_per_km']))
            for row in pd.read_csv(varDict['COST_VEHTYPE'], sep='\t').to_dict('records')
        )[id_van]
        costPerHourVan = dict(
            (int(row['vehicle_type']), float(row['cost_per_hour']))
            for row in pd.read_csv(varDict['COST_VEHTYPE'], sep='\t').to_dict('records')
        )[id_van]

        # Get zone numbering
        zones = zones.sort_values('AREANR')
        areaNumbers = np.array(zones['AREANR'], dtype=int)
        nIntZones = len(zones)
        nSupZones = len(pd.read_csv(varDict['SUP_COORDINATES_ID']))
        nZones = nIntZones + nSupZones

        ZEZzones = set(np.where(zones['ZEZ'] >= 1)[0])

        # If you want to do the spatial coupling of ZEZ zones to the links
        # here instead of in QGIS
        if addZezToLinks:
            logger.debug("\tPerforming spatial coupling of ZEZ-zones to links...")

            MRDHlinks = add_zez_to_links(MRDHlinks, MRDHlinksGeometry, zones, zonesGeometry)

        # Dictionary with zone numbers (keys) to corresponding
        # centroid node ID as given in the shapefile (values)
        zoneToCentroidKeys = np.arange(nIntZones)
        zoneToCentroidValues = []
        for i in range(nIntZones):
            zoneToCentroidValues.append(np.where(
                (MRDHnodes['AREANR'] == areaNumbers[i]) &
                (MRDHnodes['TYPENO'] == 99))[0][0])
        zoneToCentroid = dict(np.transpose(np.vstack((
            zoneToCentroidKeys,
            zoneToCentroidValues))))
        for i in range(nSupZones):
            zoneToCentroid[nIntZones + i] = np.where(
                (MRDHnodes['AREANR'] == 99999900 + i + 1) &
                (MRDHnodes['TYPENO'] == 99))[0][0]

        # Dictionary with zone number (used in this script) to
        # corresponding zone number (used in input/output)
        zoneDict = dict(np.transpose(np.vstack((
            np.arange(nIntZones),
            zones['AREANR']))))
        zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones + i] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        # The node IDs as given in the shapefile (values) and
        # a new ID numbering from 0 to nNodes (keys)
        nodeDict = dict(np.transpose(np.vstack((
            np.arange(nNodes),
            MRDHnodes['NODENR']))))
        invNodeDict = dict((v, k) for k, v in nodeDict.items())

        # Check on NODENR values
        nodeNr = set(np.array(MRDHnodes['NODENR']))
        missingNodes = list(np.unique([
            x for x in MRDHlinks['A'] if x not in nodeNr]))
        for x in MRDHlinks['B']:
            if x not in nodeNr:
                if x not in missingNodes:
                    missingNodes.append(x)
        if len(missingNodes) > 0:
            raise Exception(
                "Error! The following NODENR values were found in the links shape " +
                f" but not in the nodes shape! {missingNodes}")

        if root is not None:
            root.progressBar['value'] = 3.0

        # Recode the node IDs
        MRDHnodes['NODENR'] = np.arange(nNodes)
        MRDHlinks['A'] = [invNodeDict[x] for x in MRDHlinks['A']]
        MRDHlinks['B'] = [invNodeDict[x] for x in MRDHlinks['B']]

        # Recode the link IDs from 0 to len(MRDHlinks)
        MRDHlinks.index = np.arange(len(MRDHlinks))
        MRDHlinks.index = MRDHlinks.index.map(int)

        # Assume a speed of 50 km/h if there are links with freight speed <= 0
        nSpeedZero = np.sum(MRDHlinks[varDict['IMPEDANCE_SPEED_FREIGHT']] <= 0)
        if nSpeedZero > 0:

            MRDHlinks.loc[
                MRDHlinks[varDict['IMPEDANCE_SPEED_FREIGHT']] <= 0,
                varDict['IMPEDANCE_SPEED_FREIGHT']] = 50

            logger.warning(
                f"{nSpeedZero} links found with freight speed " +
                f"({varDict['IMPEDANCE_SPEED_FREIGHT']}) <= 0 km/h. " +
                "Adjusting those to 50 km/h.")

        # Assume a speed of 50 km/h if there are links with van speed <= 0
        nSpeedZero = np.sum(MRDHlinks[varDict['IMPEDANCE_SPEED_VAN']] <= 0)

        if nSpeedZero > 0:
            MRDHlinks.loc[
                MRDHlinks[varDict['IMPEDANCE_SPEED_VAN']] <= 0,
                varDict['IMPEDANCE_SPEED_VAN']] = 50

            logger.warning(
                f"{nSpeedZero} links found with van speed " +
                f"{varDict['IMPEDANCE_SPEED_VAN']}) <= 0 km/h. " +
                "Adjusting those to 50 km/h")

        # Travel times and travel costs
        MRDHlinks['T0_FREIGHT'] = (
            MRDHlinks['LENGTH'] /
            MRDHlinks[varDict['IMPEDANCE_SPEED_FREIGHT']])
        MRDHlinks['T0_VAN'] = (
            MRDHlinks['LENGTH'] /
            MRDHlinks[varDict['IMPEDANCE_SPEED_VAN']])
        MRDHlinks['COST_FREIGHT'] = (
            costPerKmFreight * MRDHlinks['LENGTH'] +
            costPerHourFreight * MRDHlinks['T0_FREIGHT'])
        MRDHlinks['COST_VAN'] = (
            costPerKmVan * MRDHlinks['LENGTH'] +
            costPerHourVan * MRDHlinks['T0_VAN'])

        costFreightHybrid = MRDHlinks['COST_FREIGHT'].copy()
        costVanHybrid = MRDHlinks['COST_VAN'].copy()

        # Set connector travel times high so these are not chosen
        # other than for entering/leaving network
        MRDHlinks.loc[
            MRDHlinks['WEGTYPE'] == 'voedingslink',
            ['COST_FREIGHT', 'COST_VAN']
        ] = LARGE_NUMBER

        # Set travel times for forbidden-for-freight-links high
        # so these are not chosen for freight
        MRDHlinks.loc[
            (MRDHlinks['WEGTYPE'] == 'Vrachtverbod') | (MRDHlinks['WEGTYPE'] == 'Vrachtstrook'),
            'COST_FREIGHT'
        ] = LARGE_NUMBER

        # Set travel times on links in ZEZ Rotterdam high so these
        # are only used to go to UCC and not for through traffic
        if varDict['LABEL'] == 'UCC':
            MRDHlinks.loc[MRDHlinks['ZEZ'] >= 1, 'COST_FREIGHT'] += LARGE_NUMBER
            MRDHlinks.loc[MRDHlinks['ZEZ'] >= 1, 'COST_VAN'] += LARGE_NUMBER

        # Dictionary with fromNodeID, toNodeID (keys) and link IDs (values)
        linkDict = get_link_dict(MRDHlinks)

        # Initialize empty fields with emissions and traffic intensity
        # per link (also save list with all field names)
        volCols = (
            [f'N_LS{ls}' for ls in dims.logistic_segment.keys()] +
            ['N_VAN_S', 'N_VAN_C'] +
            [f'N_VEH{vt}' for vt in dims.vehicle_type.keys()] +
            ['N_TOT']
        )

        intensityFields: List[str] = []
        intensityFieldsGeojson: List[str] = []

        for nameET in dictET.values():
            MRDHlinks[nameET] = 0.0
            intensityFields.append(nameET)
            intensityFieldsGeojson.append(nameET)

        for volCol in volCols:
            MRDHlinks[volCol] = 0
            intensityFields.append(volCol)
            intensityFieldsGeojson.append(volCol)

        # Intensity per time of day and emissions per logistic segment
        # are not in de links geojson but in a seperate CSV
        for vt in vehTypesParcels:
            intensityFields.append('N_LS8_VEH' + str(vt))
        for tod in range(nHours):
            intensityFields.append('N_TOD' + str(tod))
        for ls in range(nLS):
            for nameET in dictET.values():
                intensityFields.append(nameET + '_LS' + str(ls))
        for ls in ['VAN_S', 'VAN_C']:
            for nameET in dictET.values():
                intensityFields.append(nameET + '_' + str(ls))

        MRDHlinksIntensities = pd.DataFrame(np.zeros(
            (len(MRDHlinks), len(intensityFields))),
            columns=intensityFields)

        # Van trips for service and construction purposes
        vanTripsService = read_mtx(varDict['OUTPUTFOLDER'] + 'TripsVanService.mtx')
        vanTripsConstruction = read_mtx(varDict['OUTPUTFOLDER'] + 'TripsVanConstruction.mtx')

        # ODs with very low number of trips: set to 0 to reduce
        # memory burden of searches routes for all these ODs
        # (need to implement a smart bucket rounding or sparsification algorithm for this sometime)
        vanTripsService[np.where(vanTripsService < 0.1)[0]] = 0
        vanTripsConstruction[np.where(vanTripsConstruction < 0.1)[0]] = 0

        # Reshape to square array
        vanTripsService = vanTripsService.reshape(nZones, nZones)
        vanTripsConstruction = vanTripsConstruction.reshape(nZones, nZones)

        # Make some space available on the RAM
        del zones, zonesGeometry, MRDHnodes

        if root is not None:
            root.progressBar['value'] = 4.0

        # ------------ Information for the emission calculations --------------

        emissionFactors = get_emission_factors(varDict, dims)

        # Import trips csv
        allTrips: pd.DataFrame = pd.read_csv(
            varDict['OUTPUTFOLDER'] + "Tours_" + varDict['LABEL'] + ".csv")
        allTrips['ORIG'] = [invZoneDict[x] for x in allTrips['ORIG']]
        allTrips['DEST'] = [invZoneDict[x] for x in allTrips['DEST']]
        allTrips.loc[allTrips['TRIP_DEPTIME'] >= nHours, 'TRIP_DEPTIME'] -= nHours
        allTrips.loc[allTrips['TRIP_DEPTIME'] >= nHours, 'TRIP_DEPTIME'] -= nHours
        allTrips['CAP_UT'] = (
            (allTrips['TRIP_WEIGHT'] * 1000) /
            [truckCapacities[vt] for vt in allTrips['VEHTYPE'].astype(int).values])
        allTrips['INDEX'] = allTrips.index

        # Import parcel schedule csv
        allParcelTrips = pd.read_csv(
            varDict['OUTPUTFOLDER'] + "ParcelSchedule_" + varDict['LABEL'] + ".csv")
        allParcelTrips = allParcelTrips.rename(
            columns={
                'O_zone': 'ORIG',
                'D_zone': 'DEST',
                'TripDepTime': 'TRIP_DEPTIME'})
        allParcelTrips['ORIG'] = [invZoneDict[x] for x in allParcelTrips['ORIG']]
        allParcelTrips['DEST'] = [invZoneDict[x] for x in allParcelTrips['DEST']]
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME'] >= nHours,'TRIP_DEPTIME'] -= nHours
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME'] >= nHours,'TRIP_DEPTIME'] -= nHours
        allParcelTrips['CAP_UT'] = 0.5
        allParcelTrips['VEHTYPE'] = [
            dims.get_id_from_label("vehicle_type", vt) for vt in allParcelTrips['VehType']]
        allParcelTrips['LS' ] = dims.get_id_from_label("logistic_segment", "Parcel (deliveries)")
        allParcelTrips['INDEX'] = allParcelTrips.index

        # Trips coming from UCC to ZEZ use electric
        allParcelTrips.loc[allParcelTrips['OrigType'] == 'UCC', 'COMBTYPE'] = 1

        # For each tourID, at which indices are its trips found
        whereParcelTour = {}
        for i in allParcelTrips.index:
            tourID = allParcelTrips.at[i, 'Tour_ID']
            try:
                whereParcelTour[tourID].append(i)
            except KeyError:
                whereParcelTour[tourID] = [i]

        # Fuel as basic combustion type for parcel tours
        allParcelTrips['COMBTYPE'] = 0

        # In UCC scenario
        # Switch to electric for deliveries from UCC to ZEZ
        # Switch to hybrid for deliveries from depot to ZEZ
        origType = np.array(allParcelTrips['OrigType'])
        destParcel = np.array(allParcelTrips['DEST'])
        if varDict['LABEL'] == 'UCC':
            for i in allParcelTrips.index:
                if origType[i] == 'UCC':
                    allParcelTrips.at[i, 'COMBTYPE'] = 1
                elif destParcel[i] in ZEZzones:
                    allParcelTrips.at[i, 'COMBTYPE'] = 3

        # Possibility to switch to electric
        if doShiftVanToElectric:
            for indices in list(whereParcelTour.values()):
                if varDict['SHIFT_VAN_TO_COMB1'] <= np.random.rand():
                    allParcelTrips.loc[indices, 'COMBTYPE'] = 1

        # Determine linktypes (urban/rural/highway)
        stadLinkTypes = [
            'ETW_bibeko_30',
            'GOW_bibeko_50',
            'GOW_bibeko_70',
            'WOW_bibeko_50',
            'Verblijfsgebied_15']
        buitenwegLinkTypes = [
            'ETW_bubeko_breed_60',
            'ETW_bubeko_smal_60',
            'GOW_bubeko_gemengd_80',
            'GOW_bubeko_gesloten_80',
            'Industrieontsluitingsweg_50',
            'Industriestraat_30']
        snelwegLinkTypes = [
            'Autosnelweg',
            'Autoweg',
            'Vrachtverbod']

        whereStad = [
            MRDHlinks['WEGTYPE'][i] in stadLinkTypes
            for i in MRDHlinks.index]
        whereBuitenweg = [
            MRDHlinks['WEGTYPE'][i] in buitenwegLinkTypes
            for i in MRDHlinks.index]
        whereSnelweg = [
            MRDHlinks['WEGTYPE'][i] in snelwegLinkTypes
            for i in MRDHlinks.index]

        roadtypeArray = np.zeros((len(MRDHlinks)))
        roadtypeArray[whereStad] = 1
        roadtypeArray[whereBuitenweg] = 2
        roadtypeArray[whereSnelweg] = 3
        distArray = np.array(MRDHlinks['LENGTH'])
        ZEZarray = np.array(MRDHlinks['ZEZ'] >= 1, dtype=int)

        # Bring ORIG and DEST to the front of the list of column names
        newColOrder = volCols.copy()
        newColOrder.insert(0, 'DEST')
        newColOrder.insert(0, 'ORIG')

        # Trip matrices per time-of-day
        tripMatricesTOD = []
        for tod in range(nHours):
            tripMatricesTOD.append(pd.read_csv(
                f"{varDict['OUTPUTFOLDER']}tripmatrix_{varDict['LABEL']}_TOD{tod}.txt",
                sep='\t'))
            tripMatricesTOD[tod]['ORIG'] = [
                invZoneDict[x] for x in tripMatricesTOD[tod]['ORIG'].values]
            tripMatricesTOD[tod]['DEST'] = [
                invZoneDict[x] for x in tripMatricesTOD[tod]['DEST'].values]
            tripMatricesTOD[tod]['N_LS8'] = 0
            tripMatricesTOD[tod]['N_VAN_S'] = 0
            tripMatricesTOD[tod]['N_VAN_C'] = 0
            tripMatricesTOD[tod] = tripMatricesTOD[tod][newColOrder]
            tripMatricesTOD[tod] = np.array(tripMatricesTOD[tod])

        # Parcels trip matrices per time-of-day
        tripMatricesParcelsTOD = []
        for tod in range(nHours):
            tripMatricesParcelsTOD.append(pd.read_csv(
                varDict['OUTPUTFOLDER'] + "tripmatrix_parcels_" + str(varDict['LABEL']) + "_TOD" + str(tod) + ".txt",
                sep='\t'))
            tripMatricesParcelsTOD[tod]['ORIG'] = [
                invZoneDict[x] for x in tripMatricesParcelsTOD[tod]['ORIG'].values]
            tripMatricesParcelsTOD[tod]['DEST'] = [
                invZoneDict[x] for x in tripMatricesParcelsTOD[tod]['DEST'].values]
            tripMatricesParcelsTOD[tod] = np.array(tripMatricesParcelsTOD[tod])

        # For which origin zones do we need to find the routes
        origSelection = np.arange(nZones)
        nOrigSelection = len(origSelection)

        # Initialize arrays for intensities and emissions
        linkTripsArray = [
            np.zeros((len(MRDHlinks), len(volCols))) for tod in range(nHours)]
        linkVanTripsArray = np.zeros((len(MRDHlinks), 2))
        linkEmissionsArray = [
            [np.zeros((len(MRDHlinks), nET)) for tod in range(nHours)]
            for ls in range(nLS)]
        linkVanEmissionsArray = [
            np.zeros((len(MRDHlinks), nET)) for ls in ['VAN_S', 'VAN_C']]

        if doSelectedLink:
            selectedLinkTripsArray = np.zeros((len(MRDHlinks), nSelectedLinks))

        if root is not None:
            root.progressBar['value'] = 5.0

        # ------------------ Route search (freight) ---------------------------

        logger.debug("\tSearching routes for freight...")

        tripsCO2 = {}
        parcelTripsCO2 = {}

        # Whether a separate route search needs to be done for hybrid vehicles
        # or not
        doHybridRoutes = (
            np.any(allTrips.loc[:, 'COMBTYPE'] == 3) or
            np.any(allParcelTrips.loc[:, 'COMBTYPE'] == 3) or
            varDict['LABEL'] == 'UCC')

        # From which nodes do we need to perform the shortest path algoritm
        indices = np.array([zoneToCentroid[x] for x in origSelection], dtype=int)

        # List of matrices with for each node the previous node
        # on the shortest path
        prevFreight = []

        # Route search freight
        if nCPU > 1:
            # From which nodes does every CPU perform
            # the shortest path algorithm
            indicesPerCPU = [
                [indices[cpu::nCPU], cpu]
                for cpu in range(nCPU)]
            origSelectionPerCPU = [
                np.arange(nOrigSelection)[cpu::nCPU]
                for cpu in range(nCPU)]

            for r in range(varDict['N_MULTIROUTE']):

                logger.debug(f"\t\tRoute search (freight - multirouting part {r+1})...")

                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                csgraphFreight[
                    np.array(MRDHlinks['A']),
                    np.array(MRDHlinks['B'])] = np.array(MRDHlinks['COST_FREIGHT'])

                if varDict['N_MULTIROUTE'] > 1:
                    csgraphFreight[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                # Execute the Dijkstra route search
                with mp.Pool(nCPU) as p:
                    prevFreightPerCPU = p.map(functools.partial(
                        calc_prev,
                        csgraphFreight,
                        nNodes), indicesPerCPU)

                # Combine the results from the different CPUs
                prevFreight.append(np.zeros((nOrigSelection ,nNodes), dtype=int))
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevFreight[r][origSelectionPerCPU[cpu][i], :] = prevFreightPerCPU[cpu][i, :]

                # Make some space available on the RAM
                del prevFreightPerCPU

                if root is not None:
                    root.progressBar['value'] = (
                        5.0 +
                        (33.0 - 5.0) * (1 + r) / varDict['N_MULTIROUTE'])

        else:
            for r in range(varDict['N_MULTIROUTE']):
                logger.debug("\t\tRoute search (freight - multirouting part {r+1})...")

                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                csgraphFreight[
                    np.array(MRDHlinks['A']),
                    np.array(MRDHlinks['B'])] = np.array(MRDHlinks['COST_FREIGHT'])

                if varDict['N_MULTIROUTE'] > 1:
                    csgraphFreight[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                # Execute the Dijkstra route search
                prevFreight.append(calc_prev(csgraphFreight, nNodes, [indices, 0]))

                if root is not None:
                    root.progressBar['value'] = (
                        5.0 +
                        (33.0 - 5.0) * (1 + r) / varDict['N_MULTIROUTE'])

        # Make some space available on the RAM
        del csgraphFreight

        if doHybridRoutes:
            prevFreightHybrid = []

            # Route search freight (hybrid combustion)
            if nCPU > 1:

                for r in range(varDict['N_MULTIROUTE']):

                    logger.debug(
                        f"\t\tRoute search (freight - hybrid combustion - multirouting part {r+1})...")

                    # The network with costs between nodes (freight)
                    csgraphFreightHybrid = lil_matrix((nNodes, nNodes))
                    csgraphFreightHybrid[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] = costFreightHybrid

                    if varDict['N_MULTIROUTE'] > 1:
                        csgraphFreightHybrid[
                            np.array(MRDHlinks['A']),
                            np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                    # Execute the Dijkstra route search
                    with mp.Pool(nCPU) as p:
                        prevFreightHybridPerCPU = p.map(functools.partial(
                            calc_prev,
                            csgraphFreightHybrid,
                            nNodes), indicesPerCPU)

                    # Combine the results from the different CPUs
                    prevFreightHybrid.append(np.zeros((nOrigSelection ,nNodes), dtype=int))
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevFreightHybrid[r][origSelectionPerCPU[cpu][i], :] = prevFreightHybridPerCPU[cpu][i, :]

                    # Make some space available on the RAM
                    del prevFreightHybridPerCPU

                    if root is not None:
                        root.progressBar['value'] = (
                            5.0 +
                            (33.0 - 5.0) * (1 + r) / varDict['N_MULTIROUTE'])

            else:
                for r in range(varDict['N_MULTIROUTE']):
                    logger.debug(
                        f"\tRoute search (freight - hybrid combustion - multirouting part {r+1})...")

                    # The network with costs between nodes (freight)
                    csgraphFreightHybrid = lil_matrix((nNodes, nNodes))
                    csgraphFreightHybrid[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] = costFreightHybrid

                    if varDict['N_MULTIROUTE'] > 1:
                        csgraphFreightHybrid[
                            np.array(MRDHlinks['A']),
                            np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                    # Execute the Dijkstra route search
                    prevFreightHybrid.append(calc_prev(
                        csgraphFreightHybrid,
                        nNodes,
                        [indices, 0]))

                    if root is not None:
                        root.progressBar['value'] = (
                            5.0 +
                            (33.0 - 5.0) * (1 + r) / varDict['N_MULTIROUTE'])

            # Make some space available on the RAM
            del csgraphFreightHybrid

        # --------------- Emissions and intensities (freight) -----------------

        logger.debug("\tCalculating emissions and traffic intensities (freight)...")

        id_co2 = dims.get_id_from_label("emission_type", "CO2")

        for tod in timeOfDays:

            print(f'Hour {tod + 1} of {nHours}...', end='\r')

            # Nu de tod-specifieke tripmatrix in tripMatrix variabele zetten
            tripMatrix = tripMatricesTOD[tod]
            tripMatrixOrigins = set(tripMatrix[:, 0])

            # Selecteer de trips die vertrokken in de huidige time-of-day en
            # bereken de capacity utilization
            trips = allTrips.loc[
                (allTrips['TRIP_DEPTIME'] >= tod) & (allTrips['TRIP_DEPTIME'] < (tod + 1)),
                ['CARRIER_ID', 'ORIG', 'DEST', 'VEHTYPE', 'CAP_UT', 'LS', 'COMBTYPE', 'INDEX']
            ].values

            # At which indices are trips per orig-dest-ls found
            whereODL = {}
            for i in range(len(tripMatrix)):
                orig = tripMatrix[i, 0]
                dest = tripMatrix[i, 1]
                for ls in range(nLS):
                    whereODL[(orig, dest, ls)] = []
            for i in range(len(trips)):
                orig = trips[i, 1]
                dest = trips[i, 2]
                ls = trips[i, 5]
                whereODL[(orig, dest, ls)].append(i)

            # Assign to routes and calculate emissions
            for i in range(nOrigSelection):
                origZone = origSelection[i]

                if origZone in tripMatrixOrigins:
                    destZoneIndex = np.where(tripMatrix[:, 0] == origZone)[0]
                    destZones = tripMatrix[destZoneIndex, 1]

                    routes = [
                        [get_route(i, zoneToCentroid[j], prevFreight[r], linkDict) for j in destZones]
                        for r in range(varDict['N_MULTIROUTE'])]
                            
                    if doHybridRoutes:
                        hybridRoutes = [
                            [get_route(i, zoneToCentroid[j], prevFreightHybrid[r], linkDict) for j in destZones]
                            for r in range(varDict['N_MULTIROUTE'])]

                    # Schrijf de volumes op de links
                    for j in range(len(destZones)):
                        destZone = destZones[j]
                        nTrips = tripMatrix[destZoneIndex[j], 2:]

                        # Get route and part of route that is
                        # stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        routesStad = []
                        routesBuitenweg = []
                        routesSnelweg = []
                        ZEZSstad = []
                        ZEZSbuitenweg = []
                        ZEZSsnelweg = []

                        for r in range(varDict['N_MULTIROUTE']):
                            routesStad.append(routes[r][j][roadtypeArray[routes[r][j]] == 1])
                            routesBuitenweg.append(routes[r][j][roadtypeArray[routes[r][j]] == 2])
                            routesSnelweg.append(routes[r][j][roadtypeArray[routes[r][j]] == 3])
                            ZEZSstad.append(ZEZarray[routesStad[r]] == 1)
                            ZEZSbuitenweg.append(ZEZarray[routesBuitenweg[r]] == 1)
                            ZEZSsnelweg.append(ZEZarray[routesSnelweg[r]] == 1)

                        if doHybridRoutes:
                            hybridRoutesStad = []
                            hybridRoutesBuitenweg = []
                            hybridRoutesSnelweg = []
                            hybridZEZSstad = []
                            hybridZEZSbuitenweg = []
                            hybridZEZSsnelweg = []
    
                            for r in range(varDict['N_MULTIROUTE']):
                                hybridRoute = hybridRoutes[r][j]
                                hybridRoutesStad.append(
                                    hybridRoute[roadtypeArray[hybridRoute] == 1])
                                hybridRoutesBuitenweg.append(
                                    hybridRoute[roadtypeArray[hybridRoute] == 2])
                                hybridRoutesSnelweg.append(
                                    hybridRoute[roadtypeArray[hybridRoute] == 3])
                                hybridZEZSstad.append(
                                    ZEZarray[hybridRoutesStad[r]] == 1)
                                hybridZEZSbuitenweg.append(
                                    ZEZarray[hybridRoutesBuitenweg[r]] == 1)
                                hybridZEZSsnelweg.append(
                                    ZEZarray[hybridRoutesSnelweg[r]] == 1)

                        # Bereken en schrijf de intensiteiten/emissies op de links
                        for ls in range(nLS):

                            # Welke trips worden allemaal gemaakt op de HB
                            # van de huidige iteratie van de ij-loop
                            currentTrips = trips[whereODL[(origZone, destZone, ls)], :]
                            nCurrentTrips = len(currentTrips)

                            for trip in range(nCurrentTrips):
                                vt = int(currentTrips[trip, 3])
                                ct = int(currentTrips[trip, 6])
                                capUt = currentTrips[trip, 4]

                                # Select which of the calculated routes
                                # to use for current trip
                                whichMultiRoute = np.random.randint(varDict['N_MULTIROUTE'])

                                route = routes[whichMultiRoute][j]
                                routeStad = routesStad[
                                    whichMultiRoute]
                                routeBuitenweg = routesBuitenweg[
                                    whichMultiRoute]
                                routeSnelweg = routesSnelweg[
                                    whichMultiRoute]

                                ZEZstad = ZEZSstad[whichMultiRoute]
                                ZEZbuitenweg = ZEZSbuitenweg[whichMultiRoute]
                                ZEZsnelweg = ZEZSsnelweg[whichMultiRoute]

                                if doHybridRoutes:
                                    hybridRoute = hybridRoutes[whichMultiRoute][j]
                                    hybridRouteStad = hybridRoutesStad[whichMultiRoute]
                                    hybridRouteBuitenweg = hybridRoutesBuitenweg[whichMultiRoute]
                                    hybridRouteSnelweg = hybridRoutesSnelweg[whichMultiRoute]
    
                                    hybridZEZstad = hybridZEZSstad[whichMultiRoute]
                                    hybridZEZbuitenweg = hybridZEZSbuitenweg[whichMultiRoute]
                                    hybridZEZsnelweg = hybridZEZSsnelweg[whichMultiRoute]

                                # Keep track of links being used on the route
                                linkTripsArray[tod][route, ls] += 1
                                linkTripsArray[tod][route, nLS + 2 + vt] += 1
                                linkTripsArray[tod][route, nLS + 2 + nVT] += 1
                                if doSelectedLink:
                                    for link in range(nSelectedLinks):
                                        if selectedLinks[link] in route:
                                            selectedLinkTripsArray[
                                                route, link] += 1

                                # If combustion type is fuel or bio-fuel
                                if ct in [0, 4]:
                                    for et in range(nET):
                                        stadEmissions = (
                                            distArray[routeStad] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_stad, et, capUt))
    
                                        buitenwegEmissions  = (
                                            distArray[routeBuitenweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_buitenweg, et, capUt))
    
                                        snelwegEmissions  = (
                                            distArray[routeSnelweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_snelweg, et, capUt))
    
                                        linkEmissionsArray[ls][tod][routeStad, et] += stadEmissions
                                        linkEmissionsArray[ls][tod][routeBuitenweg, et] += buitenwegEmissions 
                                        linkEmissionsArray[ls][tod][routeSnelweg, et] += snelwegEmissions 

                                        # Total CO2 of the trip
                                        if et == id_co2:
                                            tripsCO2[currentTrips[trip, -1]] = (
                                                np.sum(stadEmissions) +
                                                np.sum(buitenwegEmissions) +
                                                np.sum(snelwegEmissions))

                                # If combustion type is hybrid
                                elif ct == 3:
                                    for et in range(nET):
                                        stadEmissions = (
                                            distArray[hybridRouteStad] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_stad, et, capUt))
    
                                        buitenwegEmissions  = (
                                            distArray[hybridRouteBuitenweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_buitenweg, et, capUt))
    
                                        snelwegEmissions  = (
                                            distArray[hybridRouteSnelweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_snelweg, et, capUt))

                                        stadEmissions[hybridZEZstad] = 0.0
                                        buitenwegEmissions[hybridZEZbuitenweg] = 0.0
                                        snelwegEmissions[hybridZEZsnelweg] = 0.0

                                        linkEmissionsArray[ls][tod][
                                            hybridRouteStad, et] += stadEmissions
                                        linkEmissionsArray[ls][tod][
                                            hybridRouteBuitenweg, et] += buitenwegEmissions 
                                        linkEmissionsArray[ls][tod][
                                            hybridRouteSnelweg, et] += snelwegEmissions 

                                        # Total CO2 of the trip
                                        if et == id_co2:
                                            tripsCO2[currentTrips[trip, -1]] = (
                                                np.sum(stadEmissions) +
                                                np.sum(buitenwegEmissions) +
                                                np.sum(snelwegEmissions))

                                else:
                                    tripsCO2[currentTrips[trip, -1]] = 0.0

            if root is not None:
                root.progressBar['value'] = (
                    33.0 +
                    (43.0 - 33.0) * (tod + 1) / nHours)

        del prevFreight
        
        if doHybridRoutes:
            del prevFreightHybrid

        # -------------------- Route search (vans) ----------------------------

        logger.debug("\tSearching routes for vans...")

        # List of matrices with for each node the previous node on the shortest path
        prevVan = []

        # Route search vans
        if nCPU > 1:
            for r in range(varDict['N_MULTIROUTE']):

                logger.debug(f"\t\tRoute search (vans - multirouting part {r+1})...")

                # The network with costs between nodes (vans)
                csgraphVan = lil_matrix((nNodes, nNodes))
                csgraphVan[
                    np.array(MRDHlinks['A']),
                    np.array(MRDHlinks['B'])] = np.array(MRDHlinks['COST_VAN'])

                if varDict['N_MULTIROUTE'] > 1:
                    csgraphVan[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                # Execute the Dijkstra route search
                with mp.Pool(nCPU) as p:
                    prevVanPerCPU = p.map(functools.partial(
                        calc_prev,
                        csgraphVan,
                        nNodes), indicesPerCPU)

                # Combine the results from the different CPUs
                prevVan.append(np.zeros((nOrigSelection, nNodes), dtype=int))
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevVan[r][origSelectionPerCPU[cpu][i], :] = prevVanPerCPU[cpu][i, :]

                # Make some space available on the RAM
                del prevVanPerCPU

                if root is not None:
                    root.progressBar['value'] = (
                        33.0 +
                        (60.0 - 33.0) * (1 + r) / varDict['N_MULTIROUTE'])

        else:
            for r in range(varDict['N_MULTIROUTE']):

                logger.debug(f"\t\tRoute search (vans - multirouting part {r+1})...")

                # The network with costs between nodes (vans)
                csgraphVan = lil_matrix((nNodes, nNodes))
                csgraphVan[
                    np.array(MRDHlinks['A']),
                    np.array(MRDHlinks['B'])] = np.array(MRDHlinks['COST_VAN'])

                if varDict['N_MULTIROUTE'] > 1:
                    csgraphVan[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                # Execute the Dijkstra route search
                prevVan.append(calc_prev(csgraphVan, nNodes, [indices, 0]))

                if root is not None:
                    root.progressBar['value'] = (
                        43.0 +
                        (70.0 - 43.0) * (1 + r) / varDict['N_MULTIROUTE'])

        # Make some space available on the RAM
        del csgraphVan

        if doHybridRoutes:
            prevVanHybrid = []

            # Route search vans
            if nCPU > 1:
                for r in range(varDict['N_MULTIROUTE']):

                    logger.debug(
                        f"\t\tRoute search (vans - hybrid combustion - multirouting part {r+1})...")

                    # The network with costs between nodes (vans)
                    csgraphVanHybrid = lil_matrix((nNodes, nNodes))
                    csgraphVanHybrid[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] = costVanHybrid
    
                    if varDict['N_MULTIROUTE'] > 1:
                        csgraphVanHybrid[
                            np.array(MRDHlinks['A']),
                            np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                    # Execute the Dijkstra route search
                    with mp.Pool(nCPU) as p:
                        prevVanHybridPerCPU = p.map(functools.partial(
                            calc_prev,
                            csgraphVanHybrid,
                            nNodes), indicesPerCPU)

                    # Combine the results from the different CPUs
                    prevVanHybrid.append(np.zeros((nOrigSelection, nNodes), dtype=int))
                    for cpu in range(nCPU):
                        for i in range(len(indicesPerCPU[cpu][0])):
                            prevVanHybrid[r][origSelectionPerCPU[cpu][i], :] = prevVanHybridPerCPU[cpu][i, :]

                    # Make some space available on the RAM
                    del prevVanHybridPerCPU

                    if root is not None:
                        root.progressBar['value'] = (
                            33.0 +
                            (60.0 - 33.0) * (1 + r) / varDict['N_MULTIROUTE'])

            else:
                for r in range(varDict['N_MULTIROUTE']):

                    logger.debug(
                        f"\t\tRoute search (vans - hybrid combustion - multirouting part {r+1})...")

                    # The network with costs between nodes (vans)
                    csgraphVanHybrid = lil_matrix((nNodes, nNodes))
                    csgraphVanHybrid[
                        np.array(MRDHlinks['A']),
                        np.array(MRDHlinks['B'])] = costVanHybrid

                    if varDict['N_MULTIROUTE'] > 1:
                        csgraphVanHybrid[
                            np.array(MRDHlinks['A']),
                            np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))

                    # Execute the Dijkstra route search
                    prevVanHybrid.append(calc_prev(csgraphVanHybrid, nNodes, [indices, 0]))

                    if root is not None:
                        root.progressBar['value'] = (
                            43.0 +
                            (70.0 - 43.0) * (1 + r) / varDict['N_MULTIROUTE'])

            # Make some space available on the RAM
            del csgraphVanHybrid

        # --------------- Emissions and intensities (parcel vans) -------------

        logger.debug("\tCalculating emissions and traffic intensities (vans)...")

        logger.debug("\t\tParcels tours...")

        ls = dims.get_id_from_label("logistic_segment", "Parcel (deliveries)")

        for tod in timeOfDays:

            print(f'Hour {tod + 1} of {nHours}...', end='\r')

            # Nu de tod-specifieke tripmatrix in tripMatrix variabele zetten
            tripMatrixParcels = tripMatricesParcelsTOD[tod]
            tripMatrixParcelsOrigins = set(tripMatrixParcels[:, 0])

            # Selecteer de trips die vertrokken in de huidige time-of-day
            # en bereken de capacity utilization
            trips = allParcelTrips.loc[
                (allParcelTrips['TRIP_DEPTIME'] >= tod) &
                (allParcelTrips['TRIP_DEPTIME'] < (tod + 1)), :]

            if len(trips) > 0:
                trips = np.array(trips[[
                    'Depot_ID',
                    'ORIG', 'DEST',
                    'VEHTYPE', 'CAP_UT',
                    'LS', 'COMBTYPE', 'INDEX']])

                for i in range(nOrigSelection):
                    origZone = origSelection[i]

                    if origZone in tripMatrixParcelsOrigins:
                        destZoneIndex = np.where(
                            tripMatrixParcels[:, 0] == origZone)[0]

                        # Schrijf de volumes op de links
                        for j in destZoneIndex:
                            destZone = tripMatrixParcels[j, 1]

                            route = get_route(
                                i, zoneToCentroid[destZone],
                                prevVan[0],
                                linkDict)

                            # Selecteer het deel van de route met
                            # linktype stad/buitenweg/snelweg
                            routeStad = route[roadtypeArray[route] == 1]
                            routeBuitenweg = route[roadtypeArray[route] == 2]
                            routeSnelweg = route[roadtypeArray[route] == 3]
                            ZEZstad = ZEZarray[routeStad] == 1
                            ZEZbuitenweg = ZEZarray[routeBuitenweg] == 1
                            ZEZsnelweg = ZEZarray[routeSnelweg] == 1

                            if doHybridRoutes:
                                hybridRoute = get_route(
                                    i, zoneToCentroid[destZone],
                                    prevVanHybrid[0],
                                    linkDict)

                                hybridRouteStad = hybridRoute[
                                    roadtypeArray[hybridRoute] == 1]
                                hybridRouteBuitenweg = hybridRoute[
                                    roadtypeArray[hybridRoute] == 2]
                                hybridRouteSnelweg = hybridRoute[
                                    roadtypeArray[hybridRoute] == 3]
                                hybridZEZstad = ZEZarray[
                                    hybridRouteStad] == 1
                                hybridZEZbuitenweg = ZEZarray[
                                    hybridRouteBuitenweg] == 1
                                hybridZEZsnelweg = ZEZarray[
                                    hybridRouteSnelweg] == 1

                            # Welke trips worden allemaal gemaakt op de
                            # HB van de huidige iteratie van de ij-loop
                            currentTrips = trips[
                                (trips[:, 1] == origZone) &
                                (trips[:, 2] == destZone), :]

                            # Bereken en schrijf de emissies op de links
                            for trip in range(len(currentTrips)):
                                vt = int(currentTrips[trip, 3])
                                ct = int(currentTrips[trip, 6])
                                capUt = currentTrips[trip, 4]
                                
                                if ct == 3:
                                    tmpRoute = hybridRoute
                                else:
                                    tmpRoute = route

                                # Number of trips for LS8 (= parcel deliveries)
                                linkTripsArray[tod][tmpRoute, ls] += 1

                                # Number of trips for vehicle type
                                linkTripsArray[tod][tmpRoute, nLS + 2 + vt] += 1

                                # Total number of trips
                                linkTripsArray[tod][tmpRoute, -1] += 1

                                # De parcel demand trips per voertuigtype
                                MRDHlinksIntensities.loc[
                                    tmpRoute,
                                    'N_LS8_VEH' + str(vt)] += 1

                                if doSelectedLink:
                                    for link in range(nSelectedLinks):
                                        if selectedLinks[link] in tmpRoute:
                                            selectedLinkTripsArray[tmpRoute, link] += 1

                                # If combustion type is fuel or bio-fuel
                                if ct in [0, 4]:
                                    for et in range(nET):
                                        stadEmissions = (
                                            distArray[routeStad] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_stad, et, capUt))

                                        buitenwegEmissions = (
                                            distArray[routeBuitenweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_buitenweg, et, capUt))

                                        snelwegEmissions = (
                                            distArray[routeSnelweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_snelweg, et, capUt))

                                        linkEmissionsArray[ls][tod][routeStad, et] = stadEmissions
                                        linkEmissionsArray[ls][tod][routeBuitenweg, et] += buitenwegEmissions
                                        linkEmissionsArray[ls][tod][routeSnelweg, et] += snelwegEmissions

                                        # CO2-emissions for the current trip
                                        if et == id_co2:
                                            parcelTripsCO2[currentTrips[trip, -1]] = (
                                                np.sum(stadEmissions) +
                                                np.sum(buitenwegEmissions) +
                                                np.sum(snelwegEmissions))

                                # Hybrid combustion
                                elif ct == 3:
                                    for et in range(nET):
                                        stadEmissions = (
                                            distArray[hybridRouteStad] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_stad, et, capUt))

                                        buitenwegEmissions = (
                                            distArray[hybridRouteBuitenweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_buitenweg, et, capUt))

                                        snelwegEmissions = (
                                            distArray[hybridRouteSnelweg] *
                                            get_applicable_emission_fac(
                                                emissionFactors, vt, id_snelweg, et, capUt))

                                        stadEmissions[hybridZEZstad] = 0.0
                                        buitenwegEmissions[hybridZEZbuitenweg] = 0.0
                                        snelwegEmissions[hybridZEZsnelweg] = 0.0

                                        linkEmissionsArray[ls][tod][
                                            hybridRouteStad, et] = stadEmissions
                                        linkEmissionsArray[ls][tod][
                                            hybridRouteBuitenweg, et] += buitenwegEmissions
                                        linkEmissionsArray[ls][tod][
                                            hybridRouteSnelweg, et] += snelwegEmissions

                                        # CO2-emissions for the current trip
                                        if et == id_co2:
                                            parcelTripsCO2[currentTrips[trip, -1]] = (
                                                np.sum(stadEmissions) +
                                                np.sum(buitenwegEmissions) +
                                                np.sum(snelwegEmissions))

                                else:
                                    parcelTripsCO2[currentTrips[trip, -1]] = 0

        if root is not None:
            root.progressBar['value'] = (
                70.0 +
                (75.0 - 70.0) * (tod + 1) / nHours)

        # ------------ Emissions and intensities (serv/constr vans) -----------

        logger.debug("\t\tVan trips (service/construction)...")

        id_van = dims.get_id_from_label("vehicle_type", "Van")
        capUt = 0.5  # Assume 50% loading for service/construction vans

        for i in range(nOrigSelection):
            origZone = origSelection[i]
            destZones = np.where(
                (vanTripsService[origZone, :] > 0) |
                (vanTripsConstruction[origZone, :] > 0))[0]

            routes = [
                [get_route( i, zoneToCentroid[j], prevVan[r], linkDict) for j in destZones]
                for r in range(varDict['N_MULTIROUTE'])]
                
            if doHybridRoutes:
                hybridRoutes = [
                    [get_route(i, zoneToCentroid[j], prevVanHybrid[r], linkDict) for j in destZones]
                    for r in range(varDict['N_MULTIROUTE'])]

            # Van: Service segment
            for j in range(len(destZones)):
                destZone = destZones[j]

                tripIsZEZ = False
                if varDict['LABEL'] == 'UCC':
                    tripIsZEZ = ((i in ZEZzones) or (j in ZEZzones))

                if vanTripsService[origZone, destZone] > 0:
                    for r in range(varDict['N_MULTIROUTE']):
                        nTrips = (
                            vanTripsService[origZone, destZone] /
                            varDict['N_MULTIROUTE'])

                        if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                            route = hybridRoutes[r][j]
                        else:
                            route = routes[r][j]

                        # Number of trips made on each link
                        linkVanTripsArray[route, 0] += nTrips

                        routeStad = route[roadtypeArray[route] == 1]
                        routeBuitenweg = route[roadtypeArray[route] == 2]
                        routeSnelweg = route[roadtypeArray[route] == 3]

                        if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                            ZEZstad = np.where(ZEZarray[routeStad] == 1)[0]
                            ZEZbuitenweg = np.where(ZEZarray[routeBuitenweg] == 1)[0]
                            ZEZsnelweg = np.where(ZEZarray[routeSnelweg] == 1)[0]

                        for et in range(nET):
                            stadEmissions = (
                                nTrips * distArray[routeStad] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_stad, et, capUt))
                            buitenwegEmissions = (
                                nTrips * distArray[routeBuitenweg] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_buitenweg, et, capUt))
                            snelwegEmissions = (
                                nTrips * distArray[routeSnelweg] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_snelweg, et, capUt))
    
                            if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                                stadEmissions[ZEZstad] = 0.0
                                buitenwegEmissions[ZEZbuitenweg] = 0.0
                                snelwegEmissions[ZEZsnelweg] = 0.0
    
                            linkVanEmissionsArray[0][
                                routeStad, et] += stadEmissions
                            linkVanEmissionsArray[0][
                                routeBuitenweg, et] += buitenwegEmissions
                            linkVanEmissionsArray[0][
                                routeSnelweg, et] += snelwegEmissions

                # Van: Construction segment
                if vanTripsConstruction[origZone, destZone] > 0:
                    for r in range(varDict['N_MULTIROUTE']):
                        nTrips = (
                            vanTripsConstruction[origZone, destZone] /
                            varDict['N_MULTIROUTE'])

                        if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                            route = hybridRoutes[r][j]
                        else:
                            route = routes[r][j]

                        # Number of trips made on each link
                        linkVanTripsArray[route, 1] += nTrips

                        routeStad = route[roadtypeArray[route] == 1]
                        routeBuitenweg = route[roadtypeArray[route] == 2]
                        routeSnelweg = route[roadtypeArray[route] == 3]

                        if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                            ZEZstad = np.where(
                                ZEZarray[routeStad] == 1)[0]
                            ZEZbuitenweg = np.where(
                                ZEZarray[routeBuitenweg] == 1)[0]
                            ZEZsnelweg = np.where(
                                ZEZarray[routeSnelweg] == 1)[0]

                        for et in range(nET):
                            stadEmissions = (
                                nTrips * distArray[routeStad] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_stad, et, capUt))
                            buitenwegEmissions = (
                                nTrips * distArray[routeBuitenweg] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_buitenweg, et, capUt))
                            snelwegEmissions = (
                                nTrips * distArray[routeSnelweg] *
                                get_applicable_emission_fac(
                                    emissionFactors, id_van, id_snelweg, et, capUt))
    
                            if varDict['LABEL'] == 'UCC' and tripIsZEZ:
                                stadEmissions[ZEZstad] = 0.0
                                buitenwegEmissions[ZEZbuitenweg] = 0.0
                                snelwegEmissions[ZEZsnelweg] = 0.0
    
                            linkVanEmissionsArray[1][
                                routeStad, et] += stadEmissions
                            linkVanEmissionsArray[1][
                                routeBuitenweg, et] += buitenwegEmissions
                            linkVanEmissionsArray[1][
                                routeSnelweg, et] += snelwegEmissions

            if i % int(round(nOrigSelection / 100)) == 0:
                print(f"{round(i / nOrigSelection * 100, 1)}%", end='\r')

            if root is not None:
                root.progressBar['value'] = (
                    75.0 +
                    (85.0 - 75.0) * (i + 1) / nOrigSelection)

        if doShiftVanToElectric:
            linkVanEmissionsArray[0] *= (1 - varDict['SHIFT_VAN_TO_COMB1'])
            linkVanEmissionsArray[1] *= (1 - varDict['SHIFT_VAN_TO_COMB1'])

        # Make some space available on the RAM
        del prevVan, vanTripsService, vanTripsConstruction
        
        if doHybridRoutes:
            del prevVanHybrid

        # Write the intensities and emissions into the links-DataFrames
        for tod in timeOfDays:

            # The DataFrame to be exported to GeoJSON
            MRDHlinks.loc[:, volCols] += linkTripsArray[tod].astype(int)

            # The DataFrame to be exported to CSV
            MRDHlinksIntensities.loc[:, volCols] += linkTripsArray[tod].astype(int)
            MRDHlinksIntensities.loc[:, f'N_TOD{tod}'] += (
                linkTripsArray[tod][:, -1].astype(int))

            # Total emissions and per logistic segment
            for ls in range(nLS):
                for et, nameET in dictET.items():

                    # The DataFrame to be exported to GeoJSON
                    MRDHlinks[nameET] += linkEmissionsArray[ls][tod][:, et]

                    # The DataFrame to be exported to CSV
                    MRDHlinksIntensities[nameET] += linkEmissionsArray[ls][tod][:, et]
                    MRDHlinksIntensities[f"{nameET}_LS{ls}"] += linkEmissionsArray[ls][tod][:, et]

        if root is not None:
            root.progressBar['value'] = 87.0

        # Number of van trips
        linkVanTripsArray = np.round(linkVanTripsArray, 3)
        MRDHlinks.loc[:, 'N_VAN_S'] = linkVanTripsArray[:, 0]
        MRDHlinks.loc[:, 'N_VAN_C'] = linkVanTripsArray[:, 1]
        MRDHlinks.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 0]
        MRDHlinks.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 1]
        MRDHlinks.loc[:, 'N_TOT'] += linkVanTripsArray[:, 0]
        MRDHlinks.loc[:, 'N_TOT'] += linkVanTripsArray[:, 1]
        MRDHlinksIntensities.loc[:, 'N_VAN_S'] = linkVanTripsArray[:, 0]
        MRDHlinksIntensities.loc[:, 'N_VAN_C'] = linkVanTripsArray[:, 1]
        MRDHlinksIntensities.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 0]
        MRDHlinksIntensities.loc[:, 'N_VEH7'] += linkVanTripsArray[:, 1]
        MRDHlinksIntensities.loc[:, 'N_TOT'] += linkVanTripsArray[:, 0]
        MRDHlinksIntensities.loc[:, 'N_TOT'] += linkVanTripsArray[:, 1]

        # Emissions from van trips
        for et, nameET in dictET.items():
            MRDHlinksIntensities[f"{nameET}_VAN_S"] = linkVanEmissionsArray[0][:, et]
            MRDHlinksIntensities[f"{nameET}_VAN_C"] = linkVanEmissionsArray[1][:, et]
            MRDHlinks[nameET] += linkVanEmissionsArray[0][:, et]
            MRDHlinks[nameET] += linkVanEmissionsArray[1][:, et]
            MRDHlinksIntensities[nameET] += linkVanEmissionsArray[0][:, et]
            MRDHlinksIntensities[nameET] += linkVanEmissionsArray[1][:, et]

        if root is not None:
            root.progressBar['value'] = 90.0

        logger.debug("\tWriting link intensities to CSV...")

        MRDHlinks['A'] = [nodeDict[x] for x in MRDHlinks['A']]
        MRDHlinks['B'] = [nodeDict[x] for x in MRDHlinks['B']]
        MRDHlinksIntensities['LINKNR'] = MRDHlinks['LINKNR']
        MRDHlinksIntensities['A'] = MRDHlinks['A']
        MRDHlinksIntensities['B'] = MRDHlinks['B']
        MRDHlinksIntensities['LENGTH'] = MRDHlinks['LENGTH']
        MRDHlinksIntensities['ZEZ'] = MRDHlinks['ZEZ']
        MRDHlinksIntensities['Gemeentena'] = MRDHlinks['Gemeentena']

        cols = ['LINKNR', 'A', 'B', 'LENGTH', 'ZEZ', 'Gemeentena']
        for col in intensityFields:
            cols.append(col)

        MRDHlinksIntensities[cols].to_csv(
            f"{varDict['OUTPUTFOLDER']}links_loaded_{varDict['LABEL']}_intensities.csv",
            index=False
        )

        if doSelectedLink:
            logger.debug("\tWriting selected link analysis to CSV...")

            write_select_link_analysis(selectedLinkTripsArray, MRDHlinks, varDict)

        # Make some space available on the RAM
        del linkTripsArray, linkEmissionsArray
        del linkVanTripsArray, linkVanEmissionsArray

        if root is not None:
            root.progressBar['value'] = 93.0

        # -------------------- Enriching tours and shipments ------------------

        try:

            logger.debug("\tWriting emissions into Tours...")

            tours = write_emissions_into_tours(
                f"{varDict['OUTPUTFOLDER']}Tours_{varDict['LABEL']}.csv",
                tripsCO2
            )

            logger.debug("\tWriting emissions into Shipments...")

            write_emissions_into_shipments(
                f"{varDict['OUTPUTFOLDER']}Shipments_AfterScheduling_{varDict['LABEL']}.csv",
                tours, zoneDict, varDict
            )

            logger.debug("\tWriting emissions into ParcelSchedule...")

            write_emissions_into_tours(
                f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}.csv",
                parcelTripsCO2
            )

        except Exception:
            logger.warning(
                "Writing emissions into Tours/ParcelSchedule/Shipments failed!" +
                f"{sys.exc_info()[0]}\n{traceback.format_exc()}"
            )

        if root is not None:
            root.progressBar['value'] = 95.0

        # ------------------ Export loaded network to shapefile ---------------

        if exportShp:

            logger.debug("\tExporting network to .shp...")

            write_links_to_shp(
                MRDHlinks, MRDHlinksGeometry, intensityFieldsGeojson, varDict, root)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
