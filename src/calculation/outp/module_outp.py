import logging
import numpy as np
import pandas as pd
import sys
import traceback
import os.path

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_mtx, read_shape

from typing import Any, Dict

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Output module.
    """
    try:

        # ------------------- Open output files -------------------------------

        if root is not None:
            root.progressBar['value'] = 0

        # Define folders relative to current datapath
        datapathI = varDict['INPUTFOLDER']
        datapathO = varDict['OUTPUTFOLDER']
        datapathP = varDict['PARAMFOLDER']
        zonesPath = varDict['ZONES']
        skimTravTimePath = varDict['SKIMTIME']
        skimDistancePath = varDict['SKIMDISTANCE']  
        label = varDict['LABEL']

        datapathOI = datapathO + 'Output indicators/'

        logger.debug("\tImporting data...")

        if not os.path.isdir(datapathOI) and os.path.isdir(datapathO):
            os.mkdir(datapathOI)

        outfile = open(f"{datapathOI}Output_OutputIndicator_{label}.csv", "w")
        sep = ','

        # ---------------------- Import data ----------------------------------

        vtNums = np.array([row['ID'] for row in dims.vehicle_type.values()], dtype=int)
        vtNames = np.array([row['Comment'] for row in dims.vehicle_type.values()], dtype=str)
        lsNames = np.array([row['Comment'] for row in dims.logistic_segment.values()], dtype=str)
        etNames = np.array([row['Comment'] for row in dims.emission_type.values()], dtype=str)

        nVT = len(dims.vehicle_type)
        nLS = len(dims.logistic_segment)
        nNSTR = len(dims.nstr) - 1
        nCombType = len(dims.combustion_type)
        nShipSize = len(dims.shipment_size)

        if root is not None:
            root.progressBar['value'] = 0.1

        logger.debug("\t\tImporting shipments...")

        shipments = pd.read_csv(
            f"{datapathO}Shipments_{label}.csv",
            index_col=None)

        if root is not None:
            root.progressBar['value'] = 5.0

        logger.debug("\t\tImporting tours...")

        trips = pd.read_csv(f'{datapathO}Tours_{label}.csv')
        trips['TOUR_ID'] = [
            str(trips.at[i, 'CARRIER_ID']) + '_' + str(trips.at[i, 'TOUR_ID'])
            for i in trips.index]

        if root is not None:
            root.progressBar['value'] = 10.0

        logger.debug("\t\tImporting parcels...")

        parcels = pd.read_csv(f"{datapathO}ParcelDemand_{label}.csv")
        parcelTrips = pd.read_csv(f"{datapathO}ParcelSchedule_{label}.csv")
        parcels = parcels.rename(
            columns={
                'O_zone': 'ORIG',
                'D_zone': 'DEST'})
        parcelTrips = parcelTrips.rename(
            columns={
                'O_zone': 'ORIG',
                'D_zone': 'DEST',
                'Trip_ID': 'TRIP_ID',
                'Tour_ID': 'TOUR_ID'})  

        if root is not None:
            root.progressBar['value'] = 13.0

        logger.debug("\t\tPreparing tours/parcels...")

        temp = pd.pivot_table(
            parcelTrips,
            values='TRIP_ID',
            index='TOUR_ID',
            aggfunc=len)
        nTripsParcelTour = {}
        for i in temp.index:
            nTripsParcelTour[i] = temp.at[i, 'TRIP_ID']
        parcelTrips['N_trips_tour'] = [
            nTripsParcelTour[tourID] for tourID in parcelTrips['TOUR_ID']]

        tours = trips[trips['TRIP_ID'] == 0]
        parcelTours = parcelTrips[
            [parcelTrips['TRIP_ID'][i][-2:] == '_0'
             for i in parcelTrips.index]]

        if root is not None:
            root.progressBar['value'] = 20.0

        logger.debug("\t\tImporting van trips...")

        vanTripsFound = (
            os.path.isfile(datapathO + 'TripsVanService.mtx') and
            os.path.isfile(datapathO + 'TripsVanConstruction.mtx'))

        if vanTripsFound:
            vanTripsService = read_mtx(
                datapathO + 'TripsVanService.mtx')
            vanTripsConstruction = read_mtx(
                datapathO + 'TripsVanConstruction.mtx')
        else:
            logger.debug("\t\t\tVan trips not found in outputfolder.")

        if root is not None:
            root.progressBar['value'] = 30.0

        # Skims with travel times and distances
        logger.debug("\t\tImporting skims...")

        skimTravTime = read_mtx(skimTravTimePath)
        skimDistance = read_mtx(skimDistancePath)
        nZones = int(len(skimTravTime)**0.5)

        if root is not None:
            root.progressBar['value'] = 40.0

        # Import zonal data
        logger.debug("\t\tImporting zonal data...")

        zonesShape = read_shape(zonesPath)
        zonesShape.sort_values('AREANR')
        zonesShape.index = zonesShape['AREANR']
        zoneID = np.array(zonesShape['AREANR'])
        nInternalZones = len(zonesShape)
        zoneDict = dict(np.transpose(np.vstack((np.arange(nInternalZones), zoneID))))
        for row in pd.read_csv(varDict['SUP_COORDINATES_ID'], sep='\t').to_dict('records'):
            zoneDict[nInternalZones + int(row['zone_corop']) - 1] = int(row['zone_mrdh'])
        invZoneDict = dict((v, k) for k, v in zoneDict.items())
        zoneID = np.arange(nInternalZones)

        if root is not None:
            root.progressBar['value'] = 42.0

        # The distance of each shipment and trip in kilometers
        logger.debug("\t\tCalculating distance of each trip...")

        trips['DIST'] = skimDistance[np.array([
            invZoneDict[trips['ORIG'][i]] * nZones +
            invZoneDict[trips['DEST'][i]]
            for i in trips.index], dtype=int)] / 1000

        parcelTrips['DIST'] = skimDistance[np.array([
            invZoneDict[parcelTrips['ORIG'][i]] * nZones +
            invZoneDict[parcelTrips['DEST'][i]]
            for i in parcelTrips.index], dtype=int)] / 1000

        shipments['DIST'] = skimDistance[np.array([
            invZoneDict[shipments['ORIG'][i]] * nZones +
            invZoneDict[shipments['DEST'][i]]
            for i in shipments.index], dtype=int)] / 1000

        if root is not None:
            root.progressBar['value'] = 50.0

        # Import the loaded network
        logger.debug("\t\tImporting loaded network...")

        linksLoaded = pd.read_csv(
            datapathO + 'links_loaded_' + str(label) + '_intensities.csv')

        intCols = [
            'LINKNR', 'A', 'B', 'ZEZ',
            'N_LS0', 'N_LS1', 'N_LS2',
            'N_LS3', 'N_LS4', 'N_LS5',
            'N_LS6', 'N_LS7',
            'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3',
            'N_VEH4', 'N_VEH5', 'N_VEH6', 'N_VEH7',
            'N_VEH8', 'N_VEH9',
            'N_TOT']
        linksLoaded[intCols] = linksLoaded[intCols].astype(int)
        linksLoaded['Gemeentena'] = linksLoaded['Gemeentena'].astype(str)

        linksLoadedRdam = linksLoaded.loc[linksLoaded['Gemeentena'] == 'Rotterdam', :]
        linksLoadedDH = linksLoaded.loc[linksLoaded['Gemeentena'] == "'s-Gravenhage", :]
        linksLoadedZH = linksLoaded.loc[linksLoaded['Gemeentena'] != "nan", :]
        linksLoadedZEZ = linksLoaded.loc[linksLoaded['ZEZ'] == 1, :]

        if root is not None:
            root.progressBar['value'] = 60.0

        # ---------------------- Shipment sizes -------------------------------

        logger.debug("\tCalculating and exporting output indicators...")

        logger.debug("\t\tShipments")

        # Actual shipment sizes
        shipments['WEIGHT_LEVELS'] = 0
        for i in range(nShipSize):
            lowerSize = dims.shipment_size[i]['Lower']
            upperSize = dims.shipment_size[i]['Upper']
            shipments.loc[
                (shipments['WEIGHT'] >= lowerSize) & (shipments['WEIGHT'] < upperSize),
                'WEIGHT_LEVELS'] = i
        shipmentSizeHist = np.unique(
            shipments['WEIGHT_LEVELS'],
            return_counts=True)
        shipmentSizeLabels = np.array([row['Comment'] for row in dims.shipment_size.values()], dtype=str)

        outfile.write('Shipment size; actual weight (freight)' + sep + '\n')
        outfile.write(sep + 'Weight (tonnes)' + sep + 'Number of shipments\n')

        for i in range(nShipSize):
            outfile.write(
                sep + shipmentSizeLabels[i] +
                sep + str(shipmentSizeHist[1][i]) + '\n')

        # Chosen shipment size category
        shipmentSizeHist = np.unique(
            shipments['WEIGHT_CAT'],
            return_counts=True)
        outfile.write(
            '\nShipment size; chosen weight class (freight)' + sep + '\n')
        outfile.write(sep + 'Weight (tonnes)' + sep + 'Number of shipments\n')

        for i in range(nShipSize):
            outfile.write(
                sep + shipmentSizeLabels[i] +
                sep + str(shipmentSizeHist[1][i]) + '\n')

        # Chosen vehicle type per shipment
        nShipsVeh = np.zeros(nVT)
        outfile.write(
            '\nNumber of shipments (freight; by vehicle type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type' + sep + 'Number of shipments\n')

        for veh in range(nVT):
            nShipsVeh[veh] = len(shipments[shipments['VEHTYPE'] == veh])
            outfile.write(
                sep + vtNames[veh] +
                sep + str(nShipsVeh[veh]) + '\n')

        # Number of shipments by logistic segment
        nShipsLS = np.zeros(nLS)
        outfile.write('\nNumber of shipments (Total) ')
        outfile.write('(freight/parcels; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Number of shipments\n')

        for ls in range(nLS - 1):
            nShipsLS[ls] = len(shipments[shipments['LS'] == ls])
            outfile.write(sep + lsNames[ls] + sep + str(nShipsLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS - 1] + sep + str(len(parcels)) + '\n')

        # Number of shipments by logistic segment
        ZEZshipments = shipments[shipments['DEST'] < 99999900]
        ZEZshipments = ZEZshipments[
            np.array(zonesShape.loc[np.array(ZEZshipments['DEST'], dtype=int), 'ZEZ'] == 1)]
        ZEZparcels = parcels[parcels['DEST'] < 99999900]
        ZEZparcels = ZEZparcels[
            np.array(zonesShape.loc[np.array(ZEZparcels['DEST'], dtype=int), 'ZEZ'] == 1)]
        nShipsLS = np.zeros(nLS)

        outfile.write('\nNumber of shipments (to ZEZ) ')
        outfile.write('(freight/parcels; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Number of shipments\n')

        for ls in range(nLS - 1):
            nShipsLS[ls] = len(ZEZshipments[ZEZshipments['LS'] == ls])
            outfile.write(sep + lsNames[ls] + sep + str(nShipsLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS - 1] + sep + str(len(ZEZparcels)) + '\n')
        
        if root is not None:
            root.progressBar['value'] = 65.0

        # ------------------------- Number of trips ---------------------------

        logger.debug("\t\tTrips")

        # Number of trips by direction
        nTripsTotal = trips.shape[0]
        nTripsLeaving = trips[trips['DEST'] > 99999900].shape[0]
        nTripsEntering = trips[trips['ORIG'] > 99999900].shape[0]
        nTripsIntra = nTripsTotal - nTripsLeaving - nTripsEntering

        outfile.write('\nNumber of trips (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Number of trips'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(nTripsIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(nTripsLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(nTripsEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(nTripsTotal)    + '\n')

        # Number of parcel trips by direction
        nParcelTripsTotal = parcelTrips.shape[0]
        nParcelTripsLeaving = parcelTrips[parcelTrips['DEST'] > 99999900].shape[0]
        nParcelTripsEntering = parcelTrips[parcelTrips['ORIG'] > 99999900].shape[0]
        nParcelTripsIntra = (
            nParcelTripsTotal -
            nParcelTripsLeaving -
            nParcelTripsEntering)

        outfile.write('\nNumber of trips (parcel; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Number of trips'         + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(nParcelTripsIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(nParcelTripsLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(nParcelTripsEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(nParcelTripsTotal)    + '\n')

        # Number of trips by logistic segment and NSTR
        outfile.write('\nNumber of trips ')
        outfile.write('(freight/parcel; by logistic segment and NSTR)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for nstr in range(-1, nNSTR):
            outfile.write(str(nstr) + sep)
        outfile.write('\n')

        for ls in range(nLS - 1):
            outfile.write(sep + lsNames[ls])
            for nstr in range(-1, nNSTR):
                value = np.sum(
                    (trips['LS'] == ls) &
                    (trips['NSTR'] == nstr))
                outfile.write(sep + str(value))
            outfile.write('\n')

        # Number of trips by logistic segment and NSTR
        # (in/out/inside ZEZ Rotterdam)
        tripOrigs = np.array(trips['ORIG'])
        tripDests = np.array(trips['DEST'])
        zezZones = np.array(
            zonesShape.loc[
                (zonesShape['ZEZ'] == 1) & (zonesShape['Gemeentena'] == 'Rotterdam'),
                'AREANR'])
        tripsZEZ = trips.iloc[
            [i for i in range(len(trips))
             if tripOrigs[i] in zezZones or tripDests[i] in zezZones], :]

        outfile.write('\nNumber of trips ')
        outfile.write('(freight/parcel; by logistic segment and NSTR)' + sep + '\n')
        outfile.write('Only trips entering/leaving/inside ZEZ Rotterdam' + '\n')

        outfile.write(sep + 'LS' + sep)
        for nstr in range(-1, nNSTR):
            outfile.write(str(nstr) + sep)
        outfile.write('\n')

        for ls in range(nLS - 1):
            outfile.write(sep + lsNames[ls])
            for nstr in range(-1, nNSTR):
                value = np.sum(
                    (tripsZEZ['LS'] == ls) &
                    (tripsZEZ['NSTR'] == nstr))
                outfile.write(sep + str(value))
            outfile.write('\n')
            
        # Number of trips by vehicle type and combustion type
        nTripsVeh = [None for veh in range(nVT)]
        outfile.write('\nNumber of trips ')
        outfile.write('(freight; by vehicle type and combustion type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type')

        for comb in range(nCombType):
            outfile.write(sep + str(dims.combustion_type[comb]['Comment']))
        outfile.write('\n')

        for veh in range(nVT):
            nTripsVeh[veh] = len(trips[trips['VEHTYPE'] == veh])
            nTripsVehComb = [None for comb in range(nCombType)]
            for comb in range(nCombType):
                nTripsVehComb[comb] = np.sum(
                    (trips['VEHTYPE'] == veh) &
                    (trips['COMBTYPE'] == comb))
            outfile.write(sep + vtNames[veh])
            for comb in range(nCombType):
                outfile.write(sep + str(nTripsVehComb[comb]))
            outfile.write('\n')

        # Number of trips by logistic segment and vehicle type
        outfile.write('\nNumber of trips ')
        outfile.write('(freight/parcel; by logistic segment and vehicle type)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for vt in range(nVT):
            outfile.write(vtNames[vt] + sep)
        outfile.write('\n')

        for ls in range(nLS - 1):
            outfile.write(sep + lsNames[ls])
            for vt in range(nVT):
                value = np.sum(
                    (trips['LS'] == ls) &
                    (trips['VEHTYPE'] == vt))
                outfile.write(sep + str(value))
            outfile.write('\n')

        nParcelTripsVan  = np.sum(parcelTrips['VehType'] == 'Van')
        nParcelTripsLEVV = np.sum(parcelTrips['VehType'] == 'LEVV')
        outfile.write(
            sep + lsNames[nLS - 1] +
            sep + '0' + sep + '0' + sep + '0' + sep + '0' +
            sep + '0' + sep + '0' + sep + '0' + sep)
        outfile.write(str(nParcelTripsVan) + sep + str(nParcelTripsLEVV) + '\n')

        # Number of trips by van traffic segment
        outfile.write('\nNumber of trips (service vans; by logistic segment)' + sep + '\n')
        if vanTripsFound:
            outfile.write(
                sep + 'Segment' +
                sep + 'Number of trips\n')
            outfile.write(
                sep + 'Service' +
                sep + str(np.sum(vanTripsService)) + '\n')
            outfile.write(
                sep + 'Construction' +
                sep + str(np.sum(vanTripsConstruction)) + '\n')
        else:
            outfile.write(sep + 'Van trips not found in outputfolder.' + '\n')
            outfile.write('\n\n')

        if root is not None:
            root.progressBar['value'] = 75.0

        # ----------------------- Transported weight --------------------------

        logger.debug("\t\tTransported weight")

        # Transport weight by direction
        weightTotal = round(sum(tours['TOUR_WEIGHT']), 2)
        weightLeaving = round(sum(tours[tours['DEST'] > 99999900]['TOUR_WEIGHT']), 2)
        weightEntering = round(sum(tours[tours['ORIG'] > 99999900]['TOUR_WEIGHT']), 2)
        weightIntra = weightTotal - weightLeaving - weightEntering

        outfile.write('\nTransported tonnes (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Tonnes'            + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(weightIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(weightLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(weightEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(weightTotal)    + '\n')

        # Transported weight by NSTR
        weightNSTR = np.zeros(nNSTR)
        outfile.write('\nTransported tonnes (freight; by NSTR goods type)' + sep + '\n')
        outfile.write(sep + 'NSTR' + sep + 'Tonnes\n')

        for nstr in range(nNSTR):
            weightNSTR[nstr] = round(sum(tours[tours['NSTR'] == nstr]['TOUR_WEIGHT']), 2)
            outfile.write(sep + str(nstr) + sep + str(weightNSTR[nstr]) + '\n')

        # Transported weight by Vehicle type
        weightVeh = [None for veh in range(nVT)]
        outfile.write('\nTransported tonnes (freight; by vehicle type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type')
        for comb in range(nCombType):
            outfile.write(sep + str(dims.combustion_type[comb]['Comment']))
        outfile.write('\n')

        for veh in range(nVT):
            weightVeh[veh] = np.round(np.sum(tours[tours['VEHTYPE'] == veh]['TOUR_WEIGHT']),2)
            weightVehComb = [None for comb in range(nCombType)]
            for comb in range(nCombType):
                weightVehComb[comb] = np.round(np.sum(
                    tours[(tours['VEHTYPE'] == veh) & (tours['COMBTYPE'] == comb)]['TOUR_WEIGHT']), 2)
            outfile.write(sep + vtNames[veh])
            for comb in range(nCombType):
                outfile.write(sep + str(weightVehComb[comb]))
            outfile.write('\n')

        # Transported weight by logistic segment
        weightLS = np.zeros(nLS)
        outfile.write('\nTransported tonnes (freight; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Tonnes\n')
        for ls in range(nLS - 1):
            weightLS[ls] = round(sum(tours[tours['LS'] == ls]['TOUR_WEIGHT']), 2)
            outfile.write(sep + lsNames[ls] + sep + str(weightLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS - 1] + sep + 'n.a.' + '\n')

        if root is not None:
            root.progressBar['value'] = 78.0

        # ------------------------ Loading rate -------------------------------

        logger.debug("\t\tAverage trip loads")

        # Average trip load by direction
        loadedTrips = trips[trips['NSTR'] != -1]
        loadedTripsIntra = loadedTrips[
            [(
                loadedTrips['DEST'][i] <= 99999900 and
                loadedTrips['ORIG'][i] <= 99999900)
             for i in loadedTrips.index]]
        avgLoadIntra = round(loadedTripsIntra['TRIP_WEIGHT'].mean(), 2)
        avgLoadLeaving = round(loadedTrips[loadedTrips['DEST'] > 99999900]['TRIP_WEIGHT'].mean(),2)
        avgLoadEntering = round(loadedTrips[loadedTrips['ORIG'] > 99999900]['TRIP_WEIGHT'].mean(),2)
        avgLoad = round(loadedTrips['TRIP_WEIGHT'].mean(), 2)

        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by direction)*{sep}\n")
        outfile.write(f"{sep}Direction{sep}Average trip load (tonnes)\n")

        outfile.write(f"{sep}Inside study area{sep}{avgLoadIntra}\n")
        outfile.write(f"{sep}Leaving study area{sep}{avgLoadLeaving}\n")
        outfile.write(f"{sep}Entering study area{sep}{avgLoadEntering}\n")
        outfile.write(f"{sep}All loaded trips{sep}{avgLoad}\n")
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
        
        # Average load per NSTR
        avgLoadNSTR = np.zeros(nNSTR)
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by NSTR goods type)*{sep}\n")
        outfile.write(f"{sep}NSTR{sep}Average trip load (tonnes)\n")

        for nstr in range(nNSTR):
            avgLoadNSTR[nstr] = round(loadedTrips[
                loadedTrips['NSTR'] == nstr]['TRIP_WEIGHT'].mean(), 2)
            outfile.write(f"{sep}{nstr}{sep}{avgLoadNSTR[nstr]}\n")
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
        
        # Average load per vehicle type
        avgLoadVeh = np.zeros(nVT)
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by vehicle type)*{sep}\n")
        outfile.write(f"{sep}Vehicle type{sep}Average trip load (tonnes)\n")
        for veh in range(nVT):
            avgLoadVeh[veh] = round(loadedTrips[
                loadedTrips['VEHTYPE'] == veh]['TRIP_WEIGHT'].mean(),2)
            outfile.write(f"{sep}{vtNames[veh]}{sep}{avgLoadVeh[veh]}\n")

        outfile.write(f"{sep}*Excluding empty trips{sep}\n")

        # Average load per logistic segment
        avgLoadLS = np.zeros(nLS)
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by logistic segment)*{sep}\n")
        outfile.write(f"{sep}LS{sep}Average trip load (tonnes)\n")

        for ls in range(nLS - 1):
            avgLoadLS[ls] = round(loadedTrips[
                loadedTrips['LS'] == ls]['TRIP_WEIGHT'].mean(), 2)
            outfile.write(sep + lsNames[ls] + sep + str(avgLoadLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS - 1] + sep + 'n.a.' + '\n')

        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
        
        if root is not None:
            root.progressBar['value'] = 80.0 

        # --------------- Number of shipments per tour ------------------------

        logger.debug("\t\tNumber of shipments")

        nShipsTotal = pd.value_counts(tours['N_SHIP'], sort=False)
        nShipsNSTR = pd.crosstab(tours['N_SHIP'], tours['NSTR'])
        nShipsLS = pd.crosstab(tours['N_SHIP'], tours['LS'])

        toursInternal = tours[
            [tours['ORIG'][i] <= 99999900 and tours['DEST'][i] <= 99999900
             for i in tours.index]]
        nShipsTotalInternal = pd.value_counts(
            toursInternal['N_SHIP'],
            sort=False)
        nShipsNSTRInternal = pd.crosstab(
            toursInternal['N_SHIP'],
            toursInternal['NSTR'])
        nShipsLSInternal = pd.crosstab(
            toursInternal['N_SHIP'],
            toursInternal['LS'])
        
        outfile.write("\nNumber of shipments per tour (freight; by NSTR)\n")
        outfile.write(f"{sep}{sep}NSTR\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsNSTR.columns))[1:-1]}{sep}Total\n")

        for ships in nShipsNSTR.index:
            totalShips = sum(nShipsNSTR.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsNSTR.loc[ships, :]))[1:-1]}{sep}{totalShips}\n")
        
        outfile.write("\nNumber of shipments per tour within study area (freight; by NSTR)\n")
        outfile.write(f"{sep}{sep}NSTR\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsNSTRInternal.columns))[1:-1]}{sep}Total\n")

        for ships in nShipsNSTR.index:
            totalShips = sum(nShipsNSTRInternal.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsNSTRInternal.loc[ships, :]))[1:-1]}{sep}{totalShips}\n")

        outfile.write("\nNumber of shipments per tour (freight; by logistic segment)\n")
        outfile.write(f"{sep}{sep}Logistic segment\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsLS.columns))[1:-1]}{sep}Total\n")

        for ships in nShipsLS.index:
            totalShips = sum(nShipsLS.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsLS.loc[ships, :]))[1:-1]}{sep}{totalShips}\n")
        
        outfile.write("\nNumber of shipments per tour within study area (freight; by logistic segment)\n")
        outfile.write(f"{sep}{sep}Logistic segment\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsLSInternal.columns))[1:-1]}{sep}Total\n")

        for ships in nShipsLS.index:
            totalShips = sum(nShipsLSInternal.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsLSInternal.loc[ships, :]))[1:-1]}{sep}{totalShips}\n")   

        outfile.write("\nAverage number of trips per tour (freight/parcel; by logistic segment)\n")
        outfile.write('LS' + sep + 'Number of trips\n')

        for ls in range(nLS - 1):
            nToursLS = len(np.unique(trips.loc[trips['LS'] == ls,'TOUR_ID']))
            if nToursLS > 0:
                outfile.write(
                    lsNames[ls] + sep +
                    str(len(trips[trips['LS'] == ls]) / nToursLS) + '\n')
            else:
                outfile.write(lsNames[ls] + sep + 'n.a.' + '\n')
        outfile.write(
            lsNames[nLS - 1] + sep +
            str(np.mean(parcelTours['N_trips_tour'])) + '\n')
        outfile.write(
            'Total (freight)' + sep
            + str(len(trips) / len(np.unique(trips['TOUR_ID']))) + '\n')
        
        if root is not None:
            root.progressBar['value'] = 82.0

        # ------------------- Tonkilometers -----------------------------------

        # Trips tonkilometers
        trips['TONKM'] = trips['DIST'] * trips['TRIP_WEIGHT']
        tonKilometersTotal = round(sum(trips['TONKM']), 2)
        tonKilometersLeaving = round(sum(trips[trips['DEST'] > 99999900]['TONKM']), 2)
        tonKilometersEntering = round(sum(trips[trips['ORIG'] > 99999900]['TONKM']), 2)
        tonKilometersIntra = tonKilometersTotal - tonKilometersLeaving - tonKilometersEntering

        outfile.write('\nTonkilometers trips (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'TonKM'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(tonKilometersIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(tonKilometersLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(tonKilometersEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(tonKilometersTotal)    + '\n')
       
        # Shipments tonkilometers
        shipments['TONKM'] = shipments['DIST'] * shipments['WEIGHT']
        tonKilometersTotal = round(sum(shipments['TONKM']), 2)
        tonKilometersLeaving = round(sum(shipments[shipments['DEST'] > 99999900]['TONKM']), 2)
        tonKilometersEntering = round(sum(shipments[shipments['ORIG'] > 99999900]['TONKM']), 2)
        tonKilometersIntra  = tonKilometersTotal -  tonKilometersLeaving -  tonKilometersEntering

        outfile.write('\nTonkilometers shipments (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'TonKM'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(tonKilometersIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(tonKilometersLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(tonKilometersEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(tonKilometersTotal)    + '\n')
        
        if root is not None:
            root.progressBar['value'] = 85.0 

        # --------------- Vehicle Kilometers Travelled (Total) ------------------------
        
        logger.debug("\t\tVehicle Kilometers Travelled")

        # VKM (Total, skim-based) by logistic segment and vehicle type
        outfile.write('\nVehicle kilometers (Total; based on skim)')
        outfile.write('(freight/parcel; by logistic segment and vehicle type)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for vt in range(nVT):
            outfile.write(vtNames[vt] + sep)
        outfile.write('\n')

        for ls in range(nLS - 1):
            outfile.write(sep + lsNames[ls])
            for vt in range(nVT):
                value = np.sum(trips.loc[
                    (trips['LS']==ls) &
                    (trips['VEHTYPE']==vt), 'DIST'])
                outfile.write(sep + str(value))
            outfile.write('\n')
        nParcelTripsVan  = np.sum(parcelTrips.loc[parcelTrips['VehType'] == 'Van', 'DIST'])
        nParcelTripsLEVV = np.sum(parcelTrips.loc[parcelTrips['VehType'] == 'LEVV','DIST'])
        outfile.write(
            sep + lsNames[nLS - 1] +
            sep + '0' + sep + '0' + sep + '0' + sep +'0' +
            sep + '0' + sep + '0' + sep + '0' + sep)
        outfile.write(str(nParcelTripsVan) + sep + str(nParcelTripsLEVV) + '\n')

        # VKMs van (Total, skim-based) by van traffic segment
        outfile.write('\nVehicle kilometers (Total; based on skim)')
        outfile.write('(service vans; by logistic segment)' + sep + '\n')
        if vanTripsFound:
            outfile.write(sep + 'Segment' + sep + 'Number of trips\n')
            outfile.write(sep + 'Service'      + sep + str(np.sum(vanTripsService      * skimDistance / 1000)) + '\n')
            outfile.write(sep + 'Construction' + sep + str(np.sum(vanTripsConstruction * skimDistance / 1000)) + '\n')
        else:
            outfile.write(sep + 'Van trips not found in outputfolder.' + '\n')
            
        # VKMs (Total) by logistic segment
        VKT_LS = np.zeros(nLS)
        outfile.write("\nVehicle Kilometers Travelled (Total) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoaded['LENGTH'] * linksLoaded[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        # VKMs (Total) by vehicle type
        VKT_VEH = np.zeros(nVT)
        outfile.write("\nVehicle Kilometers Travelled (Total) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoaded['LENGTH'] * linksLoaded[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")       
        
        # VKMs van (Total=NL) by van traffic segment
        outfile.write("\nVehicle Kilometers Travelled (NL) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(sep + 'Service'      + sep + str(np.sum(linksLoaded['LENGTH'] * linksLoaded['N_VAN_S'])) + '\n')
        outfile.write(sep + 'Construction' + sep + str(np.sum(linksLoaded['LENGTH'] * linksLoaded['N_VAN_C'])) + '\n')
            
        if root is not None:
            root.progressBar['value'] = 90.0 

        # --------------- Vehicle Kilometers Travelled (Zuid-Holland) ------------------------

        # VKMs (ZH) by logistic segment
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")

        # VKMs (ZH) by vehicle type
        VKT_VEH = np.zeros(nVT)
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")    

        # VKMs parcel (ZH) by vehicle type
        VKT_VEH = np.zeros(nVT)
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            if f'N_LS8_VEH{i}' in linksLoaded.columns:
                temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_LS8_VEH{i}']
                VKT_VEH[i] = np.round(np.sum(temp), 2)
                outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")    

        # VKMs van (ZH) by van traffic segment
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(sep + 'Service'      + sep + str(np.sum(linksLoadedZH['LENGTH'] * linksLoadedZH['N_VAN_S'])) + '\n')
        outfile.write(sep + 'Construction' + sep + str(np.sum(linksLoadedZH['LENGTH'] * linksLoadedZH['N_VAN_C'])) + '\n')
         
        if root is not None:
            root.progressBar['value'] = 93.0

        # --------------- Vehicle Kilometers Travelled (Rotterdam) ------------------------

        # VKMs (Rdam) by logistic segment
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        # VKMs (Rdam) by vehicle type
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_VEH{veh}']
            VKT_VEH[veh] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{vtNames[veh]}{sep}{VKT_VEH[veh]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")               

        # VKMs parcel (Rdam) by vehicle type
        VKT_VEH = np.zeros(nVT)
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            if f'N_LS8_VEH{veh}' in linksLoaded.columns:
                temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_LS8_VEH{veh}']
                VKT_VEH[veh] = np.round(np.sum(temp),2)
                outfile.write(f"{sep}{vtNames[veh]}{sep}{VKT_VEH[veh]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n") 

        # VKMs van (Rdam) by van traffic segment
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(
            sep + 'Service' +
            sep + str(np.sum(
                linksLoadedRdam['LENGTH'] * linksLoadedRdam['N_VAN_S'])) +
            '\n')
        outfile.write(
            sep + 'Construction' +
            sep + str(np.sum(
                linksLoadedRdam['LENGTH'] * linksLoadedRdam['N_VAN_C'])) +
            '\n')

        if root is not None:
            root.progressBar['value'] = 95.0

        # --------------- Vehicle Kilometers Travelled (ZEZ) ------------------------

        # VKMs (ZEZ) by logistic segment
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in ZEZ) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedZEZ['LENGTH'] * linksLoadedZEZ[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")

        # VKMs (ZEZ) by vehicle type
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in ZEZ) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            temp = linksLoadedZEZ['LENGTH'] * linksLoadedZEZ[f'N_VEH{veh}']
            VKT_VEH[veh] = np.round(np.sum(temp), 2)
            outfile.write(f"{sep}{vtNames[veh]}{sep}{VKT_VEH[veh]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")            

        if root is not None:
            root.progressBar['value'] = 97.0

        # ----------------------- Emissions -----------------------------------

        logger.debug("\t\tEmissions\n")

        outfile.write('\nEmissions from network (freight/parcel/vans; Total)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in etNames:
            total = np.sum(linksLoaded[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in Zuid-Holland)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in etNames:
            total = np.sum(linksLoadedZH[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in Rotterdam)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in etNames:
            total = np.sum(linksLoadedRdam[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in The Hague)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in etNames:
            total = np.sum(linksLoadedDH[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in ZEZ)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in etNames:
            total = np.sum(linksLoadedZEZ[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (Total) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in etNames:
                total = np.sum(linksLoaded[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')                     
        for ls in ['VAN_S','VAN_C']:
            for emission in etNames:
                total = np.sum(linksLoaded[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')   

        if root is not None:
            root.progressBar['value'] = 98.0

        outfile.write('\nEmissions from network (in Zuid-Holland) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in etNames:
                total = np.sum(linksLoadedZH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in etNames:
                total = np.sum(linksLoadedZH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nEmissions from network (in Rotterdam) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in etNames:
                total = np.sum(linksLoadedRdam[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in etNames:
                total = np.sum(linksLoadedRdam[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nEmissions from network (in The Hague) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in etNames:
                total = np.sum(linksLoadedDH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in etNames:
                total = np.sum(linksLoadedDH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nEmissions from network (in ZEZ) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in etNames:
                total = np.sum(linksLoadedZEZ[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in etNames:
                total = np.sum(linksLoadedZEZ[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nEmissions from network (freight/parcel/vans; by municipality)\n')
        outfile.write('Municipality' + sep + 'Type' + sep + 'kg (all)' + sep + 'kg (parcel)' + '\n')
        for gem in [row['Comment'] for row in dims.municipality.values()]:
            currentLinks = linksLoaded[linksLoaded['Gemeentena'] == str(gem)]
            for emission in etNames:
                total = np.sum(currentLinks[emission])
                totalParcel = np.sum(currentLinks[emission + '_LS8'])
                outfile.write(str(gem) + sep + emission + sep + str(total) + sep + str(totalParcel) + '\n')                   

        if root is not None:
            root.progressBar['value'] = 99.0

        outfile.write('\nCO2 Emissions from network (in Zuid-Holland) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2']:
                total = np.sum(linksLoadedZH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2']:
                total = np.sum(linksLoadedZH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nCO2 Emissions from network (in Rotterdam) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2']:
                total = np.sum(linksLoadedRdam[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2']:
                total = np.sum(linksLoadedRdam[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nCO2 Emissions from network (in The Hague) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2']:
                total = np.sum(linksLoadedDH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2']:
                total = np.sum(linksLoadedDH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nCO2 Emissions from network (in ZEZ) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2']:
                total = np.sum(linksLoadedZEZ[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2']:
                total = np.sum(linksLoadedZEZ[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  

        outfile.write('\nCO2 Emissions from network (freight/parcel/vans; by municipality)\n')
        outfile.write('Municipality' + sep + 'Type' + sep + 'kg (all)' + sep + 'kg (parcel)' + '\n')
        for gem in [row['Comment'] for row in dims.municipality.values()]:
            currentLinks = linksLoaded[linksLoaded['Gemeentena'] == str(gem)]

            for emission in ['CO2']:
                total = np.sum(currentLinks[emission])
                totalParcel = np.sum(currentLinks[emission + '_LS8'])
                outfile.write(str(gem) + sep + emission + sep + str(total) + sep + str(totalParcel) + '\n')     
  
        logger.debug(f"\tTables written to {datapathOI}Output_Outputindicator_{label}.csv.")

        if root is not None:
            root.progressBar['value'] = 100.0

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
