import logging
import numpy as np
import pandas as pd
import sys
import traceback

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_shape, get_skims
from calculation.common.vrt import draw_choice_mcs
from .support_parcel_dmnd import get_cum_shares_vt_ucc, write_parcels_to_geojson, aggregate_parcels

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
            root.progressBar['value'] = 0.0

        # ------------------------- Import data -------------------------------

        logger.debug("\tImporting data...")

        zones = read_shape(varDict['ZONES'])
        zones = pd.DataFrame(zones).sort_values('AREANR')
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
        zoneDict = dict(np.transpose(np.vstack((
            np.arange(1, nIntZones + 1),
            zones['AREANR']))))
        zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones + i + 1] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        segs = pd.read_csv(varDict['SEGS'])
        segs.index = segs['zone']

        parcelNodes, coords = read_shape(varDict['PARCELNODES'],returnGeometry=True)
        parcelNodes['X'] = [coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y'] = [coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes.index= parcelNodes['id'].astype(int)
        parcelNodes = parcelNodes.sort_index()
        nParcelNodes = len(parcelNodes)

        cepShares = pd.read_csv(varDict['CEP_SHARES'], sep='\t')
        cepShares.index = cepShares['courier']

        cepList = np.unique(parcelNodes['CEP'])
        cepNodes = [
            np.where(parcelNodes['CEP'] == str(cep))[0]
            for cep in cepList]
        cepNodeDict = {}
        for cepNo in range(len(cepList)):
            cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]

        if root is not None:
            root.progressBar['value'] = 1.0

        # ------------- Get skim data and make parcel skim for REF ------------
        skimTravTime, skimDistance, nZones = get_skims(varDict)
        parcelSkim = np.zeros((nZones, nParcelNodes))

        # Skim with travel times between parcel nodes and all other zones
        i = 0
        for parcelNodeZone in parcelNodes['AREANR']:
            orig = invZoneDict[parcelNodeZone]
            dest = 1 + np.arange(nZones)

            # Time in hours
            parcelSkim[:, i] = np.round(
                (skimTravTime[(orig - 1) * nZones + (dest - 1)] / 3600), 4)

            i += 1

        if root is not None:
            root.progressBar['value'] = 2.0

        # ---------- Import and prepare data for microhub scenario ------------
        if varDict['LABEL'].startswith('MIC'):

            mode = varDict['LABEL'][-3:]

            # In case the mode label consists of 2 characters
            mode = mode.replace("_", "")

            # Check if vehicle tag is valid
            if 'collab' in varDict['LABEL']:
                tier = 'Horizontal Collaboration'
            elif 'indiv' in varDict['LABEL']:
                tier = 'Individual CEP'
            else:
                raise Exception(
                    'Invalid scenario input: ' +
                    'consolidation type in LABEL '
                    "'" + varDict['LABEL'] + '".')

            # Read information about vehicle types and scenario configuration
            vehicleTypes = pd.read_csv(varDict['VEHICLETYPES'], index_col=2)
            modeLabelsDict = dict(zip(vehicleTypes.index, vehicleTypes['Name']))
            modeNumbersDict = dict(zip(vehicleTypes.index, vehicleTypes['Veh_ID']))

            if mode not in vehicleTypes.index:
                raise Exception(
                    f"Invalid scenario input: vehicle type in LABEL '{varDict['LABEL']}' not found in VEHICLETYPES.")

            else:

                logger.debug(f"\tRunning microhubs scenario with{tier} and {modeLabelsDict[mode]}.")

                # Read csv with microhubs (ID, areanr, CEP)
                microhubs = pd.read_csv(varDict['MICROHUBS'], index_col=0)

                # Find and add coordinates of microhub zones
                microhubs['X'] = [
                    zones.iloc[i]['X'].copy()
                    for i in microhubs['AREANR']]
                microhubs['Y'] = [
                    zones.iloc[i]['Y'].copy()
                    for i in microhubs['AREANR']]

                mh_id_dict = dict(zip(microhubs.index, microhubs['AREANR']))

                # Make skim with travel DISTANCES between
                # parcel nodes and all other zones
                nZones = int(len(skimDistance) ** 0.5)
                distSkim = np.zeros((nZones, nParcelNodes))
                i = 0
                for parcelNodeZone in parcelNodes['AREANR']:
                    orig = invZoneDict[parcelNodeZone]
                    dest = 1 + np.arange(nZones)
                    distSkim[:, i] = np.round(skimDistance[(orig - 1) * nZones + (dest - 1)], 4)
                    i += 1

                if tier == 'Horizontal Collaboration':

                    # Get selected hubs for mode and tier from input data
                    hubsConfig = [
                        int(x) for x in
                        vehicleTypes['collab_microhubs'][mode].split(",")]
                    nMH = len(hubsConfig)
                    hubZones = microhubs['AREANR'][hubsConfig]

                    # Make skim for selected MHs and all zones with chosen mode
                    mh_to_zone_skim = np.zeros((nZones, nMH))
                    i = 0
                    for mh_zone in hubZones:
                        mh_orig = invZoneDict[mh_zone]
                        mh_dest = np.arange(1, nZones + 1)

                        # Time skim in seconds
                        mh_to_zone_skim[:, i] = np.round(
                            (skimDistance[(mh_orig - 1) * nZones + (mh_dest - 1)] /
                             vehicleTypes['AvgSpeed'][mode] / 3.6), 4)

                        i += 1

                    # For each zone, find the closest microhub
                    # based on travel time in mh_to_zone_skim
                    closest_MH = pd.DataFrame(columns=["MH_ID", "MH_AREA"])
                    closest_MH["MH_ID"] = pd.DataFrame(
                        mh_to_zone_skim).idxmin(axis=1) + 1
                    closest_MH["MH_AREA"] = [
                        mh_id_dict[closest_MH["MH_ID"][i]]
                        for i in closest_MH.index]

                    # Add column to main zones df:
                    # for zones served from microhub, fill in areanr
                    # of their closest hub
                    for i in zones.index:
                        if zones.at[i, "ZEZ"] == 2:
                            zones.at[i, "MH_zone"] = (
                                closest_MH.loc[invZoneDict[i], "MH_AREA"])
                        else:
                            zones.at[zones['ZEZ'] != 2, "MH_zone"] = 0

                # Individual CEP
                else:

                    # Get the hubs that are part of the scenario
                    hubsConfig = list(map(int, vehicleTypes['individual_microhubs'][mode].split(",")))
                    hubsIndex = [x - 1 for x in hubsConfig]
                    microhubsConfig = microhubs.iloc[hubsIndex]
                    cep_hubs = [
                        microhubsConfig.index[microhubsConfig['CEP'] == cep].tolist()
                        for cep in microhubsConfig.CEP.unique()]
                    selectedHubsByCEP = {k: v for k, v in zip(microhubsConfig['CEP'].unique(), cep_hubs)}

                    for courier in cepList:

                        # Number of MH of current cep
                        nMH = len(selectedHubsByCEP[courier])
                        logger.debug(f"\t\tPreparing microhub(s) no. {selectedHubsByCEP[courier][1:-1]} for {courier}")

                        # If only one hub of courier is selected,
                        # that's the one from which the courier will deliver
                        if nMH == 1:
                            current_mh_zone = mh_id_dict[
                                selectedHubsByCEP[courier][0]]
                            zones[f"MH_zone_{courier}"] = 0
                            zones.loc[
                                zones['ZEZ'] == 2,
                                f"MH_zone_{courier}"] = current_mh_zone

                        # If 2+ hubs of the same courier are in the selection,
                        # determine which one serves which area
                        if nMH > 1:
                            # Make skim for selected MHs and all zones
                            mh_to_zone_skim = np.zeros((nZones, nMH))
                            i = 0
                            for mh in selectedHubsByCEP[courier]:
                                mh_orig = invZoneDict[mh_id_dict[mh]]
                                mh_dest = np.arange(1, nZones + 1)

                                # Time skim in seconds
                                mh_to_zone_skim[:, i] = np.round(
                                    (skimDistance[(mh_orig - 1) * nZones + (mh_dest - 1)] /
                                     (vehicleTypes['AvgSpeed'][mode] / 3.6)), 4)

                                i += 1

                            # For each zone, find the closest microhub based
                            # on travel time in mh_to_zone_skim
                            closest_MH = pd.DataFrame(columns=["MH_ID", "MH_AREA"])

                            # Find for each zone the closest MH with MH_ID
                            closest_MH_pos = pd.DataFrame
                            (mh_to_zone_skim).idxmin(axis=1) + 1
                            closest_MH["MH_ID"] = [
                                selectedHubsByCEP[courier][closest_MH_pos[x] - 1]
                                for x in closest_MH_pos.index]
                            closest_MH["MH_AREA"] = [
                                mh_id_dict[closest_MH["MH_ID"][i]]
                                for i in closest_MH.index]

                            # Add column to main zones df:
                            # for zones served from microhub,
                            # fill in areanr of their closest hub
                            zones[f"MH_zone_{courier}"] = 0
                            zones.loc[zones['ZEZ'] == 2, f"MH_zone_{courier}"] = [
                                closest_MH.loc[invZoneDict[i], "MH_AREA"]
                                for i in zones.index[zones['ZEZ'] == 2]]

        if root is not None:
            root.progressBar['value'] = 3.0

        # ------------------ Start parcel generation --------------------------
        # Generate parcels for each zone based on households and employment
        # and select a parcel node for each parcel

        logger.debug("\tGenerating parcels...")

        logger.debug("\t\tB2B parcels...")

        # Calculate number of parcels per zone
        # based on number of households and
        # total number of parcels on an average day
        zones['parcels'] = (
            segs['9: arbeidspl_totaal'] *
            varDict['PARCELS_PER_EMPL'] / varDict['PARCELS_SUCCESS_B2B'])

        # ------- Calculate parcels per zone with parcel demand model ---------
        logger.debug("\t\tB2C parcels...")

        """
        1) make df with all zone/age/income combinations
        2) distribute zonal population over combinations
        3) add up 3 params (sted, inc, age) for each combination
        4) calculate cumprobs and then probs
        5) multiply each prob with corresponding no of parcels and ppl in the row
        6) aggregate to zone level divide by 60 to get daily no of parcels
        """

        demandParams = pd.read_csv(varDict['PARAMS_ECOMMERCE'], sep=',')

        # ---------------------------------------------------------------------
        # 1) make df with all zone/age/income combinations

        logger.debug("\t\t\tEnumerate all zone/age/income combinations...")

        zoneList = list(zones['AREANR'])

        # Update to grab values from SEGs once they are ready
        ageList = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Update to grab values from SEGs once they are ready
        incList = [1, 2, 3, 4, 5, 6]

        demandIndex = pd.MultiIndex.from_product(
            [zoneList, ageList, incList],
            names=['zone', 'agegr', 'incgr'])

        demandDF = pd.DataFrame(
            data=np.zeros(len(demandIndex)),
            index=demandIndex)
        demandDF = demandDF.rename(columns={0: 'pers', 1: 'sted'})
        demandDF['sted'] = 0

        if root is not None:
            root.progressBar['value'] = 5.0

        # ---------------------------------------------------------------------
        # 2) distribute zonal population over combinations

        logger.debug("\t\t\tDistribute zonal population over combinations...")

        segs = segs.fillna(0)

        # Loop over zones, agegr and incgr
        # and add population and urbanisation level to each row of demandDF
        for z in zoneList:
            tmpNumPers = segs.at[z, '2: inwoners']
            tmpSTED = segs.at[z, 'STED']

            for a in ageList:
                tmpKLEEFT = segs.at[z, f'KLEEFT_{a}']
                
                for i in incList:
                    tmpHHINK = segs.at[z, f'HHINK_{i}']

                    # Add population:
                    # (people of age a) / zonal population * (people in income classes i)
                    if tmpNumPers > 0:
                        pers = tmpKLEEFT / tmpNumPers * tmpHHINK
                        demandDF.at[(z, a, i), 'pers'] = pers

                    # Add urbanisation level
                    demandDF.at[(z, a, i), 'sted'] = tmpSTED

        if root is not None:
            root.progressBar['value'] = 15.0

        # ---------------------------------------------------------------------
        # 3) add the 3 corresponding parameter values (for sted, inc, age)
        # to each row and calculate sum

        logger.debug("\t\t\tAdd up 3 params (sted, inc, age) for each combination...")

        demandDF['param_age'] = 0.00
        demandDF['param_inc'] = 0.00
        demandDF['param_urb'] = 0.00

        # Add age param
        for a in ageList:
            param_age = demandParams['Estimate'][demandParams['Parameter'] == f'KLEEFT2_{a}']

            for i in incList:
                for z in zoneList:
                    demandDF.at[(z, a, i), 'param_age'] = param_age

        if root is not None:
            root.progressBar['value'] = 30.0

        # Add income param
        for i in incList:
            param_inc = demandParams['Estimate'][demandParams['Parameter'] == f'HHBRUTOINK2_w5_{i}']

            for z in zoneList:
                for a in ageList:
                    demandDF.at[(z, a, i), 'param_inc'] = param_inc 

        if root is not None:
            root.progressBar['value'] = 45.0

        # Add urbanisation parameter
        # This loop seems to be necessary as we do not have
        # a parameter for sted = 0
        for z in zoneList:
            sted = segs.at[z, 'STED']

            if sted > 0:
                sted_param = demandParams.loc[
                    demandParams['Parameter'] == f'STED_GM_{int(sted)}',
                    'Estimate']

            else:
                sted_param = 0

            for a in ageList:
                for i in incList:
                    demandDF.at[(z, a, i), 'param_urb'] = sted_param

        demandDF['param_sum'] = (
            demandDF['param_age'] +
            demandDF['param_inc'] +
            demandDF['param_urb'])

        if root is not None:
            root.progressBar['value'] = 60.0

        # ---------------------------------------------------------------------
        # 4) calculate cumprobs and then probs

        logger.debug("\t\t\tCalculative cumulative probabilities...")

        # Make dictionary with threshold parameters (mu)
        # and corresponding number of parcels
        mu_dict = {}
        parcel_levels = [0, 1, 2, 3, 4, 5, 10, 15, 20]
        for x in parcel_levels:
            mu_dict[x] = demandParams['Estimate'].loc[
                demandParams['Parameter'] == f'WI_nongroceries_{x}'].iloc[-1]

        # Calculate cumulative probabilities with logit formula
        for p in parcel_levels[:-1]:
            mu = mu_dict[p]
            demandDF[f'cprob_{p}p'] = (
                1 + np.exp(demandDF['param_sum'] - mu)) ** (-1)

        # Cum prob for highest category is 1 by definition
        demandDF['cprob_20p'] = 1

        # Calculate probabilities (differences between consecutive cprob)
        for p in parcel_levels[1:]:
            demandDF[f'prob_{p}p'] = (
                demandDF[f'cprob_{p}p'] -
                demandDF[f'cprob_{parcel_levels[parcel_levels.index(p) - 1]}p'])

        # Probability 0 parcels
        demandDF['prob_0p'] = demandDF['cprob_0p']

        # ---------------------------------------------------------------------
        # 5) multiply each prob with corresponding no of
        # parcels and ppl in the row
        demandDF['parcels_pp'] = 0

        for p in parcel_levels:
            demandDF['parcels_pp'] += (
                demandDF[f'prob_{p}p'] *
                demandParams['Parcels'].loc[demandParams['Parameter'] == f'WI_nongroceries_{p}'].iloc[-1]
            )

        demandDF['parcels'] = demandDF['parcels_pp'] * demandDF['pers']

        if root is not None:
            root.progressBar['value'] = 65.0

        # ---------------------------------------------------------------------
        # 6) aggregate to zone level and divide by 60 to
        # get daily no of parcels

        logger.debug("\t\t\tAggregate to zonal level...")

        demandDF = demandDF.replace(np.nan, 0)
        demandPerZone = pd.pivot_table(
            demandDF.reset_index(),
            values=['parcels', 'pers'],
            index=['zone'],
            aggfunc=np.sum)

        # Add the B2C parcels to the B2B parcels
        zones['parcels'] += (demandPerZone['parcels'] / 60)

        # Spread over couriers based on market shares
        for cep in cepList:
            zones['parcels_' + str(cep)] = np.array(
                np.round(cepShares['share_total'][cep] * zones['parcels']),
                dtype=int)

        # Total number of parcels per courier
        nParcels = int(zones[
            ["parcels_" + str(cep) for cep in cepList]].sum().sum())

        # Put parcel demand in Numpy array (faster indexing)
        cols = ['Parcel_ID', 'O_zone', 'D_zone', 'DepotNumber']
        parcels = np.zeros((nParcels, len(cols)), dtype=int)
        parcelsCep = np.array(['' for i in range(nParcels)], dtype=object)

        if root is not None:
            root.progressBar['value'] = 70.0

        # Now determine for each zone and courier from which depot
        # the parcels are delivered
        count = 0

        # Loop over zones
        for zoneID in zones['AREANR']:

            # If there are parcels for the selected zones
            if zones['parcels'][zoneID] > 0:

                # Loop over all CEPs for that zone
                for cep in cepList:
                    # Select dc of current CEP based on min in parcelSkim
                    parcelNodeIndex = cepNodeDict[cep][parcelSkim[
                        invZoneDict[zoneID] - 1, cepNodeDict[cep]].argmin()]

                    # Fill the df parcels with parcels, zone after zone.
                    # Parcels consist of ID, D and O zone and parcel node
                    # number in ongoing df from index count-1 the next x=no.
                    # of parcels rows, fill the cell in the column Parcel_ID
                    # with a number
                    n = zones.loc[zoneID, 'parcels_' + str(cep)]

                    # Parcel_ID
                    parcels[count:(count + n), 0] = (
                        np.arange(count + 1, count + 1 + n, dtype=int))

                    # O_zone
                    parcels[count:(count + n), 1] = (
                        parcelNodes['AREANR'][parcelNodeIndex + 1])

                    # D_zone, DepotNumber and CEP
                    parcels[count:(count + n), 2] = zoneID
                    parcels[count:(count + n), 3] = parcelNodeIndex + 1
                    parcelsCep[count:(count + n)] = cep

                    count += zones['parcels_' + str(cep)][zoneID]

        # Put the parcel demand data back in a DataFrame
        parcels = pd.DataFrame(parcels, columns=cols)
        parcels['CEP'] = parcelsCep

        # Default vehicle type for parcel deliveries: vans
        parcels['VEHTYPE'] = dims.get_id_from_label("vehicle_type", "Van")

        if root is not None:
            root.progressBar['value'] = 75.0

        # ----------- Extra steps for rerouting through microhubs -------------

        # Rerouting through microhubs
        if varDict['LABEL'].startswith('MIC'):
            # Write the REF parcel demand
            logger.debug(f"\tWriting REF parcels to ParcelDemand_{tier}_{mode}_.csv")
            parcels.to_csv(f"{varDict['OUTPUTFOLDER']}ParcelDemand_REF.csv", index=False)

            logger.debug(f"\tRedirecting parcels through microhubs...")

            parcels['FROM_MH'] = 0
            parcels['TO_MH'] = 0

            # Store destinations of parcels
            destZones = np.array(parcels['D_zone'].astype(int))

            # Store depot numbers (origins of parcels)
            depotNumbers = np.array(parcels['DepotNumber'].astype(int))

            # Store indices of parcels in destZones where destZone is in ZEZ==2
            mh_parcels = np.where(zones['ZEZ'][destZones] == 2)[0]

            newParcels = np.zeros((len(mh_parcels), parcels.shape[1]), dtype=object)

            count = 0
            for parcel_id in mh_parcels: 

                trueDest = destZones[parcel_id]
                cep = parcelsCep[parcel_id]

                if tier =='Horizontal Collaboration':
                    mhzone = zones['MH_zone'][trueDest]
                if tier =='Individual CEP':
                    mhzone = zones[f'MH_zone_{cep}'][trueDest]

                # Leg A: from depots to MH
                # (change existing record of the parcel)
                parcels.at[parcel_id, 'D_zone'] = mhzone.copy()   # MH as destination leg A
                parcels.at[parcel_id, 'TO_MH'] = microhubs.index[
                    microhubs['AREANR'] == mhzone][0]
                parcels.at[parcel_id, 'VEHTYPE'] = modeNumbersDict["TR"]

                # Leg B: microhub to final destination
                # (make new record to add to the end of the parcels-df)
                newParcels[count, 1] = mhzone                         # MH as origin Leg B 
                newParcels[count, 2] = trueDest                       # Destination leg B (zone of HH or business)
                newParcels[count, 3] = depotNumbers[parcel_id]        # Depot ID 
                newParcels[count, 4] = cep                            # Courier name

                if mode == "AR":
                    dist = np.round((skimDistance[
                        (invZoneDict[mhzone] - 1) * nZones + (invZoneDict[trueDest] - 1)]), 4)

                    # Radius of a microhub for AR operations
                    radius = 500

                    # Vehicle type electric bike
                    if dist > radius:
                        newParcels[count, 5] = modeNumbersDict["EB"]
                    # Vehicle type autonomous robot
                    else:
                        newParcels[count, 5] = modeNumbersDict[mode]

                # Green vehicle type
                else:
                    newParcels[count, 5] = modeNumbersDict[mode]

                #  MH that is origin of this leg
                newParcels[count, 6] = microhubs.index[microhubs['AREANR'] == mhzone][0]

                 # To MH is zero here (leg B)
                newParcels[count, 7] = 0

                count += 1

            newParcels = pd.DataFrame(newParcels)
            newParcels.columns = parcels.columns

            dtypes = {
                'Parcel_ID': int,
                'O_zone': int,
                'D_zone': int,
                'DepotNumber': int,
                'CEP': str,
                'VEHTYPE': int,
                'FROM_MH': int,
                'TO_MH': int}
            for col in dtypes.keys():
                newParcels[col] = newParcels[col].astype(dtypes[col])

            parcels = parcels.append(newParcels)
            parcels.index = np.arange(len(parcels))
            parcels['Parcel_ID'] = np.arange(1, len(parcels) + 1)

            nParcels = len(parcels)

            logger.debug(f"\t\t{len(newParcels)} out of {nParcels} parcels are redirected through microhubs")

            if mode == "AR":
                nARparcels = len(newParcels[newParcels['VEHTYPE'] == 13])
                logger.debug(f"\t{nARparcels} out of {len(newParcels)} microhub parcels are delivered by AR")

        # ---------------------------- UCC ------------------------------------
        # Rerouting through UCCs in the UCC-scenario
        if varDict['LABEL'] == 'UCC': 

            # Write the REF parcel demand
            logger.debug(f"\tWriting parcels to {varDict['OUTPUTFOLDER']}ParcelDemand_REF.csv")
            parcels.to_csv(f"{varDict['OUTPUTFOLDER']}ParcelDemand_REF.csv", index=False)  

            # Consolidation potential per logistic segment (for UCC scenario)
            probConsolidation = float(pd.read_csv(
                varDict['ZEZ_CONSOLIDATION'], sep='\t', index_col='logistic_segment'
            ).at[dims.get_id_from_label("logistic_segment", "Parcel (consolidated flows)"), "probability"])

            # Vehicle/combustion shares (for UCC scenario)
            cumSharesVehUCC = get_cum_shares_vt_ucc(varDict, dims)

            logger.debug(f"\tRedirecting parcels via UCC...")

            parcels['FROM_UCC'] = 0
            parcels['TO_UCC'] = 0

            origZones = np.array(parcels['O_zone'].astype(int))
            destZones = np.array(parcels['D_zone'].astype(int))
            depotNumbers = np.array(parcels['DepotNumber'].astype(int))
            whereDestZEZ = np.where(
                (zones['ZEZ'][destZones] >= 1) &
                (probConsolidation > np.random.rand(len(parcels))))[0]

            newParcels = np.zeros(parcels.shape, dtype=object)

            uccZones = np.unique(zones.loc[zones['UCC_zone'] != 0, 'UCC_zone'])
            uccZonesGemeente = np.array(zones.loc[uccZones, 'GEMEENTEN'])
            nUccZones = len(uccZones)

            count = 0

            for i in whereDestZEZ:

                trueOrig = origZones[i]
                trueDest = destZones[i]
                trueDestGemeente = zones.at[trueDest, 'GEMEENTEN']
                #newDest = zones['UCC_zone'][trueDest]

                distsFirstLeg = [
                    skimDistance[(invZoneDict[trueOrig] - 1) * nZones + (invZoneDict[uccZones[j]] - 1)]
                    for j in range(nUccZones)]
                distsSecondLeg = [
                    skimDistance[(invZoneDict[uccZones[j]] - 1) * nZones + (invZoneDict[trueDest] - 1)]
                    for j in range(nUccZones)]
                dists = np.array(distsFirstLeg) + np.array(distsSecondLeg)
                dists[np.where(uccZonesGemeente != trueDestGemeente)[0]] = 1000000
                newDest = uccZones[np.argmin(dists)]

                # Redirect to UCC
                parcels.at[i, 'D_zone'] = newDest
                parcels.at[i, 'TO_UCC'] = 1

                # Add parcel set to ZEZ from UCC
                newParcels[count, 1] = newDest          # Origin
                newParcels[count, 2] = trueDest         # Destination
                newParcels[count, 3] = depotNumbers[i]  # Depot ID
                newParcels[count, 4] = parcelsCep[i]    # Courier name
                newParcels[count, 6] = 1                # From UCC
                newParcels[count, 7] = 0                # To UCC

                # Vehicle type
                newParcels[count, 5] = draw_choice_mcs(cumSharesVehUCC)

                count += 1

            newParcels = pd.DataFrame(newParcels)
            newParcels.columns = parcels.columns
            newParcels = newParcels.iloc[np.arange(count), :]

            dtypes = {
                'Parcel_ID': int,
                'O_zone': int,
                'D_zone': int,
                'DepotNumber': int,
                'CEP': str,
                'VEHTYPE': int,
                'FROM_UCC': int,
                'TO_UCC': int}
            for col in dtypes.keys():
                newParcels[col] = newParcels[col].astype(dtypes[col])

            parcels = parcels.append(newParcels)
            parcels.index = np.arange(len(parcels))
            parcels['Parcel_ID'] = np.arange(1, len(parcels) + 1)

            nParcels = len(parcels)

        if root is not None:
            root.progressBar['value'] = 90.0

        # ------------------------- Prepare output ----------------------------

        # Write the parcels to CSV (each row is a parcel)
        logger.debug(
            f"\tWriting parcels CSV to {varDict['OUTPUTFOLDER']}ParcelDemand_{varDict['LABEL']}.csv")

        parcels.to_csv(
            varDict['OUTPUTFOLDER'] + "ParcelDemand_" + varDict['LABEL'] + ".csv",
            index=False)

        if root is not None:
            root.progressBar['value'] = 95.0

        # Aggregate to number of parcels per zone and export to geojson
        logger.debug(
            f"\tWriting parcels GeoJSON to {varDict['OUTPUTFOLDER']}ParcelDemand_{varDict['LABEL']}.geojson")

        parcelsAggr = aggregate_parcels(parcels, varDict)

        write_parcels_to_geojson(parcelsAggr, parcelNodes, zonesX, zonesY, varDict)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
