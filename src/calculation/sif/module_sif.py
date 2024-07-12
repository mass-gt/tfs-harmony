import logging
import numpy as np
import pandas as pd
import sys
import traceback

from calculation.common.dimensions import ModelDimensions

from typing import Any, Dict

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Spatial Interaction Freight module.
    """
    try:

        if varDict['NUTSLEVEL_INPUT'] not in [2, 3]:
            raise BaseException(
                "Error! NUTSLEVEL_INPUT needs to be either 2 or 3. " +
                "Current value is: " + str(varDict['NUTSLEVEL_INPUT']) + '.')

        nNSTR = len(dims.nstr) - 1
        nForeignZones = 3

        # ------------------- Importing and preparing data --------------------

        logger.debug("\tImporting and preparing data...")

        logger.debug("\t\tCoefficients for production and attraction...")

        # Importing production/attraction coefficients
        coeffProd = dict(
            ((int(row['nstr']), str(row['parameter'])), float(row['value']))
            for row in pd.read_csv(varDict['PARAMS_SIF_PROD'], sep='\t').to_dict('records')
        )
        coeffAttr = dict(
            ((int(row['nstr']), str(row['parameter'])), float(row['value']))
            for row in pd.read_csv(varDict['PARAMS_SIF_ATTR'], sep='\t').to_dict('records')
        )

        logger.debug(f"\t\tCommodity matrix NUTS level: {varDict['NUTSLEVEL_INPUT']}")

        if varDict['NUTSLEVEL_INPUT'] == 2:
            comMatNUTS2 = pd.read_csv(varDict['COMMODITYMATRIX'], sep='\t')
        elif varDict['NUTSLEVEL_INPUT'] == 3:
            comMatNUTS3 = pd.read_csv(varDict['COMMODITYMATRIX'], sep='\t')

        logger.debug("\t\tCoupling table MRDH to NUTS3...")

        MRDHtoNUTS3 = pd.read_csv(varDict['MRDH_TO_NUTS3'], sep='\t')
        MRDHtoNUTS3.index = MRDHtoNUTS3['zone_mrdh']

        logger.debug("\t\tMRDH SEGS...")

        segs = pd.read_csv(varDict['SEGS'], sep=',')
        segs.index = segs['zone']

        # Checking for which MRDH-zones in the SEGS we know the NUTS3-zone
        temp = np.array(MRDHtoNUTS3.index)
        toKeep = [x for x in segs['zone'] if x in temp]
        segs = segs.loc[toKeep, :]

        # Determing SEGS per NUTS3-zones
        segs['NUTS3'] = MRDHtoNUTS3.loc[segs['zone'], 'zone_nuts3']
        segs = pd.pivot_table(
            segs,
            values=[row['Comment'] for row in dims.employment_sector.values()],
            index='NUTS3',
            aggfunc=np.sum)

        logger.debug("\t\tDC surface per NUTS3...")

        dcData = pd.read_csv(varDict['DC_OPP_NUTS3'], sep='\t')
        dcData.index = dcData['zone_nuts']
        dcZones = set(np.array(dcData['zone_nuts']))
        surfaceDC = {}
        for nuts3 in segs.index:
            if nuts3 in dcZones:
                surface = dcData['surface_m2'][nuts3]
                surfaceDC[nuts3] = surface
            else:
                surfaceDC[nuts3] = 0
        surfaceDC = pd.DataFrame(surfaceDC.values(), index=surfaceDC.keys())

        # -------- Determine production and attraction per municipality -------

        logger.debug("\tDetermining production and attraction per NUTS3...")

        nNUTS3 = len(segs)
        codesNUTS3 = list(segs.index)

        for country in ['BE', 'DE', 'FR']:
            codesNUTS3.append(country)
        codesNUTS3 = np.array(codesNUTS3, dtype=str)

        production = np.zeros((nNUTS3 + nForeignZones, nNSTR), dtype=float)
        attraction = np.zeros((nNUTS3 + nForeignZones, nNSTR), dtype=float)

        for nstr in range(nNSTR):
            production[:nNUTS3, nstr] = (
                coeffProd.get((nstr, 'industrie'), 0.0) * np.array(segs['INDUSTRIE']) +
                coeffProd.get((nstr, 'detail'   ), 0.0) * np.array(segs['DETAIL'   ]) +
                coeffProd.get((nstr, 'landbouw' ), 0.0) * np.array(segs['LANDBOUW' ]) +
                coeffProd.get((nstr, 'diensten' ), 0.0) * np.array(segs['DIENSTEN' ]) +
                coeffProd.get((nstr, 'overheid' ), 0.0) * np.array(segs['OVERHEID' ]) +
                coeffProd.get((nstr, 'overig'   ), 0.0) * np.array(segs['OVERIG'   ]) +
                coeffProd.get((nstr, 'dc_opp'   ), 0.0) * np.array(surfaceDC[0]))

            attraction[:nNUTS3, nstr] = (
                coeffAttr.get((nstr, 'industrie'), 0.0) * np.array(segs['INDUSTRIE']) +
                coeffAttr.get((nstr, 'detail'   ), 0.0) * np.array(segs['DETAIL'   ]) +
                coeffAttr.get((nstr, 'landbouw' ), 0.0) * np.array(segs['LANDBOUW' ]) +
                coeffAttr.get((nstr, 'diensten' ), 0.0) * np.array(segs['DIENSTEN' ]) +
                coeffAttr.get((nstr, 'overheid' ), 0.0) * np.array(segs['OVERHEID' ]) +
                coeffAttr.get((nstr, 'overig'   ), 0.0) * np.array(segs['OVERIG'   ]) +
                coeffAttr.get((nstr, 'dc_opp'   ), 0.0) * np.array(surfaceDC[0]))

            # 3 international zones
            i = nNUTS3
            for country in ['BE', 'DE', 'FR']:

                if varDict['NUTSLEVEL_INPUT'] == 2:
                    production[i, nstr] = np.sum(comMatNUTS2.loc[
                        (comMatNUTS2['orig_nuts'] == country) & (comMatNUTS2['nstr'] == nstr),
                        'tonnes_year'])
                    attraction[i, nstr] = np.sum(comMatNUTS2.loc[
                        (comMatNUTS2['dest_nuts'] == country) & (comMatNUTS2['nstr'] == nstr),
                        'tonnes_year'])

                elif varDict['NUTSLEVEL_INPUT'] == 3:
                    production[i, nstr] = np.sum(comMatNUTS3.loc[
                        (comMatNUTS3['orig_nuts'] == country) & (comMatNUTS3['nstr'] == nstr),
                        'tonnes_year'])
                    attraction[i, nstr] = np.sum(comMatNUTS3.loc[
                        (comMatNUTS3['dest_nuts'] == country) & (comMatNUTS3['nstr'] == nstr),
                        'tonnes_year'])

                i += 1

        production = pd.DataFrame(
            production,
            index=codesNUTS3,
            columns=['NSTR' + str(nstr) for nstr in range(nNSTR)])
        attraction = pd.DataFrame(
            attraction,
            index=codesNUTS3,
            columns=['NSTR' + str(nstr) for nstr in range(nNSTR)])

        # --------- Create initial matrix for the distribution procedure ------

        logger.debug("\tCreating initial matrices per NSTR...")

        # Initialize list with OD-arrays for tonnes between NUTS3-regions
        tonnes = [
            np.zeros((nNUTS3 + nForeignZones, nNUTS3 + nForeignZones))
            for nstr in range(nNSTR)]

        # Initial matrix with NUTS2-input
        if varDict['NUTSLEVEL_INPUT'] == 2:

            # Coupling from NUTS3 (index) to NUTS2
            NUTS3toNUTS2 = ['' for i in range(nNUTS3 + nForeignZones)]

            for i in range(nNUTS3):
                NUTS3toNUTS2[i] = codesNUTS3[i][:4]

            for i in range(nForeignZones):
                NUTS3toNUTS2[nNUTS3 + i] = ['BE', 'DE', 'FR'][i]

            NUTS3toNUTS2 = np.array(NUTS3toNUTS2, dtype=str)

            # Get NUTS2-matrix as list with OD-DataFrame per NSTR
            comMatNUTS2byNSTR = [
                pd.pivot_table(
                    comMatNUTS2[comMatNUTS2['nstr'] ==  nstr],
                    values=['tonnes_year'],
                    index=['orig_nuts'],
                    columns=['dest_nuts']).fillna(0)
                for nstr in range(nNSTR)]

            for nstr in range(nNSTR):
                comMatNUTS2byNSTR[nstr].columns = [
                    comMatNUTS2byNSTR[nstr].columns[i][1]
                    for i in range(len(comMatNUTS2byNSTR[nstr].columns))]

            for nstr in range(nNSTR):

                # For each NUTS3 i-j the tonnes of the overarching NUTS2
                for i in range(nNUTS3 + nForeignZones):
                    origNUTS3 = i
                    destsNUTS3 = np.arange(nNUTS3 + nForeignZones)
                    origNUTS2 = NUTS3toNUTS2[origNUTS3]
                    destsNUTS2 = NUTS3toNUTS2[destsNUTS3]

                    tonnes[nstr][origNUTS3, destsNUTS3] = np.array(
                        comMatNUTS2byNSTR[nstr].loc[origNUTS2, destsNUTS2])

                # Per NUTS3-region the production of all NUTS3-regions
                # in the same NUTS2-region
                productionTotal = np.zeros(nNUTS3 + nForeignZones)
                for i in range(nNUTS3 + nForeignZones):
                    origNUTS2 = NUTS3toNUTS2[i]
                    NUTS3inSameNUTS2 = codesNUTS3[
                        np.where(NUTS3toNUTS2 == origNUTS2)[0]]
                    productionTotal[i] = np.sum(
                        production.loc[NUTS3inSameNUTS2, 'NSTR' + str(nstr)])

                # Per NUTS3-region the attraction of all NUTS3-regions
                # in the same NUTS2-region
                attractionTotal = np.zeros(nNUTS3 + nForeignZones)
                for i in range(nNUTS3 + nForeignZones):
                    destNUTS2 = NUTS3toNUTS2[i]
                    NUTS3inSameNUTS2 = codesNUTS3[
                        np.where(NUTS3toNUTS2 == destNUTS2)[0]]
                    attractionTotal[i] = np.sum(
                        attraction.loc[NUTS3inSameNUTS2, 'NSTR' + str(nstr)])

                # Create the initial matrix at the level of NUTS3-regions
                for i in range(nNUTS3 + 3):
                    if productionTotal[i] > 0:
                        tonnes[nstr][i, :] *= production.iat[i, nstr] / productionTotal[i]
                for j in range(nNUTS3 + 3):
                    if attractionTotal[j] > 0:
                        tonnes[nstr][:, j] *= attraction.iat[i, nstr] / attractionTotal[j]

        # Initial matrix with NUTS3-input
        if varDict['NUTSLEVEL_INPUT'] == 3:

            # Get NUTS3-matrix as list with OD-DataFrame per NSTR
            comMatNUTS3byNSTR = [
                pd.pivot_table(
                    comMatNUTS3[comMatNUTS3['nstr'] ==  nstr],
                    values=['tonnes_year'],
                    index=['orig_nuts'],
                    columns=['dest_nuts']).fillna(0)
                for nstr in range(nNSTR)]
            for nstr in range(nNSTR):
                comMatNUTS3byNSTR[nstr].columns = [
                    comMatNUTS3byNSTR[nstr].columns[i][1]
                    for i in range(len(comMatNUTS3byNSTR[nstr].columns))]

            NUTS3inIndex = set(list(comMatNUTS3byNSTR[0].index))
            NUTS3inHeader = set(list(comMatNUTS3byNSTR[0].columns))

            for nstr in range(nNSTR):
                for i in range(nNUTS3 + nForeignZones):

                    if codesNUTS3[i] in NUTS3inIndex:
                        for j in range(nNUTS3 + nForeignZones):

                            if codesNUTS3[j] in NUTS3inHeader:
                                tonnes[nstr][i, j] = (
                                    comMatNUTS3byNSTR[nstr].at[codesNUTS3[i], codesNUTS3[j]])
                            else:
                                tonnes[nstr][i, j] = 0

                    else:
                        for j in range(nNUTS3 + nForeignZones):
                            tonnes[nstr][i, j] = 0

                        if nstr == 0:
                            logger.warning(
                                f'NUTS3-regio {codesNUTS3[i]}' +
                                ' was not found in the commodity matrix.' +
                                ' Defaulting to 0.0 tonnes for this NUTS3-region.')

        # ------------------------ FRATAR distribution ------------------------

        tolerance = 0.005
        maxIter = 50

        logger.debug("\tFRATAR distribution...")

        for nstr in range(nNSTR):
            itern = 0
            conv = tolerance + 100
            convPrevIteration = -99999

            while (itern < maxIter) and (conv > tolerance):
                itern += 1
                maxColScaleFac = 0
                totalRows = np.sum(tonnes[nstr], axis=0)

                # Scale to row totals
                for j in range(nNUTS3 + nForeignZones):
                    total = totalRows[j]

                    if total > 0:
                        scaleFacCol = attraction.iat[j, nstr] / total

                        if abs(scaleFacCol) > abs(maxColScaleFac):
                            maxColScaleFac = scaleFacCol

                        tonnes[nstr][:, j] *= scaleFacCol

                maxRowScaleFac = 0
                totalCols = np.sum(tonnes[nstr], axis=1)

                # Scale to column totals
                for i in range(nNUTS3 + nForeignZones):
                    total = totalCols[i]

                    if total > 0:
                        scaleFacRow = production.iat[i, nstr] / total

                        if abs(scaleFacRow) > abs(maxRowScaleFac):
                            maxRowScaleFac = scaleFacRow

                        tonnes[nstr][i, :] *= scaleFacRow

                # Calculate convergence to check if we should continue scaling
                conv = round(max(
                    abs(maxColScaleFac - 1),
                    abs(maxRowScaleFac - 1)), 5)

                # Stop if convergence is not improved anymore
                if conv == convPrevIteration:
                    break
                convPrevIteration = conv

            logger.debug(f"\t\tNSTR {nstr} (Iteration {itern}  / Convergence {round(conv, 4)})")

        # -------------------- Exporting commodity matrix ---------------------

        logger.debug("\tExporting commodity matrix...")

        labelsNUTS3 = list(production.index)
        nZones = nNUTS3 + nForeignZones

        # Put in long-matrix-format (enumated)
        outputMat = np.zeros((nZones * nZones * nNSTR, 4), dtype=object)
        for nstr in range(nNSTR):
            for orig in range(nZones):
                indices = (
                    nstr * nZones * nZones +
                    orig * nZones +
                    np.arange(nZones))
                outputMat[indices, 0] = labelsNUTS3[orig]
                outputMat[indices, 1] = labelsNUTS3
                outputMat[indices, 2] = nstr
                outputMat[indices, 3] = tonnes[nstr][orig,:]

        # Formatting
        outputMat = pd.DataFrame(
            outputMat,
            columns=['ORIG', 'DEST', 'NSTR', 'TonnesYear'])
        outputMat['NSTR'] = outputMat['NSTR'].astype(int)
        outputMat['TonnesYear'] = outputMat['TonnesYear'].astype(float)
        outputMat['TonnesYear'] = np.round(outputMat['TonnesYear'], 3)

        # Exporting to CSV
        outputMat.to_csv(
            varDict['OUTPUTFOLDER'] + 'CommodityMatrixNUTS3.csv',
            sep=',',
            index=False)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
