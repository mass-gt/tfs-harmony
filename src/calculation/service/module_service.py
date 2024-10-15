import logging
import numpy as np
import pandas as pd
import sys
import traceback

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_shape, get_skims, write_mtx

from typing import Any, Dict

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Service module.
    """
    try:
        tolerance = 0.005
        maxIter = 25

        # --------------------------- Import data -----------------------------------

        logger.debug("\tImporting data...")

        # Import cost parameters
        id_van = dims.get_id_from_label("vehicle_type", "Van")
        costParams = dict(
            (int(row['vehicle_type']), row)
            for row in pd.read_csv(varDict['COST_VEHTYPE'], sep='\t').to_dict('records')
        )[id_van]

        # Import distance decay parameters
        id_service = dims.get_id_from_label("van_segment", "Service")
        id_construction = dims.get_id_from_label("van_segment", "Construction")
        distanceDecay = dict(
            ((int(row['van_segment']), str(row['parameter'])), float(row['value']))
            for row in pd.read_csv(varDict['SERVICE_DISTANCEDECAY'], sep='\t').to_dict('records')
        )
        alphaService = distanceDecay[(id_service, 'alpha')]
        betaService = distanceDecay[(id_service, 'beta')]
        alphaConstruction = distanceDecay[(id_construction, 'alpha')]
        betaConstruction = distanceDecay[(id_construction, 'beta')]

        # Import zone shapefile
        zones = read_shape(varDict['ZONES'])
        nInternalZones = zones.shape[0]

        superCoordinates = pd.read_csv(varDict['SUP_COORDINATES_ID'], sep='\t')
        nExternalZones = superCoordinates.shape[0]

        zoneDict = {}
        for i in range(nInternalZones):
            zoneDict[i] = zones.at[i, 'AREANR']
        for i in range(nExternalZones):
            zoneDict[nInternalZones + i] = 99999901 + i
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        # Import socio economic data
        segs = pd.read_csv(varDict['SEGS'], sep=',')
        segs = segs.sort_values('zone')
        segs.index = segs['zone']

        # Import regression coefficients
        regrCoeffs = pd.read_csv(varDict['SERVICE_PA'], sep=',', index_col=[0])

        # Which MRDH zones form which COROP region
        MRDHtoCOROP = pd.read_csv(varDict['MRDH_TO_COROP'], sep='\t')
        zonesCOROP = list(MRDHtoCOROP['zone_corop'].unique())

        MRDHwithinCOROP = {}
        for i in zonesCOROP:
            MRDHwithinCOROP[i] = np.array(MRDHtoCOROP.loc[MRDHtoCOROP['zone_corop'] == i, 'zone_mrdh'])

        # Parcel nodes
        parcelNodes = read_shape(varDict['PARCELNODES'])

        if root is not None:
            root.progressBar['value'] = 0.5

        # Skim with travel times and distances
        skimTravTime, skimDistance, nZones = get_skims(varDict)

        if (nExternalZones + nInternalZones) != nZones:
            raise Exception(
                f"The number of internal zones in the ZONES file ({nInternalZones}) " +
                f"and the number of external zones in the SUP_COORDINATES_ID file ({nExternalZones}) "
                f"don't match with the number of zones in the skim files: ({nZones})."
            )

        if root is not None:
            root.progressBar['value'] = 2.0

        # ------------------- Productions and attractions ---------------------

        logger.debug("\tCalculating productions and attractions...")

        # Surface of DCs per zone
        surfaceDC = np.array(zones['SurfaceDC'])

        # Surface of parcel nodes per zone
        surfaceParcelDepot = np.zeros(nZones, dtype=float)
        for i in range(len(parcelNodes)):
            zone = parcelNodes.at[i, 'AREANR']
            surface = parcelNodes.at[i, 'Surface']
            surfaceParcelDepot[invZoneDict[int(zone)]] += surface

        # Jobs per sector
        jobs = {}
        for sector in dims.employment_sector.keys():
            sector_name = dims.employment_sector[sector]['Comment']
            jobs[sector_name] = np.zeros(nZones, dtype=float)
            jobs[sector_name][:nInternalZones] = (
                segs.loc[[zoneDict[x] for x in range(nInternalZones)], sector_name])

            for i in range(len(zonesCOROP)):
                jobs[sector_name][nInternalZones + i] = (
                    np.sum(segs.loc[MRDHwithinCOROP[zonesCOROP[i]], sector_name]))

        population = np.array(segs.loc[zones['AREANR'].values, '2: inwoners'])

        # Determine produced trips per zone for service and construction
        prodService = np.zeros(nZones, dtype=int)
        prodConstruction = np.zeros(nZones, dtype=int)

        # For the zones in the study area (ZH)
        for i in range(nInternalZones):
            prodService[i] = (
                regrCoeffs.at['Service', 'DC_OPP'    ] * surfaceDC[i] +
                regrCoeffs.at['Service', 'PARCEL_OPP'] * surfaceParcelDepot[i] +
                regrCoeffs.at['Service', 'LANDBOUW'  ] * jobs['LANDBOUW' ][i] +
                regrCoeffs.at['Service', 'INDUSTRIE' ] * jobs['INDUSTRIE'][i] +
                regrCoeffs.at['Service', 'DETAIL'    ] * jobs['DETAIL'   ][i] +
                regrCoeffs.at['Service', 'DIENSTEN'  ] * jobs['DIENSTEN' ][i] +
                regrCoeffs.at['Service', 'OVERIG'    ] * jobs['OVERIG'   ][i] +
                regrCoeffs.at['Service', 'INWONERS'  ] * population[i])

            prodConstruction[i] = (
                regrCoeffs.at['Construction', 'DC_OPP'    ] * surfaceDC[i] +
                regrCoeffs.at['Construction', 'PARCEL_OPP'] * surfaceParcelDepot[i] +
                regrCoeffs.at['Construction', 'LANDBOUW'  ] * jobs['LANDBOUW' ][i] +
                regrCoeffs.at['Construction', 'INDUSTRIE' ] * jobs['INDUSTRIE'][i] +
                regrCoeffs.at['Construction', 'DETAIL'    ] * jobs['DETAIL'   ][i] +
                regrCoeffs.at['Construction', 'DIENSTEN'  ] * jobs['DIENSTEN' ][i] +
                regrCoeffs.at['Construction', 'OVERIG'    ] * jobs['OVERIG'   ][i] +
                regrCoeffs.at['Construction', 'INWONERS'  ] * population[i])

        if root is not None:
            root.progressBar['value'] = 3.0

        # For the external zones
        for i in range(len(zonesCOROP)):

            tmpSegsRows = MRDHwithinCOROP[zonesCOROP[i]]

            prodService[nInternalZones + i] = (
                regrCoeffs.at['Service','PARCEL_OPP'] * surfaceParcelDepot[nInternalZones + i] +
                regrCoeffs.at['Service','LANDBOUW'  ] * np.sum(segs.loc[tmpSegsRows, 'LANDBOUW'   ]) +
                regrCoeffs.at['Service','INDUSTRIE' ] * np.sum(segs.loc[tmpSegsRows, 'INDUSTRIE'  ]) +
                regrCoeffs.at['Service','DETAIL'    ] * np.sum(segs.loc[tmpSegsRows, 'DETAIL'     ]) +
                regrCoeffs.at['Service','DIENSTEN'  ] * np.sum(segs.loc[tmpSegsRows, 'DIENSTEN'   ]) +
                regrCoeffs.at['Service','OVERIG'    ] * np.sum(segs.loc[tmpSegsRows, 'OVERIG'     ]) +
                regrCoeffs.at['Service','INWONERS'  ] * np.sum(segs.loc[tmpSegsRows, '2: inwoners']))

            prodConstruction[nInternalZones + i] = (
                regrCoeffs.at['Construction','PARCEL_OPP'] * surfaceParcelDepot[nInternalZones + i] +
                regrCoeffs.at['Construction','LANDBOUW'  ] * np.sum(segs.loc[tmpSegsRows, 'LANDBOUW'   ]) +
                regrCoeffs.at['Construction','INDUSTRIE' ] * np.sum(segs.loc[tmpSegsRows, 'INDUSTRIE'  ]) +
                regrCoeffs.at['Construction','DETAIL'    ] * np.sum(segs.loc[tmpSegsRows, 'DETAIL'     ]) +
                regrCoeffs.at['Construction','DIENSTEN'  ] * np.sum(segs.loc[tmpSegsRows, 'DIENSTEN'   ]) +
                regrCoeffs.at['Construction','OVERIG'    ] * np.sum(segs.loc[tmpSegsRows, 'OVERIG'     ]) +
                regrCoeffs.at['Construction','INWONERS'  ] * np.sum(segs.loc[tmpSegsRows, '2: inwoners']))

        if root is not None:
            root.progressBar['value'] = 4.0

        # ---------------------- Trip distribution ----------------------------

        logger.debug("\tTrip distribution...")

        logger.debug("\t\tConstructing initial matrix...")

        # Travel costs
        skimCost = (
            costParams["cost_per_hour"] * (skimTravTime / 3600) +
            costParams["cost_per_km"] * (skimDistance / 1000))
        skimCost = skimCost.reshape(nZones, nZones)

        # Travel resistance
        matrixService = 100 / (1 + (np.exp(alphaService) * skimCost ** betaService))
        matrixConstruction = 100 / (1 + (np.exp(alphaConstruction) * skimCost ** betaConstruction))

        # Multiply by productions and then attractions to get start matrix
        # (assumed: productions = attractions)
        matrixService *= np.tile(prodService, (len(prodService), 1))
        matrixConstruction *= np.tile(prodConstruction, (len(prodConstruction), 1))

        matrixService *= np.tile(prodService, (len(prodService), 1)).transpose()
        matrixConstruction *= np.tile(prodConstruction, (len(prodConstruction), 1)).transpose()

        if root is not None:
            root.progressBar['value'] = 6.0

        logger.debug("\t\tDistributing service trips...")

        itern = 0
        conv = tolerance + 100

        while (itern < maxIter) and (conv > tolerance):

            itern += 1

            logger.debug(f"\t\t\tIteration {itern}")

            maxColScaleFac = 0
            totalRows = np.sum(matrixService, axis=0)

            for j in range(nZones):
                total = totalRows[j]

                if total > 0:
                    scaleFacCol = prodService[j] / total

                    if abs(scaleFacCol) > abs(maxColScaleFac):
                        maxColScaleFac = scaleFacCol

                    matrixService[:, j] *= scaleFacCol

            maxRowScaleFac = 0
            totalCols = np.sum(matrixService, axis=1)

            for i in range(nZones):
                total = totalCols[i]

                if total > 0:
                    scaleFacRow = prodService[i] / total

                    if abs(scaleFacRow) > abs(maxRowScaleFac):
                        maxRowScaleFac = scaleFacRow

                    matrixService[i, :] *= scaleFacRow

            conv = max(abs(maxColScaleFac - 1), abs(maxRowScaleFac - 1))

            logger.debug(f"\t\t\tConvergence {round(conv, 4)}")

            if root is not None:
                root.progressBar['value'] = (
                    6.0 +
                    (46.0 - 6.0) * itern / maxIter)

        if conv > tolerance:
            logger.warning(
                "Convergence is lower than the tolerance criterion, more iterations might be needed.")

        logger.debug("\t\tDistributing construction trips...")

        itern = 0
        conv = tolerance + 100

        while (itern < maxIter) and (conv > tolerance):

            itern += 1

            logger.debug(f"\t\t\tIteration {itern}")

            maxColScaleFac = 0
            totalRows = np.sum(matrixConstruction, axis=0)

            for j in range(nZones):
                total = totalRows[j]

                if total > 0:
                    scaleFacCol = prodConstruction[j] / total

                    if abs(scaleFacCol) > abs(maxColScaleFac):
                        maxColScaleFac = scaleFacCol

                    matrixConstruction[:, j] *= scaleFacCol

            maxRowScaleFac = 0
            totalCols = np.sum(matrixConstruction, axis=1)

            for i in range(nZones):
                total = totalCols[i]

                if total > 0:
                    scaleFacRow = prodConstruction[i] / total

                    if abs(scaleFacRow) > abs(maxRowScaleFac):
                        maxRowScaleFac = scaleFacRow

                    matrixConstruction[i, :] *= scaleFacRow

            conv = max(abs(maxColScaleFac - 1), abs(maxRowScaleFac - 1))

            logger.debug(f"\t\t\tConvergence {round(conv, 4)}")

            if root is not None:
                root.progressBar['value'] = (
                    46.0 +
                    (86.0 - 46.0) * itern / maxIter)

        if conv > tolerance:
            logger.warning(
                "Convergence is lower than the tolerance criterion, more iterations might be needed.")

        # Trips van extern naar extern eruithalen
        # (voor consistentie met vracht)
        for i in range(nExternalZones):
            matrixService[nInternalZones + i, nInternalZones:] = 0
            matrixConstruction[nInternalZones + i, nInternalZones:] = 0

        # Van jaar naar dag brengen
        matrixService = np.round(matrixService.flatten() / varDict['YEARFACTOR'], 3)
        matrixConstruction = np.round(matrixConstruction.flatten() / varDict['YEARFACTOR'], 3)

        # --------------------- Writing OD trip matrices ----------------------

        logger.debug("\tWriting trip matrices...")

        write_mtx(varDict['OUTPUTFOLDER'] + 'TripsVanService.mtx', matrixService, nZones)

        if root is not None:
            root.progressBar['value'] = 93.0

        write_mtx(varDict['OUTPUTFOLDER'] + 'TripsVanConstruction.mtx', matrixConstruction, nZones)

        if root is not None:
            root.progressBar['value'] = 100.0

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
