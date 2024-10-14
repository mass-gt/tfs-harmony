import logging
import numpy as np
import pandas as pd
import sys
import traceback

from typing import Any, Dict

from calculation.common.dimensions import ModelDimensions
from calculation.common.io import read_shape, get_seeds
from calculation.common.vrt import draw_choice_mcs
from .support_fs import validation_checks, add_firm_coordinates, get_shapely_zones, read_segs

logger = logging.getLogger("tfs")


def actually_run_module(
    root: Any,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Performs the calculations of the Firm Synthesis module.
    """
    try:

        doValidationChecks = False
        minEmplLevelOutput = 3

        # ------------------------- Import data -------------------------------

        logger.debug('\tImporting data...')
        if root is not None:
            root.update_statusbar('Importing data...')

        seeds = get_seeds(varDict)

        # Shapefile of study area
        zones, zonesGeometry = read_shape(varDict['ZONES'], returnGeometry=True)
        zones = zones.sort_values('AREANR')
        zonesGeometry = [zonesGeometry[i] for i in zones.index]
        zones.index = zones['AREANR']

        maxZoneNumberZH = int(zones['AREANR'].max())
        nInternalZones = len(zones)

        if root is not None:
            root.progressBar['value'] = 2

        shapelyZones = get_shapely_zones(zonesGeometry)

        if root is not None:
            root.progressBar['value'] = 3

        segs = read_segs(zones, varDict, dims)

        if root is not None:
            root.progressBar['value'] = 4

        # Read firm input data
        firmSizePerSector = pd.pivot_table(
            pd.read_csv(varDict['FIRMSIZE'], sep='\t'),
            index='employment_sector',
            columns='firm_size',
            values='value',
            aggfunc=sum,
        )

        firmSizePerSectorCumProb = dict(
            (
                sector,
                np.cumsum(firmSizePerSector.loc[sector, :].values) /
                np.sum(firmSizePerSector.loc[sector, :].values)
            )
            for sector in dims.employment_sector.keys()
        )

        firmSizeMapping = dict(
            (row['ID'], (row['LowerBound'], row['UpperBound']))
            for row in dims.firm_size.values()
        )

        if root is not None:
            root.progressBar['value'] = 5

        # ------------------------ Synthesize firms ---------------------------
        # Turn number of employees per zone into firms
        # (drawn randomly from sectorXsize table)
        logger.debug('\tSynthesizing firms...')

        if root is not None:
            root.update_statusbar('Synthesizing firms...')

        firmZones = []
        firmSectors = []
        firmSizes = []
        firmEmpl = []
        zoneEmpl = []  # Number of employees in current zone and sector

        id_max_firm_size = dims.get_id_from_label("firm_size", "Groot")

        # Select zone
        for zone in segs.index:

            if zone > maxZoneNumberZH:
                continue

            assigned_jobs = 0

            # Select industry sector I
            for sector in dims.employment_sector.keys():
                sector_label = dims.employment_sector[sector]['Comment']

                jobs_to_assign = float(segs.loc[zone][sector_label])
                tmp_id = 0

                while jobs_to_assign > 0:
                    firm_seed = seeds['firm_size'] + zone * 1000 + sector * 100 + tmp_id

                    # Draw a firm for current sector
                    firm_size = draw_choice_mcs(firmSizePerSectorCumProb[sector], firm_seed)

                    # Determine size of the firm.
                    # If we draw a firm from the largest category,
                    # draw from triangular distribution
                    if firm_size == id_max_firm_size:
                        low = firmSizeMapping[firm_size][0]
                        mode = max(low, 0.5 * jobs_to_assign)
                        right = max(150, 1.5 * jobs_to_assign)

                        np.random.seed(firm_size)
                        size = int(round(np.random.triangular(low, mode, right, size=1)[0]))

                    # If the firm is from a closed size category,
                    # draw size from uniform distribution
                    else:
                        low, high = firmSizeMapping[firm_size]

                        np.random.seed(firm_size)
                        size = np.random.randint(low, high)

                    # Check if size fits in zone.
                    # If firm is not way too large, accept it and add it to list of firms
                    size = size if size <= 1.3 * jobs_to_assign else jobs_to_assign

                    firmZones.append(zone)
                    firmSectors.append(sector)
                    firmSizes.append(firm_size)
                    firmEmpl.append(size)
                    zoneEmpl.append(assigned_jobs)

                    jobs_to_assign -= size
                    tmp_id += 1

            if (zone - 1) % int((maxZoneNumberZH - 1) / 100) == 0:
                print('\t' + str(round((zone - 1) / (maxZoneNumberZH - 1) * 100, 1)) + '%', end='\r')

                if root is not None:
                    root.progressBar['value'] = 5 + (60 - 5) * (zone - 1) / (maxZoneNumberZH - 1)

        # Put results in a dataframe
        firms_df = pd.DataFrame(
            np.c_[
                np.arange(len(firmZones)), firmZones, firmSectors, firmSizes, firmEmpl
            ],
            columns=['firm_id', 'zone_mrdh', 'employment_sector', 'firm_size','employment']
        )

        if root is not None:
            root.progressBar['value'] = 65

        # ------------------- Optional validation checks ----------------------

        if doValidationChecks:
            validation_checks()

        # ----------------------- Draw coordinates  ---------------------------

        # Remove small firms
        firms_df = firms_df[firms_df['employment'] > minEmplLevelOutput]

        # New firm IDs after filtering
        firms_df['firm_id'] = np.arange(len(firms_df))
        firms_df.index = np.arange(len(firms_df))
        nFirms = len(firms_df)

        logger.debug(f'\tGenerating coordinates for {nFirms} firms...')

        if root is not None:
            root.update_statusbar(f'Generating coordinates...')

        # Dictionary with zone number (1-6625) to corresponding
        # zone number (1-7400)
        zoneDict = dict(np.transpose(np.vstack((
            np.arange(nInternalZones),
            zones['AREANR']))))
        zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
        invZoneDict = dict((v, k) for k, v in zoneDict.items())

        firms_df = add_firm_coordinates(firms_df, shapelyZones, invZoneDict, seeds, root)

        # ---------------------- Export firms to CSV  -------------------------

        logger.debug(f"\tExporting firms to: {varDict['OUTPUTFOLDER']}Firms.csv'")

        if root is not None:
            root.update_statusbar('Exporting firms...')

        firms_df.to_csv(varDict['OUTPUTFOLDER'] + 'Firms.csv', index=False)

        # ------------------------ End of module ------------------------------

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
