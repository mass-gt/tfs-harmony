import logging
import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Any, Dict, List, Union

from calculation.common.dimensions import ModelDimensions

logger = logging.getLogger("tfs")


def get_shapely_zones(
    zonesGeometry: List[Dict[str, Any]]
) -> List[Union[Polygon, MultiPolygon]]:
    """
    Converts zonesGeometry into a list of shapely objects.
    """
    shapelyZones = []

    for row in zonesGeometry:
        if row['type'] == 'MultiPolygon':
            tmp = []

            for i in range(len(row['coordinates'])):
                if len(row['coordinates'][i]) > 1:
                    tmp.append(Polygon(
                        row['coordinates'][i][0],
                        [
                            row['coordinates'][i][j]
                            for j in range(1, len(row['coordinates'][i]))
                        ]
                    ))

                else:
                    tmp.append(Polygon(row['coordinates'][i][0]))

            shapelyZones.append(MultiPolygon(tmp))

        elif row['type'] == 'Polygon':
            if len(row['coordinates']) > 1:
                shapelyZones.append(Polygon(
                    row['coordinates'][0],
                    [
                        row['coordinates'][i]
                        for i in range(1, len(row['coordinates']))
                    ]
                ))

            else:
                shapelyZones.append(Polygon(row['coordinates'][0]))

        else:
            logger.warning(
                'Object types other than Polygon or MultiPolygon found in zones shape!'
            )

    return shapelyZones


def read_segs(
    zones: pd.DataFrame,
    varDict: Dict[str, str],
    dims: ModelDimensions,
):
    """
    Reads the socio-economic data and adjusts it for DCs and transshipment terminals.
    """
    # Zonal / Socioeconomic data
    segs = pd.read_csv(varDict['SEGS'], sep=',')
    segs.index = segs['zone']
    segs = segs[[row['Comment'] for row in dims.employment_sector.values()]].round()

    # Adjust zonal employment data for distribution centres
    # and transshipment terminals in zones with DC,
    # the jobs in DC are known and subtracted from industry jobs
    dcs_wp = pd.read_csv(varDict['DISTRIBUTIECENTRA'], sep='\t').pivot_table(
        index='zone_mrdh', values='employment', aggfunc='sum'
    )

    # Add jobs in DC to segs
    segs['WP'] = np.zeros(len(segs))
    for x in dcs_wp.index:
        segs['WP'][x] = dcs_wp['employment'][x]

    # Update number of industrie jobs and set to 0
    # in case it got smaller than 0
    segs['INDUSTRIE_voorCorr'] = segs['INDUSTRIE'].copy()
    segs['INDUSTRIE'] = segs['INDUSTRIE_voorCorr'] - segs['WP']
    segs['INDUSTRIE'][segs['INDUSTRIE'] < 0] = 0

    jobs_init = segs['INDUSTRIE_voorCorr'].sum()
    jobs_zonderDC = segs['INDUSTRIE'].sum()

    # In zones with transshipment terminals (lognode=1), remaining industry jobs are set to zero
    segs['LOGNODE'] = zones['LOGNODE'].copy()
    segs.loc[
        segs['LOGNODE'] == dims.get_id_from_label("logistic_node", "Transshipment terminal"),
        'INDUSTRIE'] = 0

    logger.debug('\tAdjusting zonal data for jobs in DCs and transshipment terminals to avoid double counting')
    logger.debug(f"\t\tINDUSTRIE jobs before adjustment: {jobs_init}")
    logger.debug(f'\t\tINDUSTRIE jobs w/o DC:{jobs_zonderDC}')
    logger.debug(f"\t\tINDUSTRIE jobs in DCs (input): {dcs_wp['employment'].sum()}")
    logger.debug(f"\t\tINDUSTRIE jobs removed for DC: {jobs_init - jobs_zonderDC}")
    logger.debug(f"\t\tINDUSTRIE jobs removed for TT: {jobs_zonderDC - segs['INDUSTRIE'].sum()}")
    logger.debug(f"\t\tINDUSTRIE jobs w/o DC and TT: {segs['INDUSTRIE'].sum()}")

    # Remove columns that are not needed for synthesis
    return segs.drop(columns=['INDUSTRIE_voorCorr', 'WP', 'LOGNODE'])


def validation_checks(
    firms_df: pd.DataFrame,
    segs: pd.DataFrame,
    maxZoneNumberZH: int,
    varDict: Dict[str, str],
) -> None:
    """
    Performs various validation checks on the synthesized firms.
    Then prints these into the command prompt or writes these to CSV files in the output folder.
    """
    unique_emplSize, counts_emplSize = np.unique(firms_df['SIZE'].values, return_counts=True)
    emplSize_counts = np.asarray(
        (
            unique_emplSize,
            counts_emplSize,
            np.round(counts_emplSize / np.sum(counts_emplSize), 5)
        )).T
    logger.debug(np.sum(counts_emplSize))
    logger.debug(emplSize_counts)

    unique_sector, counts_sector = np.unique(
        firms_df['employment_sector'].values,
        return_counts=True)
    sector_counts = np.asarray(
        (
            unique_sector,
            counts_sector,
            np.round(counts_sector / np.sum(counts_sector), 5)
        )).T
    logger.debug(sector_counts)

    zones_jobs = firms_df.pivot_table(
        index='zone_mrdh',
        columns='employment_sector',
        values='employment',
        aggfunc='sum')
    zones_jobs['TOTAAL'] = zones_jobs.sum(axis=1, skipna=True)
    zones_jobs = zones_jobs.fillna(0)

    # Eerge with input segs to better enable comparison
    emplIn = segs.copy()
    emplIn['TOTAAL'] = emplIn.sum(axis=1, skipna=True)
    emplIn = emplIn[emplIn.index <= maxZoneNumberZH].copy()

    emplAbsDiff = zones_jobs - emplIn
    emplAbsDiff.columns = [
        'DETAIL_absDiff',
        'DIENSTEN_absDiff',
        'INDUSTRIE_absDiff',
        'LANDBOUW_absDiff',
        'OVERHEID_absDiff',
        'OVERIG_absDiff',
        'TOTAAL_absDiff']

    emplRelDiff = (zones_jobs - emplIn) / emplIn
    emplRelDiff.columns = [
        'DETAIL_relDiff',
        'DIENSTEN_relDiff',
        'INDUSTRIE_relDiff',
        'LANDBOUW_relDiff',
        'OVERHEID_relDiff',
        'OVERIG_relDiff',
        'TOTAAL_relDiff']

    df_out = emplIn.merge(
        zones_jobs.iloc[:, -7:],
        how='left',
        left_on=emplIn.index,
        right_index=True,
        suffixes=('_in', '_out'))
    df_out = df_out.merge(
        emplAbsDiff,
        how='left',
        left_on=emplIn.index,
        right_index=True,
        suffixes=('', '_absDiff'))
    df_out = df_out.merge(
        emplRelDiff,
        how='left',
        left_on=emplIn.index,
        right_index=True,
        suffixes=('', '_relDiff'))

    df_out = df_out.fillna(0)
    df_out.to_csv(f"{varDict['OUTPUTFOLDER']}SynthJobsPerZone.csv")

    # Make crosstab sector X firm size for comparison with input
    sectorsXsize_out = firms_df.pivot_table(
        index='employment_sector',
        columns='firm_size',
        values='zone_mrdh',
        aggfunc='count')
    sectorsXsize_out = sectorsXsize_out.fillna(0)
    sectorsXsize_out.to_csv(f"{varDict['OUTPUTFOLDER']}SynthFirms_sectorsXsize.csv")


def add_firm_coordinates(
    firms_df: pd.DataFrame,
    shapelyZones: List[Union[Polygon, MultiPolygon]],
    invZoneDict: Dict[int, int],
    root: Any,
    nTriesAllowed: int = 500,
) -> pd.DataFrame:
    """
    Adds the fields 'X' and 'Y' to 'firms_df'.
    """
    firmZones = np.array(firms_df['zone_mrdh'], dtype=int)
    nFirms = firms_df.shape[0]

    # Initialize two dictionaries in which we'll store the
    # drawn x- and y-coordinate for each firm
    X = {}
    Y = {}

    nTimesNumberOfTriesReached = 0
    centroidZones = []

    for i in range(nFirms):
        # Get the zone polygon and its boundaries
        polygon = shapelyZones[invZoneDict[firmZones[i]]]
        minX, minY, maxX, maxY = polygon.bounds

        pointInPolygon = False
        numberOfTries = 0

        while (not pointInPolygon) and (numberOfTries <= nTriesAllowed):
            # Generate a random point within the zone boundaries
            x = minX + (maxX - minX) * np.random.rand()
            y = minY + (maxY - minY) * np.random.rand()
            point = Point(x, y)

            # Check if the random point is actually contained by
            # the zone polygon
            pointInPolygon = polygon.contains(point)
            numberOfTries += 1

        # If we haven't generated a point contained by the
        # zone polygon yet after {nTriesAllowed} times trying,
        # then we just take the centroid of the zone polygon
        if not pointInPolygon:
            x, y = polygon.centroid.coords[0]
            nTimesNumberOfTriesReached += 1
            centroidZones.append(firmZones[i])

        X[i], Y[i] = x, y

        if i % int(nFirms / 100) == 0:
            print(f"\t{round(i / nFirms * 100, 1)}%", end='\r')

            if root is not None:
                root.progressBar['value'] = 65 + (95 - 65) * i / nFirms

    firms_df['x_coord'] = X.values()
    firms_df['y_coord'] = Y.values()

    if nTimesNumberOfTriesReached > 0:
        logger.debug(f"\tHad to use the centroid coordinates for  {nTimesNumberOfTriesReached} of {nFirms} firms.")
        logger.debug(f"\t(This is the case for the following zone(s): {np.unique(centroidZones)}")

    return firms_df
