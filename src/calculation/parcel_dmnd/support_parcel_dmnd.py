import numpy as np
import pandas as pd
from typing import Dict

from calculation.common.dimensions import ModelDimensions


def get_cum_shares_vt_ucc(varDict: Dict[str, str], dims: ModelDimensions) -> Dict[int, np.ndarray]:
    """
    Returns the cumulative shares of vehicle types for the switch to UCCs.
    """
    cumSharesVehUCC = np.zeros(len(dims.vehicle_type))

    id_parcel_segment = dims.get_id_from_label("logistic_segment", "Parcel (consolidated flows)")

    for row in pd.read_csv(varDict["ZEZ_SCENARIO"], sep='\t').to_dict('records'):
        if int(row['logistic_segment']) == id_parcel_segment:
            cumSharesVehUCC[int(row['vehicle_type'])] += float(row['share_vehicle'])

    if np.sum(cumSharesVehUCC) != 0:
        cumSharesVehUCC = np.cumsum(cumSharesVehUCC) / np.sum(cumSharesVehUCC)

    return cumSharesVehUCC


def aggregate_parcels(parcels: pd.DataFrame, varDict: Dict[str, str]) -> pd.DataFrame:
    """Aggregates parcels such that each row is a combination of parcels instead of one parcel."""
    if varDict['LABEL'] == 'UCC':
        parcelsAggr: pd.DataFrame = pd.pivot_table(
            parcels,
            values=['Parcel_ID'],
            index=[
                "DepotNumber", 'CEP', 'D_zone', 'O_zone', 'VEHTYPE',
                'FROM_UCC', 'TO_UCC'],
            aggfunc={
                'DepotNumber': np.mean,
                'CEP': 'first',
                'O_zone': np.mean,
                'D_zone': np.mean,
                'Parcel_ID': 'count',
                'VEHTYPE': np.mean,
                'FROM_UCC': np.mean,
                'TO_UCC': np.mean})
        parcelsAggr = parcelsAggr.rename(columns={'Parcel_ID':'Parcels'})
        parcelsAggr = parcelsAggr.set_index(np.arange(len(parcelsAggr)))
        parcelsAggr = parcelsAggr.reindex(
            columns=[
                'O_zone', 'D_zone', 'Parcels', 'DepotNumber', 'CEP', 'VEHTYPE',
                'FROM_UCC', 'TO_UCC'])
        parcelsAggr = parcelsAggr.astype({
            'DepotNumber': int,
            'O_zone': int,
            'D_zone': int,
            'Parcels': int,
            'VEHTYPE': int,
            'FROM_UCC': int,
            'TO_UCC': int})

    elif varDict['LABEL'].startswith('MIC'):
        parcelsAggr = pd.pivot_table(
            parcels,
            values=['Parcel_ID'],
            index=[
                "DepotNumber", 'CEP', 'D_zone', 'O_zone', 'VEHTYPE',
                'FROM_MH', 'TO_MH'],
            aggfunc={
                'DepotNumber': np.mean,
                'CEP': 'first',
                'O_zone': np.mean,
                'D_zone': np.mean,
                'Parcel_ID': 'count',
                'VEHTYPE': np.mean,
                'FROM_MH': np.mean,
                'TO_MH': np.mean})
        parcelsAggr = parcelsAggr.rename(
            columns={'Parcel_ID': 'Parcels'})
        parcelsAggr = parcelsAggr.set_index(np.arange(len(parcelsAggr)))
        parcelsAggr = parcelsAggr.reindex(
            columns=[
                'O_zone', 'D_zone', 'Parcels', 'DepotNumber', 'CEP', 'VEHTYPE',
                'FROM_MH', 'TO_MH'])
        parcelsAggr = parcelsAggr.astype({
            'DepotNumber': int,
            'O_zone': int,
            'D_zone': int,
            'Parcels': int,
            'VEHTYPE': int,
            'FROM_MH': int,
            'TO_MH': int})

        parcelsPerCEP = pd.pivot_table(
            parcels[parcels['FROM_MH'] > 0],
            values=['Parcel_ID'],
            index=['CEP', 'FROM_MH', 'O_zone'],
            aggfunc={'Parcel_ID': 'count'})
        parcelsPerCEP.columns = ['ParcelCount']

        parcelsPerCEP.to_csv(
            varDict['OUTPUTFOLDER'] + "ParcelsPerMicrohubCEP_MIC.csv",
            index=True)

    else:
        parcelsAggr = pd.pivot_table(
            parcels,
            values=['Parcel_ID'],
            index=["DepotNumber", 'CEP', 'D_zone', 'O_zone'],
            aggfunc={
                'DepotNumber': np.mean,
                'CEP': 'first',
                'O_zone': np.mean,
                'D_zone': np.mean,
                'Parcel_ID': 'count'})
        parcelsAggr = parcelsAggr.rename(columns={'Parcel_ID': 'Parcels'})
        parcelsAggr = parcelsAggr.set_index(np.arange(len(parcelsAggr)))
        parcelsAggr = parcelsAggr.reindex(
            columns=['O_zone', 'D_zone', 'Parcels', 'DepotNumber', 'CEP'])
        parcelsAggr = parcelsAggr.astype({
            'DepotNumber': int,
            'O_zone': int,
            'D_zone': int,
            'Parcels': int})
        
    return parcelsAggr


def write_parcels_to_geojson(
    parcelsAggr: pd.DataFrame,
    parcelNodes: pd.DataFrame,
    zonesX: Dict[int, float],
    zonesY: Dict[int, float],
    varDict: Dict[str, str],
) -> None:
    """Writes the parcels as a geojson file with coordinates in the output folder."""
    # Initialize arrays with coordinates
    Ax = np.zeros(len(parcelsAggr), dtype=int)
    Ay = np.zeros(len(parcelsAggr), dtype=int)
    Bx = np.zeros(len(parcelsAggr), dtype=int)
    By = np.zeros(len(parcelsAggr), dtype=int)

    # Determine coordinates of LineString for each trip
    depotIDs = np.array(parcelsAggr['DepotNumber'])
    for i in parcelsAggr.index:
        if varDict['LABEL'] == 'UCC':
            if parcelsAggr.at[i, 'FROM_UCC'] == 1:
                Ax[i] = zonesX[parcelsAggr['O_zone'][i]]
                Ay[i] = zonesY[parcelsAggr['O_zone'][i]]
                Bx[i] = zonesX[parcelsAggr['D_zone'][i]]
                By[i] = zonesY[parcelsAggr['D_zone'][i]]
            else:
                Ax[i] = parcelNodes['X'][depotIDs[i]]
                Ay[i] = parcelNodes['Y'][depotIDs[i]]
                Bx[i] = zonesX[parcelsAggr['D_zone'][i]]
                By[i] = zonesY[parcelsAggr['D_zone'][i]]
        elif varDict['LABEL'][:3] == 'MIC':
            Ax[i] = zonesX[parcelsAggr['O_zone'][i]]
            Ay[i] = zonesY[parcelsAggr['O_zone'][i]]
            Bx[i] = zonesX[parcelsAggr['D_zone'][i]]
            By[i] = zonesY[parcelsAggr['D_zone'][i]]
        else:
            Ax[i] = parcelNodes['X'][depotIDs[i]]
            Ay[i] = parcelNodes['Y'][depotIDs[i]]
            Bx[i] = zonesX[parcelsAggr['D_zone'][i]]
            By[i] = zonesY[parcelsAggr['D_zone'][i]]

    Ax = np.array(Ax, dtype=str)
    Ay = np.array(Ay, dtype=str)
    Bx = np.array(Bx, dtype=str)
    By = np.array(By, dtype=str)
    nRecords = parcelsAggr.shape[0]

    with open(f"{varDict['OUTPUTFOLDER']}ParcelDemand_{varDict['LABEL']}.geojson", 'w') as geoFile:
        geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')

        for i in range(nRecords - 1):
            geoFile.write(
                '{ "type": "Feature", "properties": ' +
                str(parcelsAggr.loc[i,:].to_dict()).replace("'",'"') +
                ', "geometry": { "type": "LineString", "coordinates": [ [ ' +
                Ax[i] + ', ' + Ay[i] + ' ], [ ' +
                Bx[i] + ', ' + By[i] + ' ] ] } },\n'
            )

            if i % int(nRecords / 10) == 0:
                print('\t' + str(round(i / nRecords * 100, 1)) + '%', end='\r')

        # Bij de laatste feature moet er geen komma aan het einde
        i += 1
        geoFile.write(
            '{ "type": "Feature", "properties": ' +
            str(parcelsAggr.loc[i,:].to_dict()).replace("'",'"') +
            ', "geometry": { "type": "LineString", "coordinates": [ [ ' +
            Ax[i] + ', ' + Ay[i] + ' ], [ ' +
            Bx[i] + ', ' + By[i] + ' ] ] } }\n'
        )
        geoFile.write(']\n')
        geoFile.write('}')
