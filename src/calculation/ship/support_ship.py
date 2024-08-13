import numpy as np
import pandas as pd
import shapefile as shp

from itertools import product
from typing import Any, Dict, List, Tuple, Union

from calculation.common.dimensions import ModelDimensions
from calculation.common.vrt import draw_choice_mcs


def get_coeffs_distance_decay(varDict: Dict[str, str]) -> Tuple[float, float]:
    """Returns the distance decay parameters."""
    data = dict(
        (row['parameter'], row['value'])
        for row in pd.read_csv(varDict['FREIGHT_DISTANCEDECAY'], sep='\t').to_dict('records')
    )

    for parameter in ['alpha', 'beta']:
        if data.get(parameter) is None:
            raise Exception(f"Parameter '{parameter}' not found in '{varDict['FREIGHT_DISTANCEDECAY']}'.")

    alpha = data['alpha']
    beta =  data['beta']

    return alpha, beta


def get_nstr_to_ls(varDict: Dict[str, str], nNSTR: int, nLS) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the link between NSTR goods types and logistic segments as probability distributions."""
    data = np.zeros((nNSTR, nLS))

    for row in pd.read_csv(varDict['NSTR_TO_LS'], sep='\t').to_dict('records'):
        data[int(row['nstr']), int(row['logistic_segment'])] = float(row['tonnes'])

    nstrToLS = data.copy()
    for nstr in range(nNSTR):
        nstrToLS[nstr, :] = nstrToLS[nstr, :] / np.sum(nstrToLS[nstr, :])

    lsToNstr = data.copy()
    for ls in range(nLS):
        lsToNstr[:, ls] = lsToNstr[:, ls] / np.sum(lsToNstr[:, ls])

    return nstrToLS, lsToNstr


def get_commodity_matrix(
    nstrToLS: np.ndarray,
    nSuperZones: int,
    varDict: Dict[str, str],
    id_parcel_consolidated: int,
) -> pd.DataFrame:
    """
    Returns the commodity matrix with tonnes per day between the external zones and the study area.
    """
    nNSTR, nLS = nstrToLS.shape

    superComMatNSTR = pd.read_csv(varDict['OUTPUTFOLDER'] + 'CommodityMatrixNUTS3.csv', sep=',')
    superComMatNSTR['WeightDay'] = superComMatNSTR['TonnesYear'] / varDict['YEARFACTOR']
    superComMatNSTR = np.array(superComMatNSTR[['ORIG', 'DEST', 'NSTR', 'WeightDay']], dtype=object)

    NUTS3toAREANR = dict(
        (row['zone_nuts3'], row['zone_mrdh'])
        for row in pd.read_csv(varDict['NUTS3_TO_MRDH'], sep='\t').to_dict('records'))
    AREANRtoNUTS3 = {}
    for nuts3, areanr in NUTS3toAREANR.items():
        if areanr in AREANRtoNUTS3.keys():
            AREANRtoNUTS3[areanr].append(nuts3)
        else:
            AREANRtoNUTS3[areanr] = [nuts3]

    # Factors for increasing or decreasing total flow of certain logistics segments
    facLS = [1.0 for ls in range(nLS)]
    for ls in range(nLS):
        if varDict[f'FAC_LS{ls}'] != '':
            facLS[ls] = float(varDict[f'FAC_LS{ls}'])

    # Convert demand from NSTR to logistic segments
    nRows = (nSuperZones + 1) * (nSuperZones + 1) * nLS

    superComMat = np.zeros((nRows, 4))
    superComMat[:, 0] = (
        np.floor(np.arange(nRows) / (nSuperZones + 1) / nLS))
    superComMat[:, 1] = (
        np.floor(np.arange(nRows) / nLS) -
        superComMat[:, 0] * (nSuperZones + 1))

    for ls in range(nLS):
        superComMat[
            np.arange(ls, nRows, nLS), 2] = ls

    zonesWithKnownNUTS3 = set(list(AREANRtoNUTS3.keys()))

    for i, j in product(range(nSuperZones + 1), range(nSuperZones + 1)):
        if (99999900 + i) not in zonesWithKnownNUTS3:
            continue

        if (99999900 + j) not in zonesWithKnownNUTS3:
            continue

        origNUTS3 = AREANRtoNUTS3[99999900 + i]
        destNUTS3 = AREANRtoNUTS3[99999900 + j]

        weightDayNSTR = [0.0 for nstr in range(nNSTR)]

        for z in range(len(superComMatNSTR)):
            if superComMatNSTR[z, 0] in origNUTS3:
                if superComMatNSTR[z, 1] in destNUTS3:
                    tmpNSTR = superComMatNSTR[z, 2]
                    tmpWeight = superComMatNSTR[z, 3]
                    weightDayNSTR[tmpNSTR] += tmpWeight

        for nstr in range(nNSTR):

            if weightDayNSTR[nstr] > 0:

                for ls in range(nLS):
                    row = i * (nSuperZones + 1) * nLS + j * nLS + ls
                    superComMat[row, 3] += nstrToLS[nstr, ls] * float(weightDayNSTR[nstr])

                    # Apply growth to parcel market
                    if ls == id_parcel_consolidated:
                        growth = float(varDict['PARCELS_GROWTHFREIGHT'])
                        superComMat[row, 3] *= growth

                    # Apply increase/decrease factor for logistics segment
                    superComMat[row, 3] *= facLS[ls]

    superComMat = pd.DataFrame(superComMat, columns=['From', 'To', 'LS', 'WeightDay'])

    superComMat.loc[(superComMat['From'] != 0) & (superComMat['To'] != 0), 'WeightDay'] = 0

    return superComMat


def get_urban_density(
    segs: pd.DataFrame,
    zonesShape: pd.DataFrame,
    zoneDict: Dict[int, int],
    nInternalZones: int,
    nSuperZones: int,
) -> Dict[int, int]:
    """
    Returns a dictionary with an urban density measure for each zone.
    """
    urbanDensityCat: Dict[int, int] = {}

    for i in range(nInternalZones):
        tmpNumHouses = segs.at[zoneDict[i], '1: woningen']
        tmpNumJobs = segs.at[zoneDict[i], '9: arbeidspl_totaal']
        urbanDensity = (
            (tmpNumHouses + tmpNumJobs) /
            (zonesShape.at[zoneDict[i], 'area'] / 100000))

        if urbanDensity < 500:
            urbanDensityCat[zoneDict[i]] = 1
        elif urbanDensity < 1000:
            urbanDensityCat[zoneDict[i]] = 2
        elif urbanDensity < 1500:
            urbanDensityCat[zoneDict[i]] = 3
        elif urbanDensity < 2500:
            urbanDensityCat[zoneDict[i]] = 4
        else:
            urbanDensityCat[zoneDict[i]] = 5

    for i in range(nSuperZones):
        urbanDensityCat[99999901 + i] = 1

    return urbanDensityCat


def get_make_use_distribution(
    varDict: Dict[str, str],
    nNSTR: int,
    dims: ModelDimensions,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the probability of a sector making/using a product of an NSTR goods type.
    """
    nSector = len(dims.employment_sector)

    makeDistribution = np.zeros((nNSTR, nSector))
    useDistribution = np.zeros((nNSTR, nSector))

    for row in pd.read_csv(varDict['MAKE_DISTRIBUTION'], sep='\t').to_dict('records'):
        makeDistribution[int(row['nstr']), int(row['employment_sector'])] = float(row['share'])

    for row in pd.read_csv(varDict['USE_DISTRIBUTION'], sep='\t').to_dict('records'):
        useDistribution[int(row['nstr']), int(row['employment_sector'])] = float(row['share'])

    return makeDistribution, useDistribution


def get_flow_type_shares(varDict: Dict[str, str], dims: ModelDimensions) -> Dict[int, np.ndarray]:
    """Returns the distribution of flow types per logistic segment."""
    data = dict(
        ((int(row['logistic_segment']), int(row['flow_type'])), float(row['share']))
        for row in pd.read_csv(varDict['LOGISTIC_FLOWTYPES'], sep='\t').to_dict('records')
    )

    return dict(
        (ls, np.array([data.get((ls, ft), 0.0) for ft in dims.flow_type.keys()]))
        for ls in dims.logistic_segment.keys()
        if ls != dims.get_id_from_label("logistic_segment", "Parcel (deliveries)")
    )


def get_cum_shares_vt_ucc(varDict: Dict[str, str], dims: ModelDimensions) -> Dict[int, np.ndarray]:
    """
    Returns the cumulative shares of vehicle types for the switch to UCCs.
    """
    cumSharesVehUCC = dict(
        (ls, np.zeros(len(dims.vehicle_type)))
        for ls in dims.logistic_segment.keys()
    )

    for row in pd.read_csv(varDict["ZEZ_SCENARIO"], sep='\t').to_dict('records'):
        cumSharesVehUCC[int(row['logistic_segment'])][int(row['vehicle_type'])] += float(row['share_vehicle'])

    for ls in dims.logistic_segment.keys():
        if np.sum(cumSharesVehUCC[ls]) != 0:
            cumSharesVehUCC[ls] = np.cumsum(cumSharesVehUCC[ls]) / np.sum(cumSharesVehUCC[ls])

    return cumSharesVehUCC


def __try_float(number: Any, filename: str = '') -> Union[str, float]:
    try:
        return str(number) if '_' in str(number) else float(number)
    except ValueError:
        message = f"Could not convert '{number}' to float.\n"
        if filename != '':
            message = message +"This value was found in: " + filename + "."
        raise ValueError(message)


def get_params_time_of_day(
    varDict: Dict[str, str],
    nLS: int
) -> Tuple[
    List[Dict[str, Any]],
    List[List[List[int]]],
    List[List[int]],
    List[int],
]:
    """
    Returns the time-of-day choice model coefficients and information about its time intervals.
    """
    paramsTimeOfDay_df = pd.read_csv(varDict['PARAMS_TOD'], index_col=0)

    paramsTimeOfDay: List[Dict[str, Any]] = [
        dict(zip(
            paramsTimeOfDay_df.index,
            [__try_float(x, varDict['PARAMS_TOD']) for x in paramsTimeOfDay_df.loc[:, str(ls + 1)]]))
        for ls in range(nLS)]
        
    nTimeIntervals = len([
        x for x in paramsTimeOfDay[0].keys()
        if x.split('_')[0] == 'Interval'])

    timeIntervals = []
    timeIntervalsDur = []
    nTimeIntervalsLS = [0 for ls in range(nLS)]

    for ls in range(nLS):
        tmp = [
            str(paramsTimeOfDay[ls][f'Interval_{t+1}']).split('_')
            for t in range(nTimeIntervals)]
        timeIntervalsDur.append([])
        for t in range(nTimeIntervals):
            if len(tmp[t]) > 1:
                tmp[t] = [int(tmp[t][i]) for i in range(len(tmp[t]))]
                timeIntervalsDur[ls].append(int(tmp[t][1] - tmp[t][0]))
                nTimeIntervalsLS[ls] += 1
        timeIntervals.append(tmp)

    return paramsTimeOfDay, timeIntervals, timeIntervalsDur, nTimeIntervalsLS


def get_cep_shares(
    varDict: Dict[str, str],
    parcelNodes: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Returns the market shares of the parcel couriers."""
    cepShares = pd.read_csv(varDict['CEP_SHARES'], sep='\t')

    nDepots = np.array([
        np.sum(parcelNodes['CEP'] == str(cep)) for cep in cepShares['courier'].values])

    cepSharesTotal = cepShares['share_total'].values
    cepSharesInternal = cepShares['share_nl'].values

    # Remove share of couriers with 1 or 0 depots, we cannot form a trip between 2 depots there
    cepSharesTotal[nDepots <= 1] = 0
    cepSharesInternal[nDepots <= 1] = 0

    return (
        np.cumsum(cepSharesTotal) / np.sum(cepSharesTotal),
        np.cumsum(cepSharesInternal) / np.sum(cepSharesInternal),
        dict((i, row['courier']) for i, row in enumerate(cepShares.to_dict('records')))
    )


def generate_shipment_seeds(
    baseSeed: int,
    maxNumShipments: int,
    ls: int,
    nstr: int,
    ft: int,
    externalZone: int = 0
) -> np.ndarray:
    """Generates seeds for the synthesis of a set of shipments."""
    np.random.seed(baseSeed + externalZone * 1000 + ls * 100 + nstr * 10 + ft)
    np.random.seed(np.random.randint(10000000))

    return np.random.randint(low=0, high=10000000, size=maxNumShipments)


def draw_vehicle_type_and_shipment_size(
    paramsShipSizeVehType: Dict[Tuple[int, str], float],
    nstr: int,
    travTime: float,
    distance: float,
    isFromDC: bool,
    isToDC: bool,
    costPerHour: Dict[int, float],
    costPerKm: Dict[int, float],
    truckCapacities: Dict[int, float],
    nVT: int,
    absoluteShipmentSizes: np.ndarray,
    seed: int,
) -> Tuple[int, int]:
    """
    Calculates the utilities of the choice model for vehicle type and shipment size and then draws one choice.
    """
    nShipSize = len(absoluteShipmentSizes)

    # Selecting the logit parameters for this NSTR group
    B_TransportCosts = paramsShipSizeVehType[(nstr,'B_TransportCosts')]
    B_InventoryCosts = paramsShipSizeVehType[(nstr,'B_InventoryCosts')]
    B_FromDC = paramsShipSizeVehType[(nstr,'B_FromDC')]
    B_ToDC = paramsShipSizeVehType[(nstr,'B_ToDC')]
    B_LongHaul_TruckTrailer = paramsShipSizeVehType[(nstr,'B_LongHaul_TruckTrailer')]
    B_LongHaul_TractorTrailer = paramsShipSizeVehType[(nstr,'B_LongHaul_TractorTrailer')]
    ASC_VT = [paramsShipSizeVehType[(nstr, f'ASC_VT_{i+1}')] for i in range(nVT)]

    inventoryCosts = absoluteShipmentSizes
    longHaul = (distance > 100)

    # Determine the utility and probability for each alternative
    utilities = np.zeros(nVT * nShipSize)

    for ss, vt in product(range(nShipSize), range(nVT)):
        transportCosts = costPerHour[vt] * travTime + costPerKm[vt] * distance

        # Multiply transport costs by number of required vehicles
        transportCosts *= np.ceil(absoluteShipmentSizes[ss] / truckCapacities[vt])

        # Utility function
        index = ss * nVT + vt
        utilities[index] = (
            B_TransportCosts * transportCosts +
            B_InventoryCosts * inventoryCosts[ss] +
            B_FromDC * isFromDC * (vt == 0) +
            B_ToDC * isToDC * (vt in [3, 4, 5]) +
            B_LongHaul_TruckTrailer * longHaul * (vt in [3, 4]) +
            B_LongHaul_TractorTrailer * longHaul * (vt == 5) +
            ASC_VT[vt])

    probabilities = np.exp(utilities) / np.sum(np.exp(utilities))
    cumProbabilities = np.cumsum(probabilities)

    # Sample one choice based on the cumulative probability distribution
    ssvt = draw_choice_mcs(cumProbabilities, seed)

    # Deduce from this the chosen shipment size category and vehicle type
    ssChosen = int(np.floor(ssvt / nVT))
    vtChosen = int(ssvt - ssChosen * nVT)

    return ssChosen, vtChosen


def draw_delivery_times(
    origZone: Dict[int, int],
    destZone: Dict[int, int],
    logisticSegment: Dict[int, int],
    vehicleType: Dict[int, int],
    isTT: Dict[int, bool],
    isPC: Dict[int, bool],
    urbanDensityCat: Dict[int, int],
    zoneDict: Dict[int, int],
    nLS: int,
    varDict: Dict[str, str],
    dims: ModelDimensions,
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Reads the time-of-day coefficients and then uses these to draw the delivery time for each shipment.
    """
    paramsTimeOfDay, timeIntervals, timeIntervalsDur, nTimeIntervalsLS = get_params_time_of_day(varDict, nLS)

    id_small_truck = dims.get_id_from_label("vehicle_type", "Truck (small)")
    id_medium_truck = dims.get_id_from_label("vehicle_type", "Truck (medium)")
    ids_truck_trailer = [
        dims.get_id_from_label("vehicle_type", vt)
        for vt in ["Truck+trailer (small)", "Truck+trailer (large)"]]
    id_tractor_trailer = dims.get_id_from_label("vehicle_type", "Tractor+trailer")

    deliveryTimePeriod = {}
    lowerTOD = {}
    upperTOD = {}

    for i in range(len(origZone)):
        orig = zoneDict[origZone[i]]
        dest = zoneDict[destZone[i]]
        ls = logisticSegment[i]

        utilities = [
            paramsTimeOfDay[ls][f'ASC_{t+1}'] +
            paramsTimeOfDay[ls]['DurTimePeriod'] * np.log(2 * timeIntervalsDur[ls][t]) +
            paramsTimeOfDay[ls][f'ToTT_{t+1}']   * isTT[dest] * urbanDensityCat[dest] +
            paramsTimeOfDay[ls][f'ToPC_{t+1}']   * isPC[dest] * urbanDensityCat[dest] +
            paramsTimeOfDay[ls][f'FromTT_{t+1}'] * isTT[orig] * urbanDensityCat[orig] +
            paramsTimeOfDay[ls][f'FromPC_{t+1}'] * isPC[orig] * urbanDensityCat[orig] +
            paramsTimeOfDay[ls][f'VT_SmallTruck_{t+1}']     * (vehicleType[i] == id_small_truck) +
            paramsTimeOfDay[ls][f'VT_MediumTruck_{t+1}']    * (vehicleType[i] == id_medium_truck) +
            paramsTimeOfDay[ls][f'VT_TruckTrailer_{t+1}']   * (vehicleType[i] in ids_truck_trailer) +
            paramsTimeOfDay[ls][f'VT_TractorTrailer_{t+1}'] * (vehicleType[i] == id_tractor_trailer)
            for t in range(nTimeIntervalsLS[ls])
        ]

        probs = np.exp(np.array(utilities))
        probs /= np.sum(probs)
        cumProbs = np.cumsum(probs)
        cumProbs /= cumProbs[-1]

        deliveryTimePeriod[i] = draw_choice_mcs(cumProbs, seed + i)
        lowerTOD[i] = timeIntervals[ls][deliveryTimePeriod[i]][0]
        upperTOD[i] = timeIntervals[ls][deliveryTimePeriod[i]][1]

    return deliveryTimePeriod, lowerTOD, upperTOD


def get_zonal_prod_attr(
    shipments: pd.DataFrame,
    zoneDict: Dict[int, int],
    invZoneDict: Dict[int, int],
    nInternalZones: int,
    nSuperZones: int,
    nLS: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the tonnes produced and attracted by each zone in each logistic segment.
    """
    prodWeight = pd.pivot_table(shipments, values=['WEIGHT'], index=['ORIG', 'LS'], aggfunc=sum)
    attrWeight = pd.pivot_table(shipments, values=['WEIGHT'], index=['DEST', 'LS'], aggfunc=sum)
    nRows = nInternalZones + nSuperZones
    nCols = nLS
    zonalProductions = np.zeros((nRows, nCols))
    zonalAttractions = np.zeros((nRows, nCols))

    for x in prodWeight.index:
        orig = invZoneDict[x[0]]
        ls   = x[1]
        zonalProductions[orig, ls] += prodWeight['WEIGHT'][x]
    for x in attrWeight.index:
        orig = invZoneDict[x[0]]
        ls   = x[1]
        zonalAttractions[orig, ls] += attrWeight['WEIGHT'][x]

    cols = ['LS0', 'LS1', 'LS2', 'LS3',  'LS4', 'LS5', 'LS6', 'LS7']
    zonalProductions = pd.DataFrame(zonalProductions, columns=cols)
    zonalAttractions = pd.DataFrame(zonalAttractions, columns=cols)
    zonalProductions['ZONE'] = list(zoneDict.values())
    zonalAttractions['ZONE'] = list(zoneDict.values())
    zonalProductions['TOT_WEIGHT'] = np.sum(zonalProductions[cols], axis=1)
    zonalAttractions['TOT_WEIGHT'] = np.sum(zonalAttractions[cols], axis=1)

    cols = ['ZONE', 'LS0', 'LS1', 'LS2', 'LS3', 'LS4', 'LS5', 'LS6', 'LS7', 'TOT_WEIGHT']
    zonalProductions = zonalProductions[cols]
    zonalAttractions = zonalAttractions[cols]

    return zonalProductions, zonalAttractions


def write_shipments_to_shp(
    shipments: pd.DataFrame,
    origX: Dict[int, float],
    origY: Dict[int, float],
    destX: Dict[int, float],
    destY: Dict[int, float],
    varDict: Dict[str, str],
    root: Any,
    percStart: float,
    percEnd: float,
) -> None:
    """
    Writes the shipments to a shapefile in the output folder.
    """
    Ax = list(origX.values())
    Ay = list(origY.values())
    Bx = list(destX.values())
    By = list(destY.values())

    # Initialize shapefile fields
    w = shp.Writer(f"{varDict['OUTPUTFOLDER']}Shipments_{varDict['LABEL']}.shp")

    w.field('SHIP_ID',      'N', size=6, decimal=0)
    w.field('ORIG',         'N', size=8, decimal=0)
    w.field('DEST',         'N', size=8, decimal=0)
    w.field('NSTR',         'N', size=2, decimal=0)
    w.field('WEIGHT',       'N', size=4, decimal=2)
    w.field('WEIGHT_CAT',   'N', size=2, decimal=0)
    w.field('FLOWTYPE',     'N', size=2, decimal=0)
    w.field('LS',           'N', size=2, decimal=0)
    w.field('VEHTYPE',      'N', size=2, decimal=0)
    w.field('SEND_FIRM',    'N', size=8, decimal=0)
    w.field('RECEIVE_FIRM', 'N', size=8, decimal=0)
    w.field('SEND_DC',      'N', size=6, decimal=0)
    w.field('RECEIVE_DC',   'N', size=6, decimal=0)
    w.field('TOD_PERIOD',   'N', size=2, decimal=0)
    w.field('TOD_LOWER',    'N', size=2, decimal=0)
    w.field('TOD_UPPER',    'N', size=2, decimal=0)
    if varDict['LABEL'] == 'UCC':
        w.field('FROM_UCC', 'N', size=2, decimal=0)
        w.field('TO_UCC',   'N', size=2, decimal=0)
            
    dbfData = np.array(shipments, dtype=object)
    nShips = shipments.shape[0]

    for i in range(nShips):

        # Add geometry
        w.line([[[Ax[i], Ay[i]], [Bx[i], By[i]]]])
        
        # Add data fields
        w.record(*dbfData[i, :])
                        
        if i % int(round(nShips / 20)) == 0:
            print('\t' + str(round(i / nShips * 100, 1)) + '%',end='\r')

            if root is not None:
                root.progressBar['value'] = (
                    percStart +
                    (percEnd - percStart) * i / nShips)

    w.close()