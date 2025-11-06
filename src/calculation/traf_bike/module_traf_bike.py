import logging
import numpy as np
import pandas as pd
import sys
import traceback

from datetime import datetime
from typing import Any, Dict, List

from calculation.common.dimensions import ModelDimensions
from .support_traf_bike import (
    read_shape, write_shape, writeTestMTX, writeMTX,
    createLinkDict, run_in_batches, process_mat,
    create_parking_heatmap_shape,
)

logger = logging.getLogger("tfs")

# Fill in the maximum memory wished to be used per run
MEMORY_PER_RUN = 7000  # MB


def actually_run_module(
    root: Any,
    varDict: Dict[str, Any],
    dims: ModelDimensions,
):
    """Perform the calculations of the Bike Traffic Assignment module."""
    try:

        if root is not None:
            root.progressBar['value'] = 0

        # Fill the label from the two options: 'MIC_individual_EB' / 'MIC_collab_EB'
        label = str(varDict['LABEL'])
        if not label.startswith('MIC'):
            raise Exception("Label should start with 'MIC' for TRAF_BIKE module.")

        # Fill the area from the three options: 'OD' / 'Studiegebied' / 'All'
        area = "Studiegebied"

        # Fill the chosen OD
        testMatrix = [6200, 6201]

        # Apply micronodes for trip ends or not
        use_micronodes = True

        # Use speed factors or time factors (False means using time factors)
        speed_factors = False

        # Fill the user classes from the four options: 'DIST', 'TIME', 'COMB', 'CABI'
        user_classes: List[str] = varDict['BIKE_USER_CLASSES']
        user_class_mapping = {item: i + 1 for i, item in enumerate(['DIST', 'TIME', 'COMB', 'CABI'])}

        for user_class in user_classes:
            if user_class_mapping.get(user_class) is None:
                raise ValueError(f"User class ({user_class}) should be one of 'DIST', 'TIME', 'COMB' or 'CABI'.")

        # Fill the timeperiod from the three options: 'OS' (07:00-09:00) / 'RD' (Rest of the day) / 'AS' (16:00-18:00)
        times: List[str] = varDict['BIKE_PERIODS']

        for time_period in times:
            if time_period not in ['OS', 'RD', 'AS']:
                raise ValueError(f"Time period ({time_period}) should be one of 'OS', 'RD' or 'AS'.")

        # Average waiting time at a traffic light
        traffic_light_seconds = float(varDict['BIKE_TRAFFIC_LIGHT_SECONDS'])

        # Factor dictionaries
        if speed_factors:
            v_fac_wt = {
                'Normale_weg':  1,
                '<undefined>':  1,
                'Onbekend':     1,
                'Fietspad':     1.1,
                '':             1,
                'Solitair_fietspad': 1,
                'Bromfietspad': 1.2,
                'Weg_met_fietssuggestiestrook': 1.2,
                'Voetgangersgebied': 0.6,
                'Solitair_bromfietspad': 1.1,
                'Ventweg':      1.1,
                'Voetgangerdoorsteekje': 0.5,
                'Fietsstraat':  1.2,
                'Brug_RTD':     0.8,
                'Veerpont':     0.5,
            }
            v_fac_verh = {
                'Klinkers':     0.6,
                '<undefined>':  1,
                'Asfalt_beton': 1.2,
                'Onbekend':     1,
                'Tegels':       0.7,
                '':             1,
                'Overig':       0.8,
                'Halfverhard':  0.6,
                'Onverhard':    0.4,
                'Schelpenpad':  0.5,
            }
        else:
            v_fac_wt = {
                'Normale_weg':  1.085,
                '<undefined>':  1,
                'Onbekend':     1,
                'Fietspad':     1.05,
                '':             1,
                'Solitair_fietspad': 0.862,
                'Bromfietspad': 1.05,
                'Weg_met_fietssuggestiestrook': 1.0,
                'Voetgangersgebied': 1.5,
                'Solitair_bromfietspad': 0.862,
                'Ventweg':      1.1,
                'Voetgangerdoorsteekje': 1.5,
                'Fietsstraat':  0.956,
                'Brug_RTD':     1.1,
                'Veerpont':     1.5,
            }
            v_fac_verh = {
                'Klinkers':     1.183,
                '<undefined>':  1,
                'Asfalt_beton': 1,
                'Onbekend':     1,
                'Tegels':       1.169,
                '':             1,
                'Overig':       1,
                'Halfverhard':  1.183,
                'Onverhard':    1.183,
                'Schelpenpad':  1.183,
            }

        start_time = datetime.now()

        demand_path = f"{varDict['OUTPUTFOLDER']}ParcelDemand_{label}.csv"

        if area == 'OD':
            loadcsv_path = f"{varDict['OUTPUTFOLDER']}bike_links_loaded_{area}.csv"
            updated_links_path = f"{varDict['OUTPUTFOLDER']}bike_links_loaded_{area}.shp"
            disaggregated_trips_path = f"{varDict['OUTPUTFOLDER']}disaggregated_trips_{area}.csv"
            parking_heatmap_path = f"{varDict['OUTPUTFOLDER']}parking_heatmap_{area}.shp"
        else:
            loadcsv_path = f"{varDict['OUTPUTFOLDER']}bike_links_loaded_{area}_{label}.csv"
            updated_links_path = f"{varDict['OUTPUTFOLDER']}bike_links_loaded_{area}_{label}.shp"
            disaggregated_trips_path = f"{varDict['OUTPUTFOLDER']}disaggregated_trips_{area}_{label}.csv"
            parking_heatmap_path = f"{varDict['OUTPUTFOLDER']}parking_heatmap_{area}_{label}.shp"

        logger.debug("\tSettings:")
        for attr_name in (
            "area",
            "testMatrix",
            "use_micronodes",
            "speed_factors",
            "v_fac_wt",
            "v_fac_verh",
        ):
            attr_value = locals().get(attr_name)
            if isinstance(attr_value, dict):
                logger.debug(f"\t\t{attr_name}:")
                for k, v in attr_value.items():
                    logger.debug(f"\t\t\t{k}: {v}")
            else:
                logger.debug(f"\t\t{attr_name}: {attr_value}")

        # Log network assignment details
        total_assignments = len(times) * len(user_class_mapping)
        logger.debug(
            f'\tPerforming {total_assignments} network assignments [{len(times)} time period(s) x {len(user_classes)} user classes]',
            extra={'memory_usage': '-'},
        )

        micro_vt_tag = label.split('_')[-1]

        micro_vt = {
            row['Tag']: row['Veh_ID']
            for row in pd.read_csv(varDict['VEHICLETYPES'], sep=',').to_dict('records')
        }.get(micro_vt_tag)

        if micro_vt is None:
            raise Exception(f"Vehicle type '{micro_vt_tag}' not found in '{varDict['VEHICLETYPES']}'.")

        logger.debug(f'\tMicro vehicle type: {micro_vt}')

        logger.debug('\tReading network shapefiles', extra={'memory_usage': '-'})
        zones = read_shape(varDict['CENTROIDS']).sort_values(by='CENTROIDNR').reset_index(drop=True)
        areas_validated = read_shape(varDict['ZONES'])

        links = read_shape(varDict['BIKE_LINKS'], encoding='latin1')
        nodes = read_shape(varDict['BIKE_NODES'])
        traffic_lights_nodes = read_shape(varDict["TRAFFIC_LIGHTS"])

        if root is not None:
            root.progressBar['value'] = 3.0

        logger.debug('\tReading parcel demand', extra={'memory_usage': '-'})
        demand = pd.read_csv(demand_path)
        demand = demand.loc[demand['VEHTYPE'] == micro_vt]

        if label == "MIC_collab_EB":
            demand['CEP'] = 'all'

        # Calculate parcel count per destination zone and receiver type
        parcel_counts = demand.groupby(['D_zone', 'Receiver'])['Parcel_ID'].count().reset_index(name='ParcelCount')

        # Calculate total parcels per destination zone
        parcel_counts['TotalParcels'] = parcel_counts.groupby(['D_zone'])['ParcelCount'].transform('sum')

        # Calculate share per Receiver in each O-D destination zone
        parcel_counts['Share'] = parcel_counts['ParcelCount'] / parcel_counts['TotalParcels']

        logger.debug('\tAdjusting network', extra={'memory_usage': '-'})

        nodes.loc[
            (nodes['TYPENO'] == 99) & (nodes['AREANR'].isnull()), 'AREANR'
        ] = nodes.loc[(nodes['TYPENO'] == 99) & (nodes['AREANR'].isnull()), 'NODENR'] - 10000000

        # Add traffic light delays
        links['TRAFFIC_LIGHT'] = 0.0
        nodes_with_delay = links.loc[links['B'].isin(traffic_lights_nodes['NODENR']), 'A'].values
        links.loc[links['A'].isin(nodes_with_delay), 'TRAFFIC_LIGHT'] = (traffic_light_seconds / 3600)

        # Create dictionaries for object mapping
        idx2zone = {idx: val for idx, val in enumerate(zones['CENTROIDNR'])}
        zone2idx = {val: idx for idx, val in enumerate(zones['CENTROIDNR'])}
        idx2node = {idx: val for idx, val in enumerate(nodes['NODENR'])}
        node2idx = {val: idx for idx, val in enumerate(nodes['NODENR'])}

        nzones = int(zones['AREANR'].max())

        zones_zez2 = areas_validated.loc[areas_validated['ZEZ'] == 2, 'AREANR'].values
        micro_nodes = read_shape(varDict["MICRONODES"])
        micro_nodes = micro_nodes[['AREANR', 'NODENR', 'WOON', 'KANTOOR']]

        micro_nodes = micro_nodes.astype(int)
        for node_nr_init, node_nr_new in (
            (7394, 197397),
            (7294, 300845),
            (7349, 199068),
            (181417, 194263),
            (196726, 194215),
            (95590, 195520),
            (190245, 195665),
        ):
            micro_nodes.loc[micro_nodes['NODENR'] == idx2node[node_nr_init], 'NODENR'] = node_nr_new

        micro_nodes = micro_nodes.loc[(micro_nodes['WOON'] != 0) | (micro_nodes['KANTOOR'] != 0)]
        micro_nodes = micro_nodes.loc[micro_nodes['AREANR'].isin(zones_zez2)].reset_index(drop=True)

        micro_nodes['NODENR_INDEX'] = micro_nodes['NODENR'].map(node2idx)

        # Set origin selection
        if area == "OD":
            zone_idx = node2idx[nodes.loc[(nodes['TYPENO'] == 99) & (nodes['AREANR'] == testMatrix[0]), 'NODENR'].values[0]]
            zoneArray = np.array([zone_idx]) if zone_idx is not None else np.array([])
            mtx_path = f"{varDict['OUTPUTFOLDER']}test_OD_{testMatrix[0]}-{testMatrix[1]}.MTX"
            writeTestMTX(testMatrix, mtx_path, nzones)
        else:
            for t in times:
                writeMTX(varDict['OUTPUTFOLDER'], times, label, nzones, micro_vt)
            zoneArray = np.array([node2idx[zone] for zone in nodes.loc[nodes['TYPENO'] == 99, 'NODENR'].values])

        nodeArray = np.arange(len(nodes))

        # Determine link dataframe columns
        speed_cols = ['A', 'B', 'LENGTH', 'WEGTYPE', 'FIETS_WT', 'VERHARDING', 'TRAFFIC_LIGHT']
        load_cols = ['LINKNR', 'A', 'B']

        for time_period in times:
            speed_cols.append(f'V_{time_period}')

        # Create numpy link array
        linkArray = links[speed_cols].copy()

        # High penalty for traveling on connectors
        linkArray.loc[linkArray['WEGTYPE'] == 'voedingslink', 'LENGTH'] = 999

        linkArray['A_map'] = linkArray["A"].map(node2idx)
        linkArray['B_map'] = linkArray["B"].map(node2idx)

        # Remove unmapped links to test remainder of the module
        linkArray = linkArray.dropna(subset=['A_map', 'B_map'])
        linkArray = linkArray.drop(columns=['A_map', 'B_map'])

        linkArray['A'] = linkArray['A'].map(node2idx).astype(np.int32)
        linkArray['B'] = linkArray['B'].map(node2idx).astype(np.int32)

        linkArray = linkArray.set_index(pd.Index(range(len(linkArray))))

        linksAB = linkArray[['A', 'B']].to_numpy()

        # Get maximum number of connections on nodes for Dijkstra algorithm
        maxConnections = linkArray['A'].value_counts().max()

        # Create linkDict for network loading
        linkDict = createLinkDict(linksAB, nodeArray, maxConnections)

        headerType = np.dtype([('soort', np.uint8), ('nZones', np.uint16), ('zeros', np.int32)])

        userClassInput = []
        cpu = 0
        for t in times:
            linkArray['DIST'] = (linkArray['LENGTH'] / 15)  # Divided by 15 for normalization
            linkArray['TIME'] = linkArray['LENGTH'] / linkArray[f'V_{t}']
            linkArray['COMB'] = linkArray['TIME'] * 0.5 + linkArray['DIST'] * 0.5

            if speed_factors:
                cargobike_time = linkArray['TRAFFIC_LIGHT'] + linkArray['LENGTH'] / (
                    linkArray[f'V_{t}'] *
                    linkArray['FIETS_WT'].map(v_fac_wt) *
                    linkArray['VERHARDING'].map(v_fac_verh)
                )
            else:
                cargobike_time = linkArray['TRAFFIC_LIGHT'] + linkArray['TIME'] * (
                    linkArray['FIETS_WT'].map(v_fac_wt) *
                    linkArray['VERHARDING'].map(v_fac_verh)
                )

            # Set the weight of distance to 0
            linkArray['CABI'] = (cargobike_time * 1.0 + linkArray['DIST'] * 0.0)

            if area == "OD":
                mtx_path = f"{varDict['OUTPUTFOLDER']}test_OD_{testMatrix[0]}-{testMatrix[1]}.MTX"
                cMat = process_mat(mtx_path, headerType, zone2idx)
            else:
                regularbikes_mtx_path = f"{varDict['INPUTFOLDER']}Fiets-{t}.MTX"
                regularbikes_mat = process_mat(regularbikes_mtx_path, headerType, zone2idx)
                regularbikes_mat *= 0.3333
                regularbikes_mat_dict = {}
                regularbikes_mat_dict['intermediate'] = regularbikes_mat

                cargobikes_mat_dict = {}
                for position in ['first', 'intermediate', 'last']:
                    cargobikes_mtx_path = f"{varDict['OUTPUTFOLDER']}cargo_bikes_{t}_{label}_{position}.MTX"
                    cargobikes_mat = process_mat(cargobikes_mtx_path, headerType, zone2idx)
                    cargobikes_mat_dict[f'{position}'] = cargobikes_mat

            for u in user_classes:
                logger.debug(f'\tCollecting input for time period {t} and user class {u}', extra={'memory_usage': '-'})

                if area == "OD":
                    raise NotImplementedError("No cMatDict specified for area=OD")

                if u == 'CABI':
                    cMatDict = cargobikes_mat_dict.copy()
                else:
                    cMatDict = regularbikes_mat_dict.copy()

                costArray = linkArray[u].fillna(999999).to_numpy()
                cpu += 1
                userClassInput.append((
                    cpu, t, u, zoneArray, linksAB, costArray, maxConnections, cMatDict,
                    linkDict, user_class_mapping[u], idx2node, idx2zone,
                    parcel_counts, micro_nodes, use_micronodes, varDict['OUTPUTFOLDER'],
                ))

        if root is not None:
            root.progressBar['value'] = 5.0

        results, trip_matrix = run_in_batches(userClassInput, MEMORY_PER_RUN)

        if root is not None:
            root.progressBar['value'] = 95.0

        trip_matrix.to_csv(disaggregated_trips_path, index=False)

        create_parking_heatmap_shape(trip_matrix, nodes, parking_heatmap_path)

        # Merge results
        logger.debug(f'\tMerging links with {len(results)} loads', extra={'memory_usage': '-'})

        links_loads = pd.merge(links[load_cols], results, left_on=['A', 'B'], right_on=['a_map', 'b_map'])

        links_loads = links_loads.drop(columns=['a_map', 'b_map'])
        links_loads = links_loads.fillna(0)

        logger.debug('\tWriting loads to CSV file', extra={'memory_usage': '-'})
        links_loads.to_csv(loadcsv_path, index=False)

        logger.debug('\tWriting loads to SHP file', extra={'memory_usage': '-'})

        links_updated = pd.merge(links_loads, links[['LINKNR', 'A', 'B', 'geometry']], on=['LINKNR', 'A', 'B'])

        write_shape(links_updated, updated_links_path)

        # Log completion time
        calc_time = datetime.now() - start_time
        logger.debug(f'\tDone! Calculation time: {calc_time}', extra={'memory_usage': '-'})

        if root is not None:
            root.progressBar['value'] = 100

        return [0, [0, 0]]

    except Exception:
        return [1, [sys.exc_info()[0], traceback.format_exc()]]
