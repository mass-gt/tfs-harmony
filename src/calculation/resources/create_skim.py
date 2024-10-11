import array
import pandas as pd
import numpy as np
import time
import datetime
import sys
import traceback
import multiprocessing as mp
import functools
import shapefile as shp

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
from typing import Any, List, Tuple, Union


datapathO = "P:/Projects_Active/23034 ROT GLEAM Cargo Bike Simulation/Work/Updaten basisjaar/[02] Netwerk/"

pathLinks = "P:/Projects_Active/23034 ROT GLEAM Cargo Bike Simulation/Work/Updaten basisjaar/[02] Netwerk/links_v12.shp"
pathNodes = "P:/Projects_Active/23034 ROT GLEAM Cargo Bike Simulation/Work/Updaten basisjaar/[02] Netwerk/nodes_v12.shp"
pathZones = "P:/Projects_Active/23034 ROT GLEAM Cargo Bike Simulation/Work/Updaten basisjaar/[02] Netwerk/areas_validated.shp"

label = 'REF'


#%%

def write_mtx(file_path: str, mat: np.ndarray, nZones: int) -> None:
    '''
    Write an array into a binary file
    '''
    mat = np.append(nZones, mat)
    matBin = array.array('f')
    matBin.fromlist(list(mat))
    matBin.tofile(open(file_path, 'wb'))


def read_shape(
    shapePath: str, encoding: str = 'latin1', returnGeometry: bool=False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Any]]]:
    '''
    Read the shapefile with zones (using pyshp --> import shapefile as shp)
    '''
    # Load the shape
    sf = shp.Reader(shapePath, encoding=encoding)
    records = sf.records()
    if returnGeometry:
        geometry = sf.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf.fields
    sf.close()

    # Get information on the fields in the DBF
    columns = [x[0] for x in fields[1:]]
    colTypes = [x[1:] for x in fields[1:]]
    nRecords = len(records)

    # Check for headers that appear twice
    for col in range(len(columns)):
        name = columns[col]
        whereName = [i for i in range(len(columns)) if columns[i] == name]
        if len(whereName) > 1:
            for i in range(1, len(whereName)):
                columns[whereName[i]] = (
                    str(columns[whereName[i]]) + '_' + str(i))

    # Put all the data records into a NumPy array
    # (much faster than Pandas DataFrame)
    shape = np.zeros((nRecords, len(columns)), dtype=object)
    for i in range(nRecords):
        shape[i, :] = records[i][0:]

    # Then put this into a Pandas DataFrame with
    # the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for col in range(len(columns)):
        if colTypes[col][0] == 'C':
            shape[columns[col]] = shape[columns[col]].astype(str)
        else:
            shape.loc[pd.isna(shape[columns[col]]), columns[col]] = -99999
            if colTypes[col][-1] > 0:
                shape[columns[col]] = shape[columns[col]].astype(float)
            else:
                shape[columns[col]] = shape[columns[col]].astype(int)

    if returnGeometry:
        return (shape, geometry)
    else:
        return shape


def get_skims(csgraph, nNodes, zoneDict, linkDict, linksTime, linksDist, nZones, indices):
    '''
    For each origin zone and destination node, determine the previously visited node on the shortest path.
    Then deduce the path and calculate the total travel time and distance.
    '''
    whichCPU = indices[1]
    indices  = indices[0]
    nOrigSelection = len(indices)

    skimTime = np.zeros((nOrigSelection,nZones), dtype=float)
    skimDist = np.zeros((nOrigSelection,nZones), dtype=float)

    for i in range(nOrigSelection):
        prev = dijkstra(csgraph, indices=indices[i], return_predecessors=True)[1]

        for j in range(nZones):
            destNode = zoneDict[j]
            sequenceNodes = []
            sequenceLinks = []

            if prev[destNode] >= 0:
                while prev[destNode]>=0:
                    sequenceNodes.insert(0,destNode)
                    destNode = prev[destNode]
                else:
                    sequenceNodes.insert(0,destNode)

            sequenceLinks = [linkDict[sequenceNodes[w]][sequenceNodes[w+1]] for w in range(len(sequenceNodes)-1)]

            skimTime[i,j] = np.sum(linksTime[sequenceLinks])
            skimDist[i,j] = np.sum(linksDist[sequenceLinks])

        if whichCPU == 0:
            if i%int(round(nOrigSelection/20,0)) == 0:
                print('\t\t' + str(int(round((i / nOrigSelection)*100, 0))) + '%')

    if whichCPU == 0:
        if i%int(round(nOrigSelection/10,0)) != 0:
            print('\t\t100%')

                
    return [skimTime, skimDist]


def main():
    '''
    Skim: Main body of the script where all calculations are performed. 
    '''
    try:

        start_time = time.time()

        log_file=open(f"{datapathO}Logfile_Skim_{label}.log", "w")
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")

        # ------------------- Importing and preprocessing network ---------------------------------------
        print("Importing and preprocessing network..."), log_file.write("Importing and preprocessing network...\n")

        # Import node/link shapefiles into dataframes
        MRDHlinks, MRDHlinksGeometry = read_shape(pathLinks, returnGeometry=True)
        MRDHnodes = read_shape(pathNodes)
        nNodes = len(MRDHnodes)

        # Get zone numbering
        zones = read_shape(pathZones)
        zones = zones.loc[zones['AREANR'] <= 7400]
        zones = zones.sort_values('AREANR')
        areaNumbers = np.array(zones['AREANR'], dtype=int)
        nIntZones   = len(zones)
        nSupZones   = 43
        nZones      = nIntZones + nSupZones

        # Dictionary with zone numbers (keys) to corresponding centroid node ID as given in the shapefile (values)
        zoneDict = {}
        for i in range(nIntZones):
            zoneDict[i] = np.where(MRDHnodes['NODENR']==areaNumbers[i] + 10000000)[0][0]
        for i in range(nSupZones):
            zoneDict[nIntZones+i] = np.where((MRDHnodes['AREANR']==99999900+i+1) & (MRDHnodes['TYPENO']==99))[0][0]

        # The node IDs as given in the shapefile (values) and a new ID numbering from 0 to nNodes (keys)
        nodeDict    = dict((i, value) for i, value in enumerate(MRDHnodes['NODENR'].astype(int).values))
        invNodeDict = dict((v, k) for k, v in nodeDict.items())

        # Recode the node IDs
        MRDHnodes['NODENR'] = np.arange(nNodes)

        unknownNodes = list(set(
            [x for x in MRDHlinks['A'] if invNodeDict.get(x) is None] +
            [x for x in MRDHlinks['B'] if invNodeDict.get(x) is None]
        ))

        if unknownNodes:
            raise Exception(
                f"The following node numbers are found in the links file but not in the nodes file: {unknownNodes}"
            )

        # Recode all links
        MRDHlinks['A'] = [invNodeDict[x] for x in MRDHlinks['A']]
        MRDHlinks['B'] = [invNodeDict[x] for x in MRDHlinks['B']]

        # Recode the link IDs from 0 to len(MRDHlinks)
        MRDHlinks.index = np.arange(len(MRDHlinks))
        MRDHlinks.index = MRDHlinks.index.map(int)

        # Matrix with fromNodeID/toNodeID (row/col no.) and link IDs (values)
        linkDict = {}
        for i in MRDHlinks.index:
            aNode = MRDHlinks['A'][i]
            bNode = MRDHlinks['B'][i]
            try:
                linkDict[aNode][bNode] = i
            except:
                linkDict[aNode] = {}
                linkDict[aNode][bNode] = i

        # Travel times
        MRDHlinks.loc[MRDHlinks['V_FR_OS']<=0,'V_FR_OS'] = 50
        MRDHlinks['T0'] = MRDHlinks['LENGTH'] / MRDHlinks['V_FR_OS']
        MRDHlinks['Impedance'] = np.array(MRDHlinks['T0'].copy())

        # Set connector travel times high so these are not chosen other than for entering/leaving network
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='voedingslink','Impedance'] = 10000
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='Vrachtverbod','Impedance'] = 10000

        # Set travel times on links in ZEZ Rotterdam high so these are only used to go to UCC and not for through traffic
        if label == 'UCC':
            MRDHlinks.loc[MRDHlinks['ZEZ']==1, 'Impedance'] = 10000

        # The network with costs between nodes
        csgraph = lil_matrix((nNodes, nNodes))
        csgraph[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])] = np.array(MRDHlinks['Impedance'])

        # ------------------------- Route search --------------------------------------------
        print("Start traffic assignment"), log_file.write("Start traffic assignment\n")

        print("\tSearching routes..."), log_file.write("\tSearching routes...\n")
        # Het aantal CPUs waarover we de routekeuze spreiden
        nCPU = max(1, min(mp.cpu_count() - 1, 16))
        origSelection = np.arange(nZones)
        nOrigSelection = len(origSelection)

        # Vanuit welke nodes moet 'ie gaan zoeken
        indices = np.array([zoneDict[x] for x in origSelection], dtype=int)
        indicesPerCPU       = [[indices[cpu::nCPU], cpu] for cpu in range(nCPU)]
        origSelectionPerCPU = [np.arange(nOrigSelection)[cpu::nCPU] for cpu in range(nCPU)]

        linksTime = np.array(MRDHlinks['T0'], dtype=float)
        linksDist = np.array(MRDHlinks['LENGTH'], dtype=float)

        skimTime = np.zeros((nOrigSelection,nZones), dtype=float)
        skimDist = np.zeros((nOrigSelection,nZones), dtype=float)

        # Voer de Dijkstra functie uit
        with mp.Pool(nCPU) as p:
            skimPerCPU = p.map(functools.partial(
                get_skims, csgraph, nNodes, zoneDict, linkDict, linksTime, linksDist, nZones
            ), indicesPerCPU)

        # Combine the results from the different CPUs
        for cpu in range(nCPU):
            for i in range(len(indicesPerCPU[cpu][0])):
                skimTime[origSelectionPerCPU[cpu][i], :] = skimPerCPU[cpu][0][i, :]
                skimDist[origSelectionPerCPU[cpu][i], :] = skimPerCPU[cpu][1][i, :]

        print('Writing skim time...')
        write_mtx(datapathO + 'skimTijd_REF.mtx', np.array(np.round(skimTime.flatten()*3600,0), dtype=int), nZones)
        
        print('Writing skim distance...')
        write_mtx(datapathO + 'skimAfstand_REF.mtx', np.array(np.round(skimDist.flatten()*1000,0), dtype=int), nZones)

        # --------------------------- End of module ---------------------------

        totaltime = round(time.time() - start_time, 2)
        print("Total runtime: %s seconds" % (totaltime)), log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()

    except BaseException:
        print(sys.exc_info()[0]), log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        print(traceback.format_exc()), log_file.write(str(traceback.format_exc())), log_file.write("\n")
        print("Execution failed!")
        log_file.write("Execution failed!")
        log_file.close()

        if __name__ == '__main__':
            input()


#%%

if __name__ == '__main__':
    main()


#%% For visualization of a particular OD-path during debugging
    
#import matplotlib.pyplot as plt
#import geopandas as gpd
##xLower =  65000
##xUpper = 100000
##yLower = 425000
##yUpper = 470000
#
#xLower =  50000
#xUpper = 250000
#yLower = 300000
#yUpper = 500000
#
##zones = gpd.read_file(datapathI + 'Zones_v4.shp')
##links = gpd.read_file(linksPath)
#
#route = np.array(dijkstraPaths[6666][6])
##route = np.array(dijkstraPaths[6666][6])
#ax = zones.plot(figsize=(15,15),color='#b2b2b2')
#links.plot(ax=ax, color='k', linewidth=0.1)
#links.iloc[route,:].plot(ax=ax, color='r')
#plt.xlim(xLower,xUpper)
#plt.ylim(yLower,yUpper)

