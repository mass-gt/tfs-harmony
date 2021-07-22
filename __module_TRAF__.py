# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:33:15 2019

@author: STH
"""
import numpy as np
import pandas as pd
import shapefile as shp
import time
import datetime
import scipy.sparse.csgraph
from scipy.sparse import lil_matrix
import multiprocessing as mp
import functools
from shapely.geometry import Point, Polygon, MultiPolygon
from __functions__ import read_mtx, read_shape

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread



#%% Main

def main(varDict):
    '''
    Start the GUI object which runs the module
    '''
    root = Root(varDict)
    
    return root.returnInfo
    


class Root:
    
    def __init__(self, args):       
        '''
        Initialize a GUI object
        '''        
        # Set graphics parameters
        self.width  = 500
        self.height = 60
        self.bg     = 'black'
        self.fg     = 'white'
        self.font   = 'Verdana'
        
        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Progress Traffic Assignment")
        self.root.geometry(f'{self.width}x{self.height}+0+200')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg=self.bg)
        self.canvas.place(x=0, y=0)
        self.statusBar = tk.Label(self.root, text="", anchor='w', borderwidth=0, fg='black')
        self.statusBar.place(x=2, y=self.height-22, width=self.width, height=22)
        
        # Remove the default tkinter icon from the window
        icon = zlib.decompress(base64.b64decode('eJxjYGAEQgEBBiDJwZDBy''sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
        _, self.iconPath = tempfile.mkstemp()
        with open(self.iconPath, 'wb') as iconFile:
            iconFile.write(icon)
        self.root.iconbitmap(bitmap=self.iconPath)
        
        # Create a progress bar
        self.progressBar = Progressbar(self.root, length=self.width-20)
        self.progressBar.place(x=10, y=10)
        
        self.returnInfo = ""
        
        if __name__ == '__main__':
            self.args = [[self, args]]
        else:
            self.args = [args]
        
        self.run_module()       
        
        # Keep GUI active until closed    
        self.root.mainloop()
        
        
        
    def update_statusbar(self, text):
        self.statusBar.configure(text=text)



    def error_screen(self, text='', event=None, size=[800,50], title='Error message'):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{200+50+self.height}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(default=self.iconPath)
        labelError = tk.Label(windowError, text=text, anchor='w', justify='left')
        labelError.place(x=10, y=10)  
        
        

    def run_module(self, event=None):
        Thread(target=actually_run_module, args=self.args, daemon=True).start()
        


def actually_run_module(args):
    '''
    Traffic assignment: Main body of the script where all calculations are performed. 
    '''
    try:      
        
        start_time = time.time()
        
        root    = args[0]
        varDict = args[1]
        
        if root != '':
            root.progressBar['value'] = 0
                
        # Define folders relative to current datapath
        datapathI        = varDict['INPUTFOLDER']
        datapathO        = varDict['OUTPUTFOLDER']
        datapathP        = varDict['PARAMFOLDER']
        linksPath        = varDict['LINKS']
        nodesPath        = varDict['NODES']
        zonesPath        = varDict['ZONES']
        selectedLinks    = varDict['SELECTED_LINKS']
        skimDistancePath = varDict['SKIMDISTANCE']
        costSourcingPath = varDict['COST_SOURCING']
        costVehTypePath  = varDict['COST_VEHTYPE'] 
        label            = varDict['LABEL']    
        nCPU             = varDict['N_CPU']
        nMultiRoute      = varDict['N_MULTIROUTE']
        impedanceSpeedFreight = varDict['IMPEDANCE_SPEED_FREIGHT']
        impedanceSpeedVan     = varDict['IMPEDANCE_SPEED_VAN']
        
        exportShp  = True
        addZezToLinks = False
        
        if nMultiRoute == '':
            nMultiRoute = 1
        else:
            nMultiRoute = int(nMultiRoute)
        
        log_file = open(datapathO + "Logfile_TrafficAssignment.log", 'w')
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
            
        # To convert emissions to kilograms
        emissionDivFac = [1000, 1000000, 1000000, 1000] 
        etDict    = {0:'CO2', 1:'SO2', 2:'PM', 3:'NOX'}
        etInvDict = {'CO2':0, 'SO2':1, 'PM':2, 'NOX':3}
           
        nLS = 8 + 1 # Number of logistic segments (+ parcel module)
        nET = 4     # Number of emission types
        nVT = 10
        
        # Which vehicle type can be used in the parcel module        
        vehTypesParcels = [7, 8]
        
        # Enumerate the different time periods (i.e. hours) of the day
        nHours = 24
        timeOfDays = np.arange(nHours)
        
        # Carrying capacity in kg
        carryingCapacity = np.array(pd.read_csv(datapathP + 'CarryingCapacity.csv', index_col='Vehicle Type'))

        # For which LINKNR-values to perform selected link analyses
        doSelectedLink = (selectedLinks != "") and (selectedLinks != "''") and (selectedLinks != '""')
        if doSelectedLink:
            selectedLinks = selectedLinks.split(',')
            nSelectedLinks = len(selectedLinks)
            try:
                selectedLinks = [int(x) for x in selectedLinks]
            except:
                print('Warning! Could not convert SELECTED_LINKS to integers!')
                log_file.write('Warning! Could not convert SELECTED_LINKS to integers!' + '\n')
        
        # Number of CPUs over which the route search procedure is parallelized
        maxCPU = 16
        if nCPU not in ['', '""', "''"]:
            try:
                nCPU = int(nCPU)
                if nCPU > mp.cpu_count():
                    nCPU = max(1,min(mp.cpu_count()-1,maxCPU))
                    print(         'N_CPU parameter too high. Only ' + str(mp.cpu_count()) + ' CPUs available. Hence defaulting to ' + str(nCPU) + 'CPUs.')
                    log_file.write('N_CPU parameter too high. Only ' + str(mp.cpu_count()) + ' CPUs available. Hence defaulting to ' + str(nCPU) + 'CPUs.' + '\n')
                if nCPU < 1:
                    nCPU = max(1,min(mp.cpu_count()-1,maxCPU))
            except:
                nCPU = max(1,min(mp.cpu_count()-1,maxCPU))
                print(         'Could not convert N_CPU parameter to an integer. Hence defaulting to ' + str(nCPU) + 'CPUs.')
                log_file.write('Could not convert N_CPU parameter to an integer. Hence defaulting to ' + str(nCPU) + 'CPUs.' + '\n')
        else:
            nCPU = max(1,min(mp.cpu_count()-1,maxCPU))
            
        if root != '':
            root.progressBar['value'] = 0.2
            
            
        # ------------------- Importing and preprocessing network ---------------------------------------
        print("Importing and preprocessing network..."), log_file.write("Importing and preprocessing network...\n")
        
        # Import links
        MRDHlinks, MRDHlinksGeometry = read_shape(linksPath, returnGeometry=True)
        nLinks = len(MRDHlinks)
        
        if root != '':
            root.progressBar['value'] = 2.0
            
        # Import nodes and zones
        MRDHnodes            = read_shape(nodesPath)
        zones, zonesGeometry = read_shape(zonesPath, returnGeometry=True)
        nNodes = len(MRDHnodes)

        if root != '':
            root.progressBar['value'] = 2.5
            
        # Cost parameters freight
        costParamsSourcing = pd.read_csv(costSourcingPath)
        costPerKmFreight    = costParamsSourcing['CostPerKm'][0]
        costPerHourFreight  = costParamsSourcing['CostPerHour'][0]  

        # Cost parameters vans
        costParamsVehType = pd.read_csv(costVehTypePath, index_col=0)        
        costPerKmVan   = costParamsVehType.at['Van', 'CostPerKm']
        costPerHourVan = costParamsVehType.at['Van', 'CostPerH']
        
        # Get zone numbering
        zones = zones.sort_values('AREANR')
        areaNumbers = np.array(zones['AREANR'], dtype=int)
        nIntZones   = len(zones)
        nSupZones   = 43
        nZones      = nIntZones + nSupZones
        
        # If you want to do the spatial coupling of ZEZ zones to the links here instead of in QGIS
        if addZezToLinks:
            print('Performing spatial coupling of ZEZ-zones to links...'), log_file.write('Performing spatial coupling of ZEZ-zones to links...\n')
            # Get links as shapely Point objects
            shapelyLinks = []
            for x in MRDHlinksGeometry:
                shapelyLinks.append([Point(x['coordinates'][0]),Point(x['coordinates'][-1])])
                
            # Get zones as shapely MultiPolygon/Polygon objects
            shapelyZones = []
            for x in zonesGeometry:
                if x['type'] == 'MultiPolygon':
                    temp = [Polygon(x['coordinates'][0][i]) for i in range(len(x['coordinates'][0]))]
                    shapelyZones.append(MultiPolygon(temp))
                else:
                    shapelyZones.append(Polygon(x['coordinates'][0]))
            shapelyZonesZEZ = np.array(shapelyZones, dtype=object)[np.where(zones['ZEZ']==1)[0]]
            
            # Check if links are in ZEZ
            zezLinks      = np.zeros((len(MRDHlinks)), dtype=int)
            linksToCheck  = np.where((MRDHlinks['Gemeentena']!='') & (MRDHlinks['WEGTYPE']!='Autosnelweg'))[0]
            nLinksToCheck = len(linksToCheck)
            for i in range(nLinksToCheck):
                linkNo = linksToCheck[i]
                startPoint = shapelyLinks[linkNo][0]
                endPoint   = shapelyLinks[linkNo][1]
                
                # Check if startpoint is in ZEZ
                for shapelyZone in shapelyZonesZEZ:
                    if shapelyZone.contains(startPoint):
                        zezLinks[linkNo] = 1
                        break
                
                # If startpoint not in ZEZ, check if endpoint is in ZEZ
                if zezLinks[linkNo] == 0:
                    for shapelyZone in shapelyZonesZEZ:
                        if shapelyZone.contains(endPoint):
                            zezLinks[linkNo] = 1
                            break  
                
                if i%int(nLinksToCheck/20) == 0:
                    print('\t' + str(int(round((i / nLinksToCheck)*100, 0))) + '%') 
            
            print('\tFound ' + str(np.sum(zezLinks)) + ' links located in ZEZ.')

            MRDHlinks['ZEZ'] = zezLinks
            
            del shapelyLinks, shapelyZones, shapelyZonesZEZ
                    
        # Dictionary with zone numbers (keys) to corresponding centroid node ID as given in the shapefile (values) 
        zoneToCentroidKeys   = np.arange(nIntZones)
        zoneToCentroidValues = []
        for i in range(nIntZones):
            zoneToCentroidValues.append(np.where((MRDHnodes['AREANR']==areaNumbers[i]) & (MRDHnodes['TYPENO']==99))[0][0])
        zoneToCentroid = dict(np.transpose(np.vstack((zoneToCentroidKeys,zoneToCentroidValues))))
        for i in range(nSupZones):
            zoneToCentroid[nIntZones+i] = np.where((MRDHnodes['AREANR']==99999900+i+1) & (MRDHnodes['TYPENO']==99))[0][0]
            
        # Dictionary with zone number (used in this script) to corresponding zone number (used in input/output)
        zoneDict    = dict(np.transpose(np.vstack((np.arange(nIntZones), zones['AREANR']))))
        zoneDict    = {int(a):int(b) for a,b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones+i] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())   
        
        # The node IDs as given in the shapefile (values) and a new ID numbering from 0 to nNodes (keys)
        nodeDict    = dict(np.transpose(np.vstack( (np.arange(nNodes), MRDHnodes['NODENR']) )))
        invNodeDict = dict((v, k) for k, v in nodeDict.items())
        
        # Check on NODENR values
        nodenr = set(np.array(MRDHnodes['NODENR']))
        missingNodes = list(np.unique([x for x in MRDHlinks['A'] if x not in nodenr]))
        for x in MRDHlinks['B']:
            if x not in nodenr:
                if x not in missingNodes:
                    missingNodes.append(x)
        if len(missingNodes) > 0:
            raise BaseException("Error! The following NODENR values were found in the links shape but not in the nodes shape! " + str(missingNodes))            

        if root != '':
            root.progressBar['value'] = 3.0
            
        # Recode the node IDs
        MRDHnodes['NODENR'] = np.arange(nNodes)
        MRDHlinks['A']      = [invNodeDict[x] for x in MRDHlinks['A']]
        MRDHlinks['B']      = [invNodeDict[x] for x in MRDHlinks['B']]
        
        # Recode the link IDs from 0 to len(MRDHlinks)
        MRDHlinks.index = np.arange(len(MRDHlinks))
        MRDHlinks.index = MRDHlinks.index.map(int)
        
        # Dictionary with fromNodeID, toNodeID (keys) and link IDs (values)        
        linkDict = {}
        for i in MRDHlinks.index:
            aNode = MRDHlinks['A'][i]
            bNode = MRDHlinks['B'][i]
            try:
                linkDict[aNode][bNode] = i
            except:
                linkDict[aNode] = {}
                linkDict[aNode][bNode] = i
        
        # Assume a speed of 50 km/h if there are links with freight speed <= 0
        nSpeedZero = np.sum(MRDHlinks[impedanceSpeedFreight]<=0)
        if nSpeedZero > 0:
            MRDHlinks.loc[MRDHlinks[impedanceSpeedFreight]<=0, impedanceSpeedFreight] = 50 
            print(         '\tWarning: ' + str(nSpeedZero) + ' links found with freight speed (' + impedanceSpeedFreight + ') <= 0 km/h. Adjusting those to 50 km/h.')
            log_file.write('\tWarning: ' + str(nSpeedZero) + ' links found with freight speed (' + impedanceSpeedFreight + ') <= 0 km/h. Adjusting those to 50 km/h.' + '\n')

        # Assume a speed of 50 km/h if there are links with van speed <= 0
        nSpeedZero = np.sum(MRDHlinks[impedanceSpeedVan]<=0)
        if nSpeedZero > 0:
            MRDHlinks.loc[MRDHlinks[impedanceSpeedVan]<=0, impedanceSpeedVan] = 50 
            print(         '\tWarning: ' + str(nSpeedZero) + ' links found with van speed (' + impedanceSpeedVan + ') <= 0 km/h. Adjusting those to 50 km/h.')
            log_file.write('\tWarning: ' + str(nSpeedZero) + ' links found with van speed (' + impedanceSpeedVan + ') <= 0 km/h. Adjusting those to 50 km/h.' + '\n')
            
        # Travel times and travel costs
        MRDHlinks['T0_FREIGHT'  ] = MRDHlinks['LENGTH'] / MRDHlinks[impedanceSpeedFreight]
        MRDHlinks['T0_VAN'      ] = MRDHlinks['LENGTH'] / MRDHlinks[impedanceSpeedVan]
        MRDHlinks['COST_FREIGHT'] = costPerKmFreight * MRDHlinks['LENGTH'] + costPerHourFreight * MRDHlinks['T0_FREIGHT']
        MRDHlinks['COST_VAN'    ] = costPerKmVan     * MRDHlinks['LENGTH'] + costPerHourVan     * MRDHlinks['T0_VAN']
        
        # Set connector travel times high so these are not chosen other than for entering/leaving network
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='voedingslink','COST_FREIGHT'] = 10000
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='voedingslink','COST_VAN'    ] = 10000
        
        # Set travel times for forbidden-for-freight-links high so these are not chosen for freight
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='Vrachtverbod','COST_FREIGHT'] = 10000
        MRDHlinks.loc[MRDHlinks['WEGTYPE']=='Vrachtstrook','COST_FREIGHT'] = 10000
        
        # Set travel times on links in ZEZ Rotterdam high so these are only used to go to UCC and not for through traffic
        if label == 'UCC':
            MRDHlinks.loc[MRDHlinks['ZEZ']==1, 'COST_FREIGHT'] += 10000
            
        # Initialize empty fields with emissions and traffic intensity per link (also save list with all field names)
        volCols = ['N_LS0','N_LS1','N_LS2','N_LS3','N_LS4','N_LS5','N_LS6','N_LS7','N_LS8',\
                   'N_VAN_S','N_VAN_C',\
                   'N_VEH0','N_VEH1','N_VEH2','N_VEH3','N_VEH4','N_VEH5','N_VEH6','N_VEH7','N_VEH8','N_VEH9',\
                   'N_TOT']
        intensityFields = []
        intensityFieldsGeojson = []        
        for et in etDict.values():
            MRDHlinks[et] = 0
            intensityFields.append(et)
            intensityFieldsGeojson.append(et)
        for volCol in volCols:
            MRDHlinks[volCol] = 0
            intensityFields.append(volCol)
            intensityFieldsGeojson.append(volCol)
            
        # Intensity per time of day and emissions per logistic segment are not in de links geojson but in a seperate CSV
        for vt in vehTypesParcels:
            intensityFields.append('N_LS8_VEH' + str(vt))
        for tod in range(nHours):
            intensityFields.append('N_TOD' + str(tod))
        for ls in range(nLS):
            for et in etDict.values():
                intensityFields.append(et + '_LS' + str(ls))
        for ls in ['VAN_S', 'VAN_C']:
            for et in etDict.values():
                intensityFields.append(et + '_' + str(ls))
        MRDHlinksIntensities = pd.DataFrame(np.zeros((len(MRDHlinks),len(intensityFields))), columns=intensityFields)                                
        
        # Van trips for service and construction purposes
        vanTripsService      = read_mtx(datapathO + 'TripsVanService.mtx')
        vanTripsConstruction = read_mtx(datapathO + 'TripsVanConstruction.mtx')
        
        # ODs with very low number of trips: set to 0 to reduce memory burden of searches routes for all these ODs
        vanTripsService[np.where(vanTripsService<0.1)[0]] = 0
        vanTripsConstruction[np.where(vanTripsConstruction<0.1)[0]] = 0
        
        # Reshape to square array
        vanTripsService      = vanTripsService.reshape(nZones,nZones)
        vanTripsConstruction = vanTripsConstruction.reshape(nZones,nZones)
        
        # Make some space available on the RAM
        del zones, zonesGeometry, MRDHnodes
        
        if root != '':
            root.progressBar['value'] = 4.0
            
            
        # ----------------- Information for the emission calculations ------------------------------------
        # Lees emissiefactoren in (kolom 0=CO2, 1=SO2, 2=PM, 3=NOX)
        emissionsBuitenwegLeeg  = np.array(pd.read_csv(datapathI + "EmissieFactoren_BUITENWEG_LEEG.csv", index_col='Voertuigtype'))
        emissionsBuitenwegVol   = np.array(pd.read_csv(datapathI + "EmissieFactoren_BUITENWEG_VOL.csv", index_col='Voertuigtype'))
        emissionsSnelwegLeeg    = np.array(pd.read_csv(datapathI + "EmissieFactoren_SNELWEG_LEEG.csv", index_col='Voertuigtype'))
        emissionsSnelwegVol     = np.array(pd.read_csv(datapathI + "EmissieFactoren_SNELWEG_VOL.csv", index_col='Voertuigtype'))
        emissionsStadLeeg       = np.array(pd.read_csv(datapathI + "EmissieFactoren_STAD_LEEG.csv", index_col='Voertuigtype'))
        emissionsStadVol        = np.array(pd.read_csv(datapathI + "EmissieFactoren_STAD_VOL.csv", index_col='Voertuigtype'))
        
        # Average of small and large tractor+trailer
        emissionsBuitenwegLeeg[7,:] = (emissionsBuitenwegLeeg[7,:]  + emissionsBuitenwegLeeg[8,:]) / 2
        emissionsBuitenwegVol[ 7,:] = (emissionsBuitenwegVol[ 7,:]  + emissionsBuitenwegVol[ 8,:]) / 2
        emissionsSnelwegLeeg[  7,:] = (emissionsSnelwegLeeg[  7,:]  + emissionsSnelwegLeeg[  8,:]) / 2
        emissionsSnelwegVol[   7,:] = (emissionsSnelwegVol[   7,:]  + emissionsSnelwegVol[   8,:]) / 2
        emissionsStadLeeg[     7,:] = (emissionsStadLeeg[     7,:]  + emissionsStadLeeg[     8,:]) / 2
        emissionsStadVol[      7,:] = (emissionsStadVol[      7,:]  + emissionsStadVol[      8,:]) / 2

        # Average of small and large van
        emissionsBuitenwegLeeg[0,:] = (emissionsBuitenwegLeeg[0,:]  + emissionsBuitenwegLeeg[1,:]) / 2
        emissionsBuitenwegVol[ 0,:] = (emissionsBuitenwegVol[ 0,:]  + emissionsBuitenwegVol[ 1,:]) / 2
        emissionsSnelwegLeeg[  0,:] = (emissionsSnelwegLeeg[  0,:]  + emissionsSnelwegLeeg[  1,:]) / 2
        emissionsSnelwegVol[   0,:] = (emissionsSnelwegVol[   0,:]  + emissionsSnelwegVol[   1,:]) / 2
        emissionsStadLeeg[     0,:] = (emissionsStadLeeg[     0,:]  + emissionsStadLeeg[     1,:]) / 2
        emissionsStadVol[      0,:] = (emissionsStadVol[      0,:]  + emissionsStadVol[      1,:]) / 2
        
        # To vehicle type in the emission factors (value) does each of our vehicle types (key) belong
        vtDict = {0:2, 1:3, 2:5, 3:4, 4:6, 5:7, 6:9, 7:0, 8:0, 9:0}                
            
        # Import trips csv
        allTrips = pd.read_csv(datapathO + "Tours_" + label + ".csv")
        allTrips['ORIG'] = [invZoneDict[x] for x in allTrips['ORIG']]
        allTrips['DEST'] = [invZoneDict[x] for x in allTrips['DEST']]
        allTrips.loc[allTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        allTrips.loc[allTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        capUt = (allTrips['TRIP_WEIGHT']*1000) / carryingCapacity[np.array(allTrips['VEHTYPE'], dtype=int)][:,0]
        allTrips['CAP_UT'] = capUt
        allTrips['INDEX' ] = allTrips.index
        
        # Import parcel schedule csv         
        allParcelTrips = pd.read_csv(datapathO + "ParcelSchedule_" + label + ".csv")
        allParcelTrips = allParcelTrips.rename(columns={'O_zone':'ORIG', 'D_zone':'DEST', 'TripDepTime':'TRIP_DEPTIME'})
        allParcelTrips['ORIG'] = [invZoneDict[x] for x in allParcelTrips['ORIG']]
        allParcelTrips['DEST'] = [invZoneDict[x] for x in allParcelTrips['DEST']]
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        allParcelTrips.loc[allParcelTrips['TRIP_DEPTIME']>=24,'TRIP_DEPTIME'] -= 24
        allParcelTrips['CAP_UT'  ] = 0.5
        allParcelTrips['VEHTYPE' ] = [{'Van':7, 'LEVV':8}[vt] for vt in allParcelTrips['VehType']]
        allParcelTrips['COMBTYPE'] = 0 # Fuel as basic combustion type
        allParcelTrips.loc[allParcelTrips['OrigType']=='UCC', 'COMBTYPE'] = 1 # Trips coming from UCC to ZEZ use electric
        allParcelTrips['LOG_SEG' ] = 6
        allParcelTrips['INDEX'   ] = allParcelTrips.index
        
        # Determine linktypes (urban/rural/highway)
        stadLinkTypes       = ['ETW_bibeko_30',         'GOW_bibeko_50',                'GOW_bibeko_70',        \
                               'WOW_bibeko_50',         'Verblijfsgebied_15']
        buitenwegLinkTypes  = ['ETW_bubeko_breed_60',   'ETW_bubeko_smal_60',           'GOW_bubeko_gemengd_80',\
                               'GOW_bubeko_gesloten_80','Industrieontsluitingsweg_50',  'Industriestraat_30']
        snelwegLinkTypes    = ['Autosnelweg',           'Autoweg',                      'Vrachtverbod']
        
        whereStad       = [MRDHlinks['WEGTYPE'][i] in stadLinkTypes      for i in MRDHlinks.index]
        whereBuitenweg  = [MRDHlinks['WEGTYPE'][i] in buitenwegLinkTypes for i in MRDHlinks.index]
        whereSnelweg    = [MRDHlinks['WEGTYPE'][i] in snelwegLinkTypes   for i in MRDHlinks.index]      
       
        roadtypeArray = np.zeros((len(MRDHlinks)))
        roadtypeArray[whereStad     ] = 1
        roadtypeArray[whereBuitenweg] = 2
        roadtypeArray[whereSnelweg  ] = 3
        distArray = np.array(MRDHlinks['LENGTH'])
        ZEZarray  = np.array(MRDHlinks['ZEZ']==1, dtype=int)
                
        # Bring ORIG and DEST to the front of the list of column names
        newColOrder = volCols.copy()
        newColOrder.insert(0,'DEST')
        newColOrder.insert(0,'ORIG')
        
        # Trip matrices per time-of-day
        tripMatricesTOD = []
        for tod in range(nHours):
            tripMatricesTOD.append(pd.read_csv(datapathO + "tripmatrix_" + label + "_TOD" + str(tod) + ".txt", sep='\t'))
            tripMatricesTOD[tod]['ORIG'] = [invZoneDict[x] for x in tripMatricesTOD[tod]['ORIG'].values]
            tripMatricesTOD[tod]['DEST'] = [invZoneDict[x] for x in tripMatricesTOD[tod]['DEST'].values]
            tripMatricesTOD[tod]['N_LS8'  ] = 0
            tripMatricesTOD[tod]['N_VAN_S'] = 0
            tripMatricesTOD[tod]['N_VAN_C'] = 0
            tripMatricesTOD[tod] = tripMatricesTOD[tod][newColOrder]            
            tripMatricesTOD[tod] = np.array(tripMatricesTOD[tod])
            
        # Parcels trip matrices per time-of-day
        tripMatricesParcelsTOD = []
        for tod in range(nHours):
            tripMatricesParcelsTOD.append(pd.read_csv(datapathO + "tripmatrix_parcels_" + str(label) + "_TOD" + str(tod) + ".txt", sep='\t'))
            tripMatricesParcelsTOD[tod]['ORIG'] = [invZoneDict[x] for x in tripMatricesParcelsTOD[tod]['ORIG'].values]
            tripMatricesParcelsTOD[tod]['DEST'] = [invZoneDict[x] for x in tripMatricesParcelsTOD[tod]['DEST'].values]
            tripMatricesParcelsTOD[tod] = np.array(tripMatricesParcelsTOD[tod])
        
        # For which origin zones do we need to find the routes
        origSelection  = np.arange(nZones)
        nOrigSelection = len(origSelection)
                                   
        # Initialize arrays for intensities and emissions
        linkTripsArray        = [np.zeros((len(MRDHlinks),len(volCols))) for tod in range(nHours)]
        linkVanTripsArray     = np.zeros((len(MRDHlinks),2))
        linkEmissionsArray    = [[np.zeros((len(MRDHlinks),nET)) for tod in range(nHours)] for ls in range(nLS)]
        linkVanEmissionsArray = [np.zeros((len(MRDHlinks),nET)) for ls in ['VAN_S','VAN_C']]
        if doSelectedLink:
            selectedLinkTripsArray = np.zeros((len(MRDHlinks),nSelectedLinks))
       
        if root != '':
            root.progressBar['value'] = 5.0
        
        
        # ----------------------- Route search (freight) ----------------------------------------                   
        print("Start traffic assignment"), log_file.write("Start traffic assignment\n")        
        tripsCO2       = {}
        parcelTripsCO2 = {}
        
        # From which nodes do we need to perform the shortest path algoritm
        indices = np.array([zoneToCentroid[x] for x in origSelection], dtype=int)
        
        # List of matrices with for each node the previous node on the shortest path
        prevFreight = []
        
        # Route search freight
        if nCPU > 1:
            # From which nodes does every CPU perform the shortest path algorithm
            indicesPerCPU       = [[indices[cpu::nCPU], cpu] for cpu in range(nCPU)]
            origSelectionPerCPU = [np.arange(nOrigSelection)[cpu::nCPU] for cpu in range(nCPU)]
                        
            for r in range(nMultiRoute):
                print(         f"\tRoute search (freight - multirouting part {r+1})...")
                log_file.write(f"\tRoute search (freight - multirouting part {r+1})...\n")
                
                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                csgraphFreight[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])]  = np.array(MRDHlinks['COST_FREIGHT'])
                csgraphFreight[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))
                
                # Initialize a pool object that spreads tasks over different CPUs
                p = mp.Pool(nCPU)
                
                # Execute the Dijkstra route search
                prevFreightPerCPU = p.map(functools.partial(get_prev, csgraphFreight, nNodes), indicesPerCPU)
                
                # Wait for completion of processes
                p.close()
                p.join()
                
                # Combine the results from the different CPUs
                prevFreight.append(np.zeros((nOrigSelection,nNodes), dtype=int))
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevFreight[r][origSelectionPerCPU[cpu][i], :] = prevFreightPerCPU[cpu][i, :]
    
                # Make some space available on the RAM
                del prevFreightPerCPU
                
                if root != '':
                    root.progressBar['value'] = 5.0 + (33.0 - 5.0) * (1 + r) / nMultiRoute
            
        else:
            for r in range(nMultiRoute):
                print(         f"\tRoute search (freight - multirouting part {r+1})...")
                log_file.write(f"\tRoute search (freight - multirouting part {r+1})...\n")

                # The network with costs between nodes (freight)
                csgraphFreight = lil_matrix((nNodes, nNodes))
                csgraphFreight[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])]  = np.array(MRDHlinks['COST_FREIGHT'])
                csgraphFreight[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))
                
                # Execute the Dijkstra route search
                prevFreight.append(get_prev(csgraphFreight, nNodes, [indices, 0]))

                if root != '':
                    root.progressBar['value'] = 5.0 + (33.0 - 5.0) * (1 + r) / nMultiRoute
        
        # Make some space available on the RAM
        del csgraphFreight
        
            
        # --------------------- Emissions and intensities (freight) --------------------------------------     
        
        print("Calculating emissions and traffic intensities")
        log_file.write("Calculating emissions and traffic intensities\n")
        
        print('\tFreight tours...'), log_file.write('\tFreight tours...\n')
        for tod in timeOfDays:
            
            print('\t\tHour ' + str(tod+1) + ' of ' + str(nHours) + '...')
            
            # Nu de tod-specifieke tripmatrix in tripMatrix variabele zetten
            tripMatrix        = tripMatricesTOD[tod]
            tripMatrixOrigins = set(tripMatrix[:,0])
            
            # Selecteer de trips die vertrokken in de huidige time-of-day en bereken de capacity utilization
            trips = allTrips.loc[(allTrips['TRIP_DEPTIME'] >= tod) & (allTrips['TRIP_DEPTIME'] < tod+1),:]     
            trips = np.array(trips[['CARRIER_ID','ORIG','DEST','VEHTYPE','CAP_UT','LOG_SEG','COMBTYPE', 'INDEX']])
                
            for i in range(nOrigSelection):
                origZone  = origSelection[i]                
                origTrips = trips[(trips[:,1]==origZone), :]
                
                if origZone in tripMatrixOrigins:
                    destZoneIndex = np.where(tripMatrix[:,0]==origZone)[0]
                    
                    # Schrijf de volumes op de links
                    for j in destZoneIndex:
                        destZone = tripMatrix[j,1]
                        nTrips   = tripMatrix[j,2:]
                        
                        # Get route and part of route that is stad/buitenweg/snelweg and ZEZ/non-ZEZ
                        routes = []
                        routesStad      = []
                        routesBuitenweg = []
                        routesSnelweg   = []
                        ZEZSstad      = []
                        ZEZSbuitenweg = []
                        ZEZSsnelweg   = []                        
                        for r in range(nMultiRoute):
                            routes.append(get_route(i, destZone, prevFreight[r], zoneToCentroid, linkDict))
                            routesStad.append(     routes[r][roadtypeArray[routes[r]]==1])
                            routesBuitenweg.append(routes[r][roadtypeArray[routes[r]]==2])
                            routesSnelweg.append(  routes[r][roadtypeArray[routes[r]]==3])
                            ZEZSstad.append(     ZEZarray[routesStad[r]     ]==1)
                            ZEZSbuitenweg.append(ZEZarray[routesBuitenweg[r]]==1)
                            ZEZSsnelweg.append(  ZEZarray[routesSnelweg[r]  ]==1)
                        
                        # Bereken en schrijf de intensiteiten/emissies op de links
                        for ls in range(nLS):
                            # Welke trips worden allemaal gemaakt op de HB van de huidige iteratie van de ij-loop
                            currentTrips  = origTrips[(origTrips[:,2]==destZone) & (origTrips[:,5]==ls), :]
                            nCurrentTrips = len(currentTrips)
                            
                            for trip in range(nCurrentTrips):
                                vt      = int(currentTrips[trip,3])
                                ct      = int(currentTrips[trip,6])
                                capUt   =     currentTrips[trip,4]
                                
                                # Select which of the calculated routes to use for current trip
                                whichMultiRoute = np.random.randint(nMultiRoute)
                                route = routes[whichMultiRoute]
                                routeStad      = routesStad[whichMultiRoute]
                                routeBuitenweg = routesBuitenweg[whichMultiRoute]
                                routeSnelweg   = routesSnelweg[whichMultiRoute]
                                ZEZstad      = ZEZSstad[whichMultiRoute]
                                ZEZbuitenweg = ZEZSbuitenweg[whichMultiRoute]
                                ZEZsnelweg   = ZEZSsnelweg[whichMultiRoute]
                                       
                                # Keep track of links being used on the route
                                linkTripsArray[tod][route, ls]        += 1
                                linkTripsArray[tod][route, nLS+2+vt ] += 1
                                linkTripsArray[tod][route, nLS+2+nVT] += 1                                  
                                if doSelectedLink:
                                    for l in range(nSelectedLinks):
                                        if selectedLinks[l] in route:
                                            selectedLinkTripsArray[route,l] += 1
                                
                                # If combustion type is fuel, hybrid or bio-fuel                             
                                if ct in [0,3,4]:
                                    # CO2
                                    stadCO2         = distArray[routeStad]      * (emissionsStadLeeg[vtDict[vt],etInvDict['CO2']]      + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['CO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']]))
                                    buitenwegCO2    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['CO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']]))
                                    snelwegCO2      = distArray[routeSnelweg]   * (emissionsSnelwegLeeg[vtDict[vt],etInvDict['CO2']]   + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['CO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']]))
                                    if ct == 3:
                                        stadCO2[ZEZstad          ] = 0
                                        buitenwegCO2[ZEZbuitenweg] = 0
                                        snelwegCO2[ZEZsnelweg    ] = 0
                                    linkEmissionsArray[ls][tod][routeStad, etInvDict['CO2']]      += stadCO2
                                    linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['CO2']] += buitenwegCO2
                                    linkEmissionsArray[ls][tod][routeSnelweg, etInvDict['CO2']]   += snelwegCO2
                
                                    # SO2
                                    stadSO2         = distArray[routeStad]      * (emissionsStadLeeg[vtDict[vt],etInvDict['SO2']]      + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['SO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']]))
                                    buitenwegSO2    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['SO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']]))
                                    snelwegSO2      = distArray[routeSnelweg]   * (emissionsSnelwegLeeg[vtDict[vt],etInvDict['SO2']]   + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['SO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']]))
                                    if ct == 3:
                                        stadSO2[ZEZstad          ] = 0
                                        buitenwegSO2[ZEZbuitenweg] = 0
                                        snelwegSO2[ZEZsnelweg    ] = 0  
                                    linkEmissionsArray[ls][tod][routeStad, etInvDict['SO2']]      += stadSO2
                                    linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['SO2']] += buitenwegSO2
                                    linkEmissionsArray[ls][tod][routeSnelweg, etInvDict['SO2']]   += snelwegSO2
                                    
                                    # PM
                                    stadPM          = distArray[routeStad]      * (emissionsStadLeeg[vtDict[vt],etInvDict['PM']]      + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['PM']] - emissionsStadLeeg[vtDict[vt],     etInvDict['PM']]))
                                    buitenwegPM     = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['PM']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']]))
                                    snelwegPM       = distArray[routeSnelweg]   * (emissionsSnelwegLeeg[vtDict[vt],etInvDict['PM']]   + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['PM']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']]))
                                    if ct == 3:
                                        stadPM[ZEZstad          ] = 0
                                        buitenwegPM[ZEZbuitenweg] = 0
                                        snelwegPM[ZEZsnelweg    ] = 0
                                    linkEmissionsArray[ls][tod][routeStad, etInvDict['PM']]      += stadPM
                                    linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['PM']] += buitenwegPM
                                    linkEmissionsArray[ls][tod][routeSnelweg, etInvDict['PM']]   += snelwegPM
                                    
                                    # NOX
                                    stadNOX         = distArray[routeStad]      * (emissionsStadLeeg[vtDict[vt],etInvDict['NOX']]      + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['NOX']] - emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']]))
                                    buitenwegNOX    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['NOX']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']]))
                                    snelwegNOX      = distArray[routeSnelweg]   * (emissionsSnelwegLeeg[vtDict[vt],etInvDict['NOX']]   + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['NOX']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']]))
                                    if ct == 3:
                                        stadNOX[ZEZstad          ] = 0
                                        buitenwegNOX[ZEZbuitenweg] = 0
                                        snelwegNOX[ZEZsnelweg    ] = 0
                                    linkEmissionsArray[ls][tod][routeStad, etInvDict['NOX']]      += stadNOX
                                    linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['NOX']] += buitenwegNOX
                                    linkEmissionsArray[ls][tod][routeSnelweg, etInvDict['NOX']]   += snelwegNOX

                                    tripsCO2[currentTrips[trip,-1]] = (np.sum(stadCO2) + np.sum(buitenwegCO2) + np.sum(snelwegCO2))

            if root != '':
                root.progressBar['value'] = 33.0 + (43.0 - 33.0) * (tod + 1) / nHours
        
        del prevFreight
        
        
        # ----------------------- Route search (vans) ----------------------------------------   
        
        # List of matrices with for each node the previous node on the shortest path
        prevVan     = []
        
        # Route search vans
        if nCPU > 1:                        
            for r in range(nMultiRoute):
                print(         f"\tRoute search (vans - multirouting part {r+1})...")
                log_file.write(f"\tRoute search (vans - multirouting part {r+1})...\n")

                # The network with costs between nodes (vans)
                csgraphVan = lil_matrix((nNodes, nNodes))
                csgraphVan[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])]  = np.array(MRDHlinks['COST_VAN'])
                csgraphVan[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))
                
                # Initialize a pool object that spreads tasks over different CPUs
                p = mp.Pool(nCPU)
                
                # Execute the Dijkstra route search
                prevVanPerCPU = p.map(functools.partial(get_prev, csgraphVan, nNodes), indicesPerCPU)
                
                # Wait for completion of processes
                p.close()
                p.join()
                    
                # Combine the results from the different CPUs
                prevVan.append(np.zeros((nOrigSelection,nNodes), dtype=int))
                for cpu in range(nCPU):
                    for i in range(len(indicesPerCPU[cpu][0])):
                        prevVan[r][origSelectionPerCPU[cpu][i], :] = prevVanPerCPU[cpu][i, :]
    
                # Make some space available on the RAM
                del prevVanPerCPU
                
                if root != '':
                    root.progressBar['value'] = 33.0 + (60.0 - 33.0) * (1 + r) / nMultiRoute
        else:
            for r in range(nMultiRoute):
                print(         f"\tRoute search (vans - multirouting part {r+1})...")
                log_file.write(f"\tRoute search (vans - multirouting part {r+1})...\n")

                # The network with costs between nodes (vans)
                csgraphVan = lil_matrix((nNodes, nNodes))
                csgraphVan[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])]  = np.array(MRDHlinks['COST_VAN'])
                csgraphVan[np.array(MRDHlinks['A']), np.array(MRDHlinks['B'])] *= (0.9 + 0.2 * np.random.rand(len(MRDHlinks)))
                
                # Execute the Dijkstra route search
                prevVan.append(get_prev(csgraphVan, nNodes, [indices, 0]))
             
                if root != '':
                    root.progressBar['value'] = 43.0 + (70.0 - 43.0) * (1 + r) / nMultiRoute
                    
        # Make some space available on the RAM
        del csgraphVan
        
        
                    
        # -------------------- Emissions and intensities (parcel vans) ------------------------------------ 
        
        print('\tParcel tours...'), log_file.write('\tParcels tours...\n')
        ls = 8 # Logistic segment: parcel deliveries
        
        for tod in timeOfDays:

            print('\t\tHour ' + str(tod+1) + ' of ' + str(nHours) + '...')
            
            # Nu de tod-specifieke tripmatrix in tripMatrix variabele zetten
            tripMatrixParcels        = tripMatricesParcelsTOD[tod]
            tripMatrixParcelsOrigins = set(tripMatrixParcels[:,0])
            
            for vt in vehTypesParcels:
                    
                # Selecteer de trips die vertrokken in de huidige time-of-day en bereken de capacity utilization
                trips = allParcelTrips.loc[(allParcelTrips['TRIP_DEPTIME'] >=tod  ) & \
                                           (allParcelTrips['TRIP_DEPTIME'] < tod+1) & \
                                           (allParcelTrips['VEHTYPE'     ] ==vt   ), :]
                if len(trips) > 0:
                    trips = np.array(trips[['Depot_ID','ORIG','DEST','VEHTYPE','CAP_UT','LOG_SEG','COMBTYPE','INDEX']])
                        
                    for i in range(nOrigSelection):
                        origZone = origSelection[i]
                        
                        if origZone in tripMatrixParcelsOrigins:
                            destZoneIndex = np.where(tripMatrixParcels[:,0]==origZone)[0]
                            
                            # Schrijf de volumes op de links
                            for j in destZoneIndex:
                                destZone    = tripMatrixParcels[j,1]
                                nTrips      = tripMatrixParcels[j,2]
                                route       = get_route(i, destZone, prevVan[0], zoneToCentroid, linkDict)
                                linkTripsArray[tod][route, ls]       += nTrips # Number of trips for LS8 (=parcel deliveries)
                                linkTripsArray[tod][route, nLS+2+vt] += nTrips # Number of trips for vehicle type
                                linkTripsArray[tod][route,-1]        += nTrips # Total number of trips

                                # Welke trips worden allemaal gemaakt op de HB van de huidige iteratie van de ij-loop
                                currentTrips = trips[(trips[:,1]==origZone) & (trips[:,2]==destZone), :]
                                
                                # De parcel demand trips per voertuigtype
                                MRDHlinksIntensities.loc[route, 'N_LS8_VEH' + str(vt)] += len(currentTrips)
        
                                if doSelectedLink:
                                    for l in range(nSelectedLinks):
                                        if selectedLinks[l] in route:
                                            selectedLinkTripsArray[route,l] += tripMatrixParcels[j,-1]
                                            
                                # Selecteer het deel van de route met linktype stad/buitenweg/snelweg
                                routeStad       = route[roadtypeArray[route]==1]
                                routeBuitenweg  = route[roadtypeArray[route]==2]
                                routeSnelweg    = route[roadtypeArray[route]==3]                           
                                ZEZstad      = ZEZarray[routeStad     ]==1
                                ZEZbuitenweg = ZEZarray[routeBuitenweg]==1                                
                                ZEZsnelweg   = ZEZarray[routeSnelweg  ]==1
                                
                                # Bereken en schrijf de emissies op de links                                                               
                                for trip in range(len(currentTrips)):
                                    ct      = int(currentTrips[trip,6])
                                    capUt   =     currentTrips[trip,4]
                                    
                                    # If combustion type is fuel, hybrid or bio-fuel                             
                                    if ct in [0,3,4]:
                                        # CO2
                                        stadCO2         = distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['CO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']]))
                                        buitenwegCO2    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['CO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']]))
                                        snelwegCO2      = distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['CO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']]))
                                        if ct == 3:
                                            stadCO2[ZEZstad          ] = 0
                                            buitenwegCO2[ZEZbuitenweg] = 0
                                            snelwegCO2[ZEZsnelweg    ] = 0
                                        linkEmissionsArray[ls][tod][routeStad,      etInvDict['CO2']] += stadCO2
                                        linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['CO2']] += buitenwegCO2
                                        linkEmissionsArray[ls][tod][routeSnelweg,   etInvDict['CO2']] += snelwegCO2
                    
                                        # SO2
                                        stadSO2         = distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['SO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']]))
                                        buitenwegSO2    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['SO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']]))
                                        snelwegSO2      = distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['SO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']]))
                                        if ct == 3:
                                            stadSO2[ZEZstad          ] = 0
                                            buitenwegSO2[ZEZbuitenweg] = 0
                                            snelwegSO2[ZEZsnelweg    ] = 0  
                                        linkEmissionsArray[ls][tod][routeStad,      etInvDict['SO2']] += stadSO2
                                        linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['SO2']] += buitenwegSO2
                                        linkEmissionsArray[ls][tod][routeSnelweg,   etInvDict['SO2']] += snelwegSO2
                                        
                                        # PM
                                        stadPM          = distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['PM']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['PM']] - emissionsStadLeeg[vtDict[vt],     etInvDict['PM']]))
                                        buitenwegPM     = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['PM']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']]))
                                        snelwegPM       = distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['PM']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']]))
                                        if ct == 3:
                                            stadPM[ZEZstad          ] = 0
                                            buitenwegPM[ZEZbuitenweg] = 0
                                            snelwegPM[ZEZsnelweg    ] = 0
                                        linkEmissionsArray[ls][tod][routeStad,      etInvDict['PM']] += stadPM
                                        linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['PM']] += buitenwegPM
                                        linkEmissionsArray[ls][tod][routeSnelweg,   etInvDict['PM']] += snelwegPM
                                        
                                        # NOX
                                        stadNOX         = distArray[routeStad     ]  * (emissionsStadLeeg[vtDict[vt],    etInvDict['NOX']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['NOX']] - emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']]))
                                        buitenwegNOX    = distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['NOX']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']]))
                                        snelwegNOX      = distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['NOX']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']]))
                                        if ct == 3:
                                            stadNOX[ZEZstad          ] = 0
                                            buitenwegNOX[ZEZbuitenweg] = 0
                                            snelwegNOX[ZEZsnelweg    ] = 0
                                        linkEmissionsArray[ls][tod][routeStad,      etInvDict['NOX']] += stadNOX
                                        linkEmissionsArray[ls][tod][routeBuitenweg, etInvDict['NOX']] += buitenwegNOX
                                        linkEmissionsArray[ls][tod][routeSnelweg,   etInvDict['NOX']] += snelwegNOX
    
                                        # CO2-emissions for the current trip
                                        parcelTripsCO2[currentTrips[trip,-1]] = (np.sum(stadCO2) + np.sum(buitenwegCO2) + np.sum(snelwegCO2))
        
            if root != '':
                root.progressBar['value'] = 70.0 + (75.0 - 70.0) * (tod + 1) / nHours
                    
                    
        # ------------------- Emissions and intensities (serv/constr vans) -------------------------------- 
        
        print('\tVan trips (service/construction)...'), log_file.write('\tVan trips (service/construction)...\n')
        for i in range(nOrigSelection):
            origZone = origSelection[i]
            
            # Van: Service segment
            for destZone in np.where(vanTripsService[origZone,:]>0)[0]:                
                for r in range(nMultiRoute):
                    nTrips = vanTripsService[origZone,destZone] / nMultiRoute
                    route  = get_route(i, destZone, prevVan[r], zoneToCentroid, linkDict)
                    
                    # Number of trips made on each link
                    linkVanTripsArray[route,0] += nTrips
                                    
                    routeStad       = route[roadtypeArray[route]==1]
                    routeBuitenweg  = route[roadtypeArray[route]==2]
                    routeSnelweg    = route[roadtypeArray[route]==3]
                    
                    vt    = 7   # Vehicle type: Van
                    capUt = 0.5 # Assume half of loading capacity used
    
                    # CO2
                    stadCO2         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['CO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']]))
                    buitenwegCO2    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['CO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']]))
                    snelwegCO2      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['CO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']]))
                    linkVanEmissionsArray[0][routeStad,      etInvDict['CO2']] += stadCO2
                    linkVanEmissionsArray[0][routeBuitenweg, etInvDict['CO2']] += buitenwegCO2
                    linkVanEmissionsArray[0][routeSnelweg,   etInvDict['CO2']] += snelwegCO2                            

                    # SO2
                    stadSO2         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['SO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']]))
                    buitenwegSO2    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['SO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']]))
                    snelwegSO2      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['SO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']]))
                    linkVanEmissionsArray[0][routeStad,      etInvDict['SO2']] += stadSO2
                    linkVanEmissionsArray[0][routeBuitenweg, etInvDict['SO2']] += buitenwegSO2
                    linkVanEmissionsArray[0][routeSnelweg,   etInvDict['SO2']] += snelwegSO2    

                    # PM
                    stadPM         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['PM']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['PM']] - emissionsStadLeeg[vtDict[vt],     etInvDict['PM']]))
                    buitenwegPM    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['PM']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']]))
                    snelwegPM      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['PM']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']]))
                    linkVanEmissionsArray[0][routeStad,      etInvDict['PM']] += stadPM
                    linkVanEmissionsArray[0][routeBuitenweg, etInvDict['PM']] += buitenwegPM
                    linkVanEmissionsArray[0][routeSnelweg,   etInvDict['PM']] += snelwegPM  

                    # NOX
                    stadNOX         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['NOX']] - emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']]))
                    buitenwegNOX    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['NOX']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']]))
                    snelwegNOX      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['NOX']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']]))
                    linkVanEmissionsArray[0][routeStad,      etInvDict['NOX']] += stadNOX
                    linkVanEmissionsArray[0][routeBuitenweg, etInvDict['NOX']] += buitenwegNOX
                    linkVanEmissionsArray[0][routeSnelweg,   etInvDict['NOX']] += snelwegNOX  
                    
            # Van: Construction segment
            for destZone in np.where(vanTripsConstruction[origZone,:]>0)[0]:                
                for r in range(nMultiRoute):
                    nTrips = vanTripsConstruction[origZone,destZone] / nMultiRoute
                    route  = get_route(i, destZone, prevVan[r], zoneToCentroid, linkDict)

                    # Number of trips made on each link
                    linkVanTripsArray[route,1] += nTrips
                    
                    routeStad       = route[roadtypeArray[route]==1]
                    routeBuitenweg  = route[roadtypeArray[route]==2]
                    routeSnelweg    = route[roadtypeArray[route]==3]
                    
                    vt    = 7   # Vehicle type: Van
                    capUt = 0.5 # Assume half of loading capacity used
    
                    # CO2
                    stadCO2         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['CO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['CO2']]))
                    buitenwegCO2    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['CO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['CO2']]))
                    snelwegCO2      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['CO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['CO2']]))
                    linkVanEmissionsArray[1][routeStad,      etInvDict['CO2']] += stadCO2
                    linkVanEmissionsArray[1][routeBuitenweg, etInvDict['CO2']] += buitenwegCO2
                    linkVanEmissionsArray[1][routeSnelweg,   etInvDict['CO2']] += snelwegCO2                            

                    # SO2
                    stadSO2         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['SO2']] - emissionsStadLeeg[vtDict[vt],     etInvDict['SO2']]))
                    buitenwegSO2    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['SO2']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['SO2']]))
                    snelwegSO2      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['SO2']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['SO2']]))
                    linkVanEmissionsArray[1][routeStad,      etInvDict['SO2']] += stadSO2
                    linkVanEmissionsArray[1][routeBuitenweg, etInvDict['SO2']] += buitenwegSO2
                    linkVanEmissionsArray[1][routeSnelweg,   etInvDict['SO2']] += snelwegSO2    

                    # PM
                    stadPM         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['PM']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['PM']] - emissionsStadLeeg[vtDict[vt],     etInvDict['PM']]))
                    buitenwegPM    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['PM']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['PM']]))
                    snelwegPM      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['PM']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['PM']]))
                    linkVanEmissionsArray[1][routeStad,      etInvDict['PM']] += stadPM
                    linkVanEmissionsArray[1][routeBuitenweg, etInvDict['PM']] += buitenwegPM
                    linkVanEmissionsArray[1][routeSnelweg,   etInvDict['PM']] += snelwegPM  

                    # NOX
                    stadNOX         = nTrips * distArray[routeStad     ] * (emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']] + capUt * (emissionsStadVol[vtDict[vt],     etInvDict['NOX']] - emissionsStadLeeg[vtDict[vt],     etInvDict['NOX']]))
                    buitenwegNOX    = nTrips * distArray[routeBuitenweg] * (emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']] + capUt * (emissionsBuitenwegVol[vtDict[vt],etInvDict['NOX']] - emissionsBuitenwegLeeg[vtDict[vt],etInvDict['NOX']]))
                    snelwegNOX      = nTrips * distArray[routeSnelweg  ] * (emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']] + capUt * (emissionsSnelwegVol[vtDict[vt],  etInvDict['NOX']] - emissionsSnelwegLeeg[vtDict[vt],  etInvDict['NOX']]))
                    linkVanEmissionsArray[1][routeStad,      etInvDict['NOX']] += stadNOX
                    linkVanEmissionsArray[1][routeBuitenweg, etInvDict['NOX']] += buitenwegNOX
                    linkVanEmissionsArray[1][routeSnelweg,   etInvDict['NOX']] += snelwegNOX  

            if i%int(round(nOrigSelection/10,0)) == 0:
                print('\t\t' + str(int(round((i / nOrigSelection)*100, 0))) + '%')    

            if root != '':
                root.progressBar['value'] = 75.0 + (85.0 - 75.0) * (i + 1) / nOrigSelection
                               
        # Make some space available on the RAM
        del prevVan, vanTripsService, vanTripsConstruction
        
        # Assume all electric for vans in UCC scenario
        if label == 'UCC':
            linkVanEmissionsArray = [np.zeros((len(MRDHlinks),nET)) for ls in ['VAN_S','VAN_C']]            
            
        # Write the intensities and emissions into the links-DataFrames
        for tod in timeOfDays:
            # The DataFrame to be exported to GeoJSON
            MRDHlinks.loc[:, volCols] += linkTripsArray[tod].astype(int)
            
            # The DataFrame to be exported to CSV
            MRDHlinksIntensities.loc[:,volCols      ] += linkTripsArray[tod].astype(int)
            MRDHlinksIntensities.loc[:,f'N_TOD{tod}'] += linkTripsArray[tod][:,-1].astype(int)
            
            # Total emissions and per logistic segment
            for ls in range(nLS):
                for et in range(nET):
                    # The DataFrame to be exported to GeoJSON
                    MRDHlinks[etDict[et]] += (linkEmissionsArray[ls][tod][:,et] / emissionDivFac[et])
                    
                    # The DataFrame to be exported to CSV
                    MRDHlinksIntensities[etDict[et]                  ] += (linkEmissionsArray[ls][tod][:,et] / emissionDivFac[et])
                    MRDHlinksIntensities[etDict[et] + '_LS' + str(ls)] += (linkEmissionsArray[ls][tod][:,et] / emissionDivFac[et])

        if root != '':
            root.progressBar['value'] = 87.0
                
        # Number of van trips
        linkVanTripsArray = np.round(linkVanTripsArray, 3)
        MRDHlinks.loc[:,'N_VAN_S'] = linkVanTripsArray[:,0]
        MRDHlinks.loc[:,'N_VAN_C'] = linkVanTripsArray[:,1]
        MRDHlinks.loc[:,'N_VEH7'] += linkVanTripsArray[:,0]
        MRDHlinks.loc[:,'N_VEH7'] += linkVanTripsArray[:,1]
        MRDHlinks.loc[:,'N_TOT'] += linkVanTripsArray[:,0]
        MRDHlinks.loc[:,'N_TOT'] += linkVanTripsArray[:,1]
        MRDHlinksIntensities.loc[:,'N_VAN_S'] = linkVanTripsArray[:,0]
        MRDHlinksIntensities.loc[:,'N_VAN_C'] = linkVanTripsArray[:,1]
        MRDHlinksIntensities.loc[:,'N_VEH7'] += linkVanTripsArray[:,0]
        MRDHlinksIntensities.loc[:,'N_VEH7'] += linkVanTripsArray[:,1]
        MRDHlinksIntensities.loc[:,'N_TOT'] += linkVanTripsArray[:,0]
        MRDHlinksIntensities.loc[:,'N_TOT'] += linkVanTripsArray[:,1]
        
        # Emissions from van trips
        for et in range(nET):
            MRDHlinksIntensities[etDict[et] + '_' + 'VAN_S'] = (linkVanEmissionsArray[0][:,et] / emissionDivFac[et])
            MRDHlinksIntensities[etDict[et] + '_' + 'VAN_C'] = (linkVanEmissionsArray[1][:,et] / emissionDivFac[et])
            MRDHlinks[etDict[et]] += (linkVanEmissionsArray[0][:,et] / emissionDivFac[et])
            MRDHlinks[etDict[et]] += (linkVanEmissionsArray[1][:,et] / emissionDivFac[et])
            MRDHlinksIntensities[etDict[et]] += (linkVanEmissionsArray[0][:,et] / emissionDivFac[et])
            MRDHlinksIntensities[etDict[et]] += (linkVanEmissionsArray[1][:,et] / emissionDivFac[et])

        if root != '':
            root.progressBar['value'] = 90.0
                        
        print('Writing link intensities to CSV...'), log_file.write('Writing link intensities to CSV...' + '\n')
        MRDHlinks['A'] = [nodeDict[x] for x in MRDHlinks['A']]
        MRDHlinks['B'] = [nodeDict[x] for x in MRDHlinks['B']]
        MRDHlinksIntensities['LINKNR'    ] = MRDHlinks['LINKNR']
        MRDHlinksIntensities['A'         ] = MRDHlinks['A']
        MRDHlinksIntensities['B'         ] = MRDHlinks['B']
        MRDHlinksIntensities['LENGTH'    ] = MRDHlinks['LENGTH']
        MRDHlinksIntensities['ZEZ'       ] = MRDHlinks['ZEZ']
        MRDHlinksIntensities['Gemeentena'] = MRDHlinks['Gemeentena']
        cols = ['LINKNR','A','B','LENGTH','ZEZ','Gemeentena']
        for col in intensityFields:
            cols.append(col)
        MRDHlinksIntensities = MRDHlinksIntensities[cols]
        MRDHlinksIntensities.to_csv(datapathO + 'links_loaded_' + str(label) + '_intensities.csv', index=False)

        if doSelectedLink:
            print('Writing selected link analysis to CSV...'), log_file.write('Writing selected link analysis to CSV...' + '\n')
            selectedLinkTripsArray = pd.DataFrame(selectedLinkTripsArray, columns=['N_' + str(selectedLinks[l]) for l in range(nSelectedLinks)])
            selectedLinkTripsArray['LINKNR'] = MRDHlinks['LINKNR']
            selectedLinkTripsArray['A'     ] = MRDHlinks['A']
            selectedLinkTripsArray['B'     ] = MRDHlinks['B']
            cols = ['LINKNR','A','B']
            for l in range(nSelectedLinks):
                cols.append('N_' + str(selectedLinks[l]))
            selectedLinkTripsArray = selectedLinkTripsArray[cols]
            selectedLinkTripsArray.to_csv(datapathO + 'SelectedLinks.csv', sep=',', index=False)
            
        # Make some space available on the RAM
        del linkTripsArray, linkEmissionsArray
        del linkVanTripsArray, linkVanEmissionsArray
        
        if root != '':
            root.progressBar['value'] = 93.0
            
            
        # ----------------------- Enriching tours and shipments -------------------        
        try:
            print("Writing emissions into Tours and ParcelSchedule..."), log_file.write("Writing emissions into Tours and ParcelSchedule...\n")
            tours            = pd.read_csv(datapathO + 'Tours_' + label + '.csv')
            tours['TOUR_ID'] = [str(tours.at[i,'CARRIER_ID']) + '_' + str(tours.at[i,'TOUR_ID']) for i in tours.index]
            tours['CO2'    ] = [tripsCO2[i] for i in tours.index]
            tours.to_csv(datapathO + 'Tours_' + label + '.csv', index=False)
        
            parcelTours        = pd.read_csv(datapathO + 'ParcelSchedule_' + label + '.csv')
            parcelTours['CO2'] = [parcelTripsCO2[i] for i in parcelTours.index]
            parcelTours.to_csv(datapathO + 'ParcelSchedule_' + label + '.csv', index=False)

            print("Writing emissions into Shipments..."), log_file.write("Writing emissions into Shipments...\n")        
            # Calculate emissions at the tour level instead of trip level
            toursCO2 = pd.pivot_table(tours, values=['CO2'], index=['TOUR_ID'], aggfunc=np.sum)
            tourIDDict = dict(np.transpose(np.vstack((toursCO2.index, np.arange(len(toursCO2))))))
            toursCO2 = np.array(toursCO2['CO2'])
            
            # Read the shipments
            shipments = pd.read_csv(datapathO + 'Shipments_AfterScheduling_' + label + '.csv')
            shipments['ORIG'] = [invZoneDict[x] for x in shipments['ORIG']]
            shipments['DEST'] = [invZoneDict[x] for x in shipments['DEST']]
            shipments = shipments.sort_values('TOUR_ID')
            shipments.index = np.arange(len(shipments))
            
            # For each tour, which shipments belong to it
            tourIDs = [tourIDDict[x] for x in shipments['TOUR_ID']]
            shipIDs = []
            currentShipIDs = [0]
            for i in range(1,len(shipments)):
                if tourIDs[i-1] == tourIDs[i]:
                    currentShipIDs.append(i)
                else:
                    shipIDs.append(currentShipIDs.copy())
                    currentShipIDs = [i]
            shipIDs.append(currentShipIDs.copy())
                
            # Network distance of each shipment
            skimDistance = read_mtx(skimDistancePath)
            shipDist = skimDistance[(shipments['ORIG'] - 1) * nZones + (shipments['DEST'] - 1)]            
            
            # Divide CO2 of each tour over its shipments based on distance
            shipCO2  = np.zeros(len(shipments))
            for tourID in np.unique(tourIDs):
                currentDists = shipDist[shipIDs[tourID]]
                currentCO2   = toursCO2[tourID]
                if np.sum(currentDists) == 0:
                    shipCO2[shipIDs[tourID]] = 0
                else:                        
                    shipCO2[shipIDs[tourID]] = currentDists / np.sum(currentDists) * currentCO2
            shipments['CO2'] = shipCO2
            
            # Export enriched shipments with CO2 field
            shipments = shipments.sort_values('SHIP_ID')
            shipments.index = np.arange(len(shipments))            
            shipments['ORIG'] = [zoneDict[x] for x in shipments['ORIG']]
            shipments['DEST'] = [zoneDict[x] for x in shipments['DEST']]            
            shipments.to_csv(datapathO + 'Shipments_AfterScheduling_' + label + '.csv', index=False)
            
        except:
            print("Writing emissions into Tours/ParcelSchedule/Shipments failed!")
            log_file.write("Writing emissions into Tours/ParcelSchedule/Shipments failed!" + '\n')
            try:
                import sys
                print(sys.exc_info()[0]), log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
                import traceback
                print(traceback.format_exc()), log_file.write(str(traceback.format_exc())), log_file.write("\n")
            except:
                pass

        if root != '':
            root.progressBar['value'] = 95.0
            
            
        # ----------------------- Export loaded network to shapefile -------------------
        if exportShp:
            print("Exporting network to .shp..."), log_file.write("Exporting network to .shp...\n")            
            # Set travel times of connectors at 0 for in the output network shape
            MRDHlinks.loc[MRDHlinks['WEGTYPE']=='voedingslink','T0'] = 0
            
            # Afronden van sommige kolommen met overdreven veel precisie
            MRDHlinks[intensityFieldsGeojson] = np.round(MRDHlinks[intensityFieldsGeojson], 5)
            
            MRDHlinks['Gemeentena'] = MRDHlinks['Gemeentena'].astype(str)
            MRDHlinks['Gemeentena'] = [x.replace("'","") for x in MRDHlinks['Gemeentena']]
            
            # Vervang NaN's
            MRDHlinks.loc[pd.isna(MRDHlinks['ZEZ'  ]), 'ZEZ'  ] = 0.0
            MRDHlinks.loc[pd.isna(MRDHlinks['LANES']), 'LANES'] = -99999
            
            MRDHlinks = MRDHlinks.drop(columns='NAME')
            
            # Initialize shapefile fields
            w = shp.Writer(datapathO + f'links_loaded_{label}.shp')
            w.field('LINKNR',     'N', size=8, decimal=0)
            w.field('A',          'N', size=9, decimal=0)
            w.field('B',          'N', size=9, decimal=0)
            w.field('LENGTH'      'N', size=7, decimal=3)
            w.field('LANES',      'N', size=6, decimal=0)
            w.field('CAPACITY',   'N', size=6, decimal=0)
            w.field('WEGTYPE',    'C')
            w.field('COUNT_FR',   'N', size=6, decimal=0)
            w.field('V0_PA_OS',   'N', size=6, decimal=0)
            w.field('V0_PA_RD',   'N', size=6, decimal=0)
            w.field('V0_PA_AS',   'N', size=6, decimal=0)
            w.field('V0_FR_OS',   'N', size=6, decimal=0)
            w.field('V0_FR_RD',   'N', size=6, decimal=0)
            w.field('V0_FR_AS',   'N', size=6, decimal=0)
            w.field('V_PA_OS',    'N', size=6, decimal=0)
            w.field('V_PA_RD',    'N', size=6, decimal=0)
            w.field('V_PA_AS',    'N', size=6, decimal=0)
            w.field('V_FR_OS',    'N', size=6, decimal=0)
            w.field('V_FR_RD',    'N', size=6, decimal=0)
            w.field('V_FR_AS',    'N', size=6, decimal=0)            
            w.field('ZEZ',        'N', size=1, decimal=0)    
            w.field('Gemeentena', 'C')
            w.field('T0_FREIGHT',  'N', size=8, decimal=5)
            w.field('T0_VAN',      'N', size=8, decimal=5)
            w.field('COST_FREIGHT','N', size=8, decimal=5)
            w.field('COST_VAN',    'N', size=8, decimal=5)
            for field in intensityFieldsGeojson[:4]:
                w.field(field, 'N', size=9, decimal=5)
            for field in intensityFieldsGeojson[4:]:
                w.field(field, 'N', size=6, decimal=1)

            dbfData = np.array(MRDHlinks, dtype=object)
            for i in range(nLinks):
                # Add geometry
                geom = MRDHlinksGeometry[i]['coordinates']
                line = []
                for l in range(len(geom)-1):
                    line.append([[geom[l][0],geom[l][1]],[geom[l+1][0],geom[l+1][1]]])
                w.line(line)
                
                # Add data fields
                w.record(*dbfData[i,:])
                                
                if i%int(round(nLinks/10,0)) == 0:
                    print('\t' + str(int(round((i / nLinks)*100, 0))) + '%', end='\r')    

                    if root != '':
                        root.progressBar['value'] = 95.0 + (100.0 - 95.0) * i / nLinks
                        
            w.close()


        # --------------------------- End of module ---------------------------
                
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Traffic Assignment: Done")
            root.progressBar['value'] = 100
            
            # 0 means no errors in execution
            root.returnInfo = [0, [0,0]]
            
            return root.returnInfo
        
        else:
            return [0, [0,0]]
            
        
    except BaseException:
        import sys
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        import traceback
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()
        
        if root != '':
            # Use this information to display as error message in GUI
            root.returnInfo = [1, [sys.exc_info()[0], traceback.format_exc()]]
            
            if __name__ == '__main__':
                root.update_statusbar("Traffic Assignment: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]
        
        

#%% Other functions
            
def get_prev(csgraph, nNodes, indices):
    '''
    For each origin zone and destination node, determine the previously visited node on the shortest path.
    '''
    whichCPU = indices[1]
    indices  = indices[0]
    nOrigSelection = len(indices)
    
    prev = np.zeros((nOrigSelection,nNodes), dtype=int)        
    for i in range(nOrigSelection):
        prev[i,:] = scipy.sparse.csgraph.dijkstra(csgraph, 
                                                  indices=indices[i], 
                                                  return_predecessors=True)[1]
        
        if whichCPU == 0:
            if i%int(round(nOrigSelection/10,0)) == 0:
                print('\t\t' + str(int(round((i / nOrigSelection)*100, 0))) + '%')          
    
    return prev



def get_route(i, j, prev, zoneToCentroid, linkDict):
    '''
    Deduce the paths from the prev object
    '''                 
    sequenceNodes = []
    sequenceLinks = []
    
    # Deduce sequence of nodes on network
    destNode = zoneToCentroid[j]
    if prev[i][destNode] >= 0:
        while prev[i][destNode]>=0:
            sequenceNodes.insert(0,destNode)
            destNode = prev[i][destNode]
        else:
            sequenceNodes.insert(0,destNode)  
           
    # Convert to sequence of links
    if i != j:
        if len(sequenceNodes) > 1:
            sequenceLinks = np.array([linkDict[sequenceNodes[w]][sequenceNodes[w+1]] for w in range(len(sequenceNodes)-1)], dtype=int)
        else:
            print(f'\tWarning! No path found for OD {i}-{j}')
            sequenceLinks = np.array([], dtype=int)            
    else:
        sequenceLinks = np.array([], dtype=int)
    
    return sequenceLinks

       
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    INPUTFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2016/'
    OUTPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunREF2016/'
    PARAMFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/'
    
    SKIMTIME            = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimTijd_REF.mtx'
    SKIMDISTANCE        = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimAfstand_REF.mtx'
    LINKS		        = INPUTFOLDER + 'links_v5.shp'
    NODES               = INPUTFOLDER + 'nodes_v5.shp'
    ZONES               = INPUTFOLDER + 'Zones_v4.shp'
    SEGS                = INPUTFOLDER + 'SEGS2016.csv'
    COMMODITYMATRIX     = INPUTFOLDER + 'CommodityMatrixNUTS3_2016.csv'
    PARCELNODES         = INPUTFOLDER + 'parcelNodes_v2.shp'
    DISTRIBUTIECENTRA   = INPUTFOLDER + 'distributieCentra.csv'
    COST_VEHTYPE        = PARAMFOLDER + 'Cost_VehType_2016.csv'
    COST_SOURCING       = PARAMFOLDER + 'Cost_Sourcing_2016.csv'
    MRDH_TO_NUTS3       = PARAMFOLDER + 'MRDHtoNUTS32013.csv'
    NUTS3_TO_MRDH       = PARAMFOLDER + 'NUTS32013toMRDH.csv'
    
    YEARFACTOR = 209
    
    NUTSLEVEL_INPUT = 3
    
    PARCELS_PER_HH	 = 0.112
    PARCELS_PER_EMPL = 0.041
    PARCELS_MAXLOAD	 = 180
    PARCELS_DROPTIME = 120
    PARCELS_SUCCESS_B2C   = 0.75
    PARCELS_SUCCESS_B2B   = 0.95
    PARCELS_GROWTHFREIGHT = 1.0
    
    SHIPMENTS_REF = ""
    SELECTED_LINKS = ""
    N_CPU = ""
    N_MULTIROUTE = 2
    
    IMPEDANCE_SPEED_FREIGHT = 'V_FR_OS'
    IMPEDANCE_SPEED_VAN     = 'V_PA_OS'
    
    LABEL = 'REF'
    
    MODULES = ['FS', 'SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, \
            LINKS, NODES, ZONES, SEGS, \
            DISTRIBUTIECENTRA, COST_VEHTYPE,COST_SOURCING,\
            COMMODITYMATRIX, PARCELNODES, MRDH_TO_NUTS3, NUTS3_TO_MRDH, \
            PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
            PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED_FREIGHT, IMPEDANCE_SPEED_VAN, N_CPU, N_MULTIROUTE, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", \
                  "LINKS", "NODES", "ZONES", "SEGS", \
                  "DISTRIBUTIECENTRA", "COST_VEHTYPE","COST_SOURCING", \
                  "COMMODITYMATRIX", "PARCELNODES", "MRDH_TO_NUTS3", "NUTS3_TO_MRDH", \
                  "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
                  "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B",  "PARCELS_GROWTHFREIGHT", \
                  "YEARFACTOR", "NUTSLEVEL_INPUT", \
                  "IMPEDANCE_SPEED_FREIGHT", "IMPEDANCE_SPEED_VAN", "N_CPU", "N_MULTIROUTE", \
                  "SHIPMENTS_REF", "SELECTED_LINKS", \
                  "LABEL", \
                  "MODULES"]
     
    varDict = {}
    for i in range(len(args)):
        varDict[varStrings[i]] = args[i]
        
    # Run the module
    main(varDict)



     
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

