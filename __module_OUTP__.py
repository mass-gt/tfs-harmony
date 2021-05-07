import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import os.path
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
        self.root.title("Progress Output Indicators")
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
    
    try:
        
        global trips, tours, shipments, parcels, parcelTrips, linksLoaded, zonesShape
        
        #------------------- Open output files --------------------------------
        
        print("Importing data...")
        start_time = time.time()
        
        root    = args[0]
        varDict = args[1]
        
        if root != '':
            root.progressBar['value'] = 0
        
        # Define folders relative to current datapath
        datapathI = varDict['INPUTFOLDER']
        datapathO = varDict['OUTPUTFOLDER']
        datapathP = varDict['PARAMFOLDER']
        zonesPath        = varDict['ZONES']
        skimTravTimePath = varDict['SKIMTIME']
        skimDistancePath = varDict['SKIMDISTANCE']  
        label            = varDict['LABEL']
        
        datapathOI = datapathO + 'Output indicators/'
        
        log_file = open(f"{datapathO}Logfile_OutputIndicator.log", "w")
        log_file.write("Start simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.write("Importing data...\n")        
        
        if not os.path.isdir(datapathOI) and os.path.isdir(datapathO):
            os.mkdir(datapathOI)

        outfile = open(f"{datapathOI}Output_OutputIndicator_{label}.csv", "w")
        sep = ','

        
        #------------------- Import data --------------------------------------
        vtNums = np.arange(10)
        vtNames = ['Truck (small)', 'Truck (medium)', 'Truck (large)', 'Truck+trailer (small)', 'Truck+trailer (large)', 'Tractor+trailer', 'Special vehicle', 'Van', 'LEVV', 'Moped']       
        
        lsNames = ['Food (general cargo)', 'Miscellaneous (general cargo)', 'Temperature controlled', \
                   'Facility logistics', 'Construction logistics', 'Waste', 'Parcel (consolidated flows)', \
                   'Dangerous', 'Parcel (deliveries)']
                
        nVT = len(vtNames)         
        nLS = len(lsNames)
        nNSTR = 10
        
        # Read data
        print('\tImporting shipments...'), log_file.write('\tImporting shipments...\n')
        shipments   = pd.read_csv(f"{datapathO}Shipments_{label}.csv", index_col=None)

        print('\tImporting tours...'), log_file.write('\tImporting tours...\n')
        trips       = pd.read_csv(f'{datapathO}Tours_{label}.csv')

        print('\tImporting parcels...'), log_file.write('\tImporting parcels...\n')
        parcels     = pd.read_csv(f"{datapathO}ParcelDemand_{label}.csv")
        parcelTrips = pd.read_csv(f"{datapathO}ParcelSchedule_{label}.csv")
        
        parcels     = parcels.rename(columns={'O_zone':'ORIG', 'D_zone':'DEST'})
        parcelTrips = parcelTrips.rename(columns={'O_zone':'ORIG', 'D_zone':'DEST','Trip_ID':'TRIP_ID', 'Tour_ID':'TOUR_ID'})  

        print('\tPreparing tours/parcels...'), log_file.write('\tPreparing tours/parcels...\n')
        temp = pd.pivot_table(parcelTrips, values='TRIP_ID', index='TOUR_ID', aggfunc=len)
        nTripsParcelTour = {}
        for i in temp.index:
            nTripsParcelTour[i] = temp.at[i,'TRIP_ID']
        parcelTrips['N_trips_tour'] = [nTripsParcelTour[tourID] for tourID in parcelTrips['TOUR_ID']]
        
        tours       = trips[[trips['TRIP_ID'][i][-2:]=='_0' for i in trips.index]]
        parcelTours = parcelTrips[[parcelTrips['TRIP_ID'][i][-2:]=='_0' for i in parcelTrips.index]]
        
        print('\tImporting van trips...'), log_file.write('\tImporting van trips...\n')
        if os.path.isfile(datapathO + 'TripsVanService.mtx') and os.path.isfile(datapathO + 'TripsVanConstruction.mtx'):
            vanTripsFound = True
            vanTripsService      = read_mtx(datapathO + 'TripsVanService.mtx')
            vanTripsConstruction = read_mtx(datapathO + 'TripsVanConstruction.mtx')            
        else:
            vanTripsFound = False
            print('\tVan trips not found in outputfolder...'), log_file.write('\tVan trips not found in outputfolder...\n')
            
        # Skim with travel times and distances
        print('\tImporting skims...'), log_file.write('\tImporting skims...\n')
        skimTravTime = read_mtx(skimTravTimePath)
        skimDistance = read_mtx(skimDistancePath)
        nZones       = int(len(skimTravTime)**0.5)
        
        # Import external zones demand and coordinates
        superCoordinates = pd.read_csv(f'{datapathI}SupCoordinatesID.csv')
        nSuperZones      = len(superCoordinates)          
        
        # Import zonal data
        print('\tImporting zonal data...'), log_file.write('\tImporting zonal data...\n')
        zonesShape      = read_shape(zonesPath)
        zonesShape.sort_values('AREANR')
        zonesShape.index = zonesShape['AREANR']
        zoneID          = np.array(zonesShape['AREANR'])
        nInternalZones  = len(zonesShape)
        zoneDict        = dict(np.transpose(np.vstack((np.arange(nInternalZones),zoneID))))
        for i in range(nSuperZones):
            zoneDict[nInternalZones+i] = superCoordinates['AREANR'][i]
        invZoneDict     = dict((v, k) for k, v in zoneDict.items())
        zoneID          = np.arange(nInternalZones)      

        # The distance of each shipment and trip in kilometers
        print('\tCalculating distance of each trip...'), log_file.write('\tCalculating distance of each trip...\n')
        trips['DIST']       = skimDistance[np.array([invZoneDict[trips['ORIG'][i]]       * nZones + invZoneDict[trips['DEST'][i]]       for i in       trips.index], dtype=int)] / 1000
        parcelTrips['DIST'] = skimDistance[np.array([invZoneDict[parcelTrips['ORIG'][i]] * nZones + invZoneDict[parcelTrips['DEST'][i]] for i in parcelTrips.index], dtype=int)] / 1000
        shipments['DIST']   = skimDistance[np.array([invZoneDict[shipments['ORIG'][i]]   * nZones + invZoneDict[shipments['DEST'][i]]   for i in   shipments.index], dtype=int)] / 1000
        
        # Import the loaded network
        print('\tImporting loaded network...'), log_file.write('\tImporting loaded network...\n')        
        linksLoaded = pd.read_csv(datapathO + 'links_loaded_' + str(label) + '_intensities.csv')
        intCols = ['LINKNR','A','B','ZEZ',\
                   'N_LS0', 'N_LS1', 'N_LS2', 'N_LS3', 'N_LS4', \
                   'N_LS5', 'N_LS6', 'N_LS7', 'N_VEH0', 'N_VEH1', 'N_VEH2', 'N_VEH3', \
                   'N_VEH4', 'N_VEH5', 'N_VEH6', 'N_VEH7', 'N_VEH8', 'N_VEH9', 'N_TOT']
        linksLoaded[intCols] = linksLoaded[intCols].astype(int)
        linksLoaded['Gemeentena'] = linksLoaded['Gemeentena'].astype(str)
        linksLoadedRdam = linksLoaded.loc[linksLoaded['Gemeentena']=='Rotterdam',    :]
        linksLoadedDH   = linksLoaded.loc[linksLoaded['Gemeentena']=="'s-Gravenhage",:]
        linksLoadedZH   = linksLoaded.loc[linksLoaded['Gemeentena']!="nan",          :]
        linksLoadedZEZ  = linksLoaded.loc[linksLoaded['ZEZ'       ]==1,              :]

        
        #---------------------- Shipment sizes --------------------------------
        
        print("Calculating and exporting output indicators..."), log_file.write("Calculating and exporting output indicators...\n")
        
        print("\tShipments"), log_file.write("\tShipments\n")
        
        # Actual shipment sizes
        shipments['WEIGHT_LEVELS'] = 0
        shipments.loc[(shipments['WEIGHT']< 3 )                           ,'WEIGHT_LEVELS'] = 0
        shipments.loc[(shipments['WEIGHT']>=3 ) & (shipments['WEIGHT']<6 ),'WEIGHT_LEVELS'] = 1
        shipments.loc[(shipments['WEIGHT']>=6 ) & (shipments['WEIGHT']<10),'WEIGHT_LEVELS'] = 2        
        shipments.loc[(shipments['WEIGHT']>=10) & (shipments['WEIGHT']<20),'WEIGHT_LEVELS'] = 3 
        shipments.loc[(shipments['WEIGHT']>=20) & (shipments['WEIGHT']<30),'WEIGHT_LEVELS'] = 4
        shipments.loc[(shipments['WEIGHT']>=30)                           ,'WEIGHT_LEVELS'] = 5
        fig = plt.figure()
        ax = fig.add_axes([0,0,2,2])
        shipmentSizeHist = np.unique(shipments['WEIGHT_LEVELS'], return_counts=True)
        ax.bar(np.arange(6), shipmentSizeHist[1])
        xticklabels = ['','<3','3--6','6--10','10--20','20--30','>30']
        ax.set_xticklabels(xticklabels)
        ax.set_title('Shipment Size freight')
        ax.set_xlabel('Shipment size (freight; tonnes)')
        ax.set_ylabel('Number of shipments')
        fig.savefig(datapathOI + f"OUT_ShipmentSizes_{label}.png", bbox_inches='tight')
        plt.close()
        
        outfile.write('Shipment size; actual weight (freight)' + sep + '\n')
        outfile.write(sep + 'Weight (tonnes)' + sep + 'Number of shipments\n')
        for i in range(6):
            outfile.write(sep + xticklabels[i+1] + sep + str(shipmentSizeHist[1][i]) + '\n')        

        # Chosen shipment size category
        shipmentSizeHist = np.unique(shipments['WEIGHT_CAT'], return_counts=True)        
        outfile.write('\nShipment size; chosen weight class (freight)' + sep + '\n')
        outfile.write(sep + 'Weight (tonnes)' + sep + 'Number of shipments\n')
        for i in range(6):
            outfile.write(sep + xticklabels[i+1] + sep + str(shipmentSizeHist[1][i]) + '\n')       

        # Chosen vehicle type per shipment
        nShipsVeh = np.zeros((nVT,1))[:,0]
        outfile.write('\nNumber of shipments (freight; by vehicle type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type' + sep + 'Number of shipments\n')
        for veh in range(nVT):
            nShipsVeh[veh] = len(shipments[shipments['VEHTYPE']==veh])
            outfile.write(sep + vtNames[veh] + sep + str(nShipsVeh[veh]) + '\n')
            
        # Number of shipments by logistic segment
        nShipsLS = np.zeros((nLS,1))[:,0]
        outfile.write('\nNumber of shipments (Total) (freight/parcels; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Number of shipments\n')
        for ls in range(nLS-1):
            nShipsLS[ls] = len(shipments[shipments['LOGSEG']==ls])
            outfile.write(sep + lsNames[ls] + sep + str(nShipsLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS-1] + sep + str(len(parcels)) + '\n')

        # Number of shipments by logistic segment
        ZEZshipments = shipments[shipments['DEST']<99999900]
        ZEZshipments = ZEZshipments[np.array(zonesShape.loc[np.array(ZEZshipments['DEST'], dtype=int), 'ZEZ']==1)]
        ZEZparcels   = parcels[parcels['DEST']<99999900]
        ZEZparcels   = ZEZparcels[np.array(zonesShape.loc[np.array(ZEZparcels['DEST'], dtype=int), 'ZEZ']==1)]
        nShipsLS = np.zeros((nLS,1))[:,0]
        outfile.write('\nNumber of shipments (to ZEZ) (freight/parcels; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Number of shipments\n')
        for ls in range(nLS-1):
            nShipsLS[ls] = len(ZEZshipments[ZEZshipments['LOGSEG']==ls])
            outfile.write(sep + lsNames[ls] + sep + str(nShipsLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS-1] + sep + str(len(ZEZparcels)) + '\n')
        
            
            
        # -------------------------- Number of trips --------------------------
                
        print("\tTrips"), log_file.write("\tTrips\n")
        
        # Number of trips by direction
        nTripsTotal    = trips.shape[0]
        nTripsLeaving  = trips[trips['DEST']>99999900].shape[0]
        nTripsEntering = trips[trips['ORIG']>99999900].shape[0]
        nTripsIntra = nTripsTotal - nTripsLeaving - nTripsEntering
        
        outfile.write('\nNumber of trips (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Number of trips'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(nTripsIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(nTripsLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(nTripsEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(nTripsTotal)    + '\n')

        # Number of parcel trips by direction
        nParcelTripsTotal    = parcelTrips.shape[0]
        nParcelTripsLeaving  = parcelTrips[parcelTrips['DEST']>99999900].shape[0]
        nParcelTripsEntering = parcelTrips[parcelTrips['ORIG']>99999900].shape[0]
        nParcelTripsIntra = nParcelTripsTotal - nParcelTripsLeaving - nParcelTripsEntering

        outfile.write('\nNumber of trips (parcel; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Number of trips'         + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(nParcelTripsIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(nParcelTripsLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(nParcelTripsEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(nParcelTripsTotal)    + '\n')
        
        # Number of trips by logistic segment and NSTR
        outfile.write('\nNumber of trips (freight/parcel; by logistic segment and NSTR)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for nstr in range(-1,nNSTR):
            outfile.write(str(nstr) + sep)
        outfile.write('\n')
        for ls in range(nLS-1):
            outfile.write(sep + lsNames[ls])
            for nstr in range(-1,nNSTR):
                value = np.sum((trips['LOG_SEG']==ls) & (trips['NSTR']==nstr))
                outfile.write(sep + str(value))
            outfile.write('\n')
        
        # Number of trips by vehicle type and combustion type
        nTripsVeh = [None for veh in range(nVT)]
        outfile.write('\nNumber of trips (freight; by vehicle type and combustion type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type' + sep + 'Fuel' + sep + 'Electric' + sep + 'Hydrogen' + sep + 'Hybrid (electric)' + sep + 'Biofuel' + '\n')
        for veh in range(nVT):
            nTripsVeh[veh] = len(trips[trips['VEHTYPE']==veh])
            nTripsVehComb = [None for comb in range(5)]
            for comb in range(5):
                nTripsVehComb[comb] = np.sum((trips['VEHTYPE']==veh) & (trips['COMBTYPE']==comb))
            outfile.write(sep + vtNames[veh] + sep + str(nTripsVehComb[0]) + sep + str(nTripsVehComb[1]) + sep + str(nTripsVehComb[2]) \
                                             + sep + str(nTripsVehComb[3]) + sep + str(nTripsVehComb[4]) + '\n')
        
        # Number of trips by logistic segment and vehicle type
        outfile.write('\nNumber of trips (freight/parcel; by logistic segment and vehicle type)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for vt in range(nVT):
            outfile.write(vtNames[vt] + sep)
        outfile.write('\n')
        for ls in range(nLS-1):
            outfile.write(sep + lsNames[ls])
            for vt in range(nVT):
                value = np.sum((trips['LOG_SEG']==ls) & (trips['VEHTYPE']==vt))
                outfile.write(sep + str(value))
            outfile.write('\n')
        nParcelTripsVan  = np.sum(parcelTrips['VehType']=='Van')
        nParcelTripsLEVV = np.sum(parcelTrips['VehType']=='LEVV')
        outfile.write(sep + lsNames[nLS-1] + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep)
        outfile.write(str(nParcelTripsVan) + sep + str(nParcelTripsLEVV) + '\n')

        # Number of trips by van traffic segment
        outfile.write('\nNumber of trips (service vans; by logistic segment)' + sep + '\n')
        if vanTripsFound:
            outfile.write(sep + 'Segment' + sep + 'Number of trips\n')
            outfile.write(sep + 'Service'      + sep + str(np.sum(vanTripsService)     ) + '\n')
            outfile.write(sep + 'Construction' + sep + str(np.sum(vanTripsConstruction)) + '\n')
        else:
            outfile.write(sep + 'Van trips not found in outputfolder.' + '\n')

        
        #----------------------- Transported weight ---------------------------
        
        print("\tTransported weight"), log_file.write("\tTransported weight\n")
        
        # Transport weight by direction
        weightTotal     = round(sum(tours['TOUR_WEIGHT']),2)
        weightLeaving   = round(sum(tours[tours['DEST']>99999900]['TOUR_WEIGHT']),2)
        weightEntering  = round(sum(tours[tours['ORIG']>99999900]['TOUR_WEIGHT']),2)
        weightIntra     = weightTotal - weightLeaving - weightEntering

        outfile.write('\nTransported tonnes (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'Tonnes'            + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(weightIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(weightLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(weightEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(weightTotal)    + '\n')
        
        # Transported weight by NSTR
        weightNSTR = np.zeros((nNSTR,1))[:,0]
        outfile.write('\nTransported tonnes (freight; by NSTR goods type)' + sep + '\n')
        outfile.write(sep + 'NSTR' + sep + 'Tonnes\n')
        for nstr in range(nNSTR):
            weightNSTR[nstr] = round(sum(tours[tours['NSTR']==nstr]['TOUR_WEIGHT']),2)
            outfile.write(sep + str(nstr) + sep + str(weightNSTR[nstr]) + '\n')
         
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(10), weightNSTR)
        plt.title("Transported weight by commodity type (freight)")
        plt.xlabel('NSTR')
        plt.ylabel('Transported weight (tonnes)')
        plt.xticks(np.arange(10))
        fig.savefig(datapathOI + f"OUT_WeightByNSTR_{label}.png", bbox_inches='tight')
        plt.close()
        
        # Transported weight by Vehicle type
        weightVeh = [None for veh in range(nVT)]
        outfile.write('\nTransported tonnes (freight; by vehicle type)' + sep + '\n')
        outfile.write(sep + 'Vehicle type' + sep + 'Fuel' + sep + 'Electric' + sep + 'Hydrogen' + sep + 'Hybrid (electric)' + sep + 'Biofuel' + '\n')
        for veh in range(nVT):
            weightVeh[veh] = np.round(np.sum(tours[tours['VEHTYPE']==veh]['TOUR_WEIGHT']),2)
            weightVehComb = [None for comb in range(5)]
            for comb in range(5):
                weightVehComb[comb] = np.round(np.sum(tours[(tours['VEHTYPE']==veh) & (tours['COMBTYPE']==comb)]['TOUR_WEIGHT']),2)
            outfile.write(sep + vtNames[veh] + sep + str(weightVehComb[0]) + sep + str(weightVehComb[1]) + sep + str(weightVehComb[2]) \
                                             + sep + str(weightVehComb[3]) + sep + str(weightVehComb[4]) + '\n')
        
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(nVT), weightVeh)
        plt.title("Transported weight by vehicle type (freight)")
        plt.xlabel('Vehicle type')
        plt.ylabel('Transported weight (tonnes)')
        plt.xticks(np.arange(nVT)-0.5, vtNames, rotation=45)
        plt.tick_params(length=0)
        fig.savefig(datapathOI + f"OUT_WeightByVehType_{label}.png", bbox_inches='tight')
        plt.close()

        # Transported weight by logistic segment
        weightLS = np.zeros((nLS,1))[:,0]
        outfile.write('\nTransported tonnes (freight; by logistic segment)' + sep + '\n')
        outfile.write(sep + 'LS' + sep + 'Tonnes\n')
        for ls in range(nLS-1):
            weightLS[ls] = round(sum(tours[tours['LOG_SEG']==ls]['TOUR_WEIGHT']),2)
            outfile.write(sep + lsNames[ls] + sep + str(weightLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS-1] + sep + 'n.a.' + '\n')
         
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(nLS), weightLS)
        plt.title("Transported weight by logistic segment (freight)")
        plt.xlabel('Logistic segment')
        plt.ylabel('Transported weight (tonnes)')
        plt.xticks(np.arange(nLS), lsNames, rotation=45)
        fig.savefig(datapathOI + f"OUT_WeightByLS_{label}.png", bbox_inches='tight')
        plt.close()
                    
        print("\tAverage trip loads"), log_file.write("\tAverage trip loads\n")
        
        # Average trip load by direction
        loadedTrips         = trips[trips['NSTR']!=-1]
        loadedTripsIntra    = loadedTrips[[loadedTrips['DEST'][i]<=99999900 and loadedTrips['ORIG'][i]<= 99999900 for i in loadedTrips.index]]
        avgLoadIntra        = round(loadedTripsIntra['TRIP_WEIGHT'].mean(),2)
        avgLoadLeaving      = round(loadedTrips[loadedTrips['DEST']>99999900]['TRIP_WEIGHT'].mean(),2)
        avgLoadEntering     = round(loadedTrips[loadedTrips['ORIG']>99999900]['TRIP_WEIGHT'].mean(),2)
        avgLoad             = round(loadedTrips['TRIP_WEIGHT'].mean(),2)
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by direction)*{sep}\n")
        outfile.write(f"{sep}Direction{sep}Average trip load (tonnes)\n")
        outfile.write(f"{sep}Inside study area{sep}{avgLoadIntra}\n")
        outfile.write(f"{sep}Leaving study area{sep}{avgLoadLeaving}\n")
        outfile.write(f"{sep}Entering study area{sep}{avgLoadEntering}\n")
        outfile.write(f"{sep}All loaded trips{sep}{avgLoad}\n")
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
        
        # Average load per NSTR
        avgLoadNSTR = np.zeros((nNSTR))
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by NSTR goods type)*{sep}\n")
        outfile.write(f"{sep}NSTR{sep}Average trip load (tonnes)\n")
        for nstr in range(nNSTR):
            avgLoadNSTR[nstr] = round(loadedTrips[loadedTrips['NSTR']==nstr]['TRIP_WEIGHT'].mean(),2)
            outfile.write(f"{sep}{nstr}{sep}{avgLoadNSTR[nstr]}\n")
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
           
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(nNSTR), avgLoadNSTR)
        plt.title("Average load carried in trip (freight; excl. empty trips)")
        plt.xlabel('NSTR')
        plt.ylabel('Average load carried in trip (tonnes)')
        plt.xticks(np.arange(nNSTR))
        fig.savefig(f"{datapathOI}OUT_AvgLoadByNSTR_{label}.png", bbox_inches='tight')
        plt.close()
        
        # Average load per vehicle type
        avgLoadVeh = np.zeros(nVT)
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by vehicle type)*{sep}\n")
        outfile.write(f"{sep}Vehicle type{sep}Average trip load (tonnes)\n")
        for veh in range(nVT):
            i = veh
            avgLoadVeh[i] = round(loadedTrips[loadedTrips['VEHTYPE']==i]['TRIP_WEIGHT'].mean(),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{avgLoadVeh[i]}\n")
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
        
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(nVT), avgLoadVeh)
        plt.title("Average load carried in trip (freight; excl. empty trips)")
        plt.xlabel('Vehicle type')
        plt.ylabel('Average load carried in trip (tonnes)')
        plt.xticks(np.arange(nVT)-0.5, vtNames, rotation=45)
        plt.tick_params(length=0)
        fig.savefig(f"{datapathOI}OUT_AvgLoadByVehType_{label}.png", bbox_inches='tight')
        plt.close()

        # Average load per logistic segment
        avgLoadLS = np.zeros((nLS))
        outfile.write(f"\nAverage load carried in trip (tonnes) (freight; by logistic segment)*{sep}\n")
        outfile.write(f"{sep}LS{sep}Average trip load (tonnes)\n")
        for ls in range(nLS-1):
            avgLoadLS[ls] = round(loadedTrips[loadedTrips['LOG_SEG']==ls]['TRIP_WEIGHT'].mean(),2)
            outfile.write(sep + lsNames[ls] + sep + str(avgLoadLS[ls]) + '\n')
        outfile.write(sep + lsNames[nLS-1] + sep + 'n.a.' + '\n')
        outfile.write(f"{sep}*Excluding empty trips{sep}\n")
           
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(np.arange(nLS), avgLoadLS)
        plt.title("Average load carried in trip (freight; excl. empty trips)")
        plt.xlabel('Logistic segment')
        plt.ylabel('Average load carried in trip (tonnes)')
        plt.xticks(np.arange(nLS), lsNames, rotation=45)
        fig.savefig(f"{datapathOI}OUT_AvgLoadByLS_{label}.png", bbox_inches='tight')
        plt.close()
        
        
        #---------------- Number of shipments per tour ------------------------
        
        print("\tNumber of shipments"), log_file.write("\tNumber of shipments\n")
        
        nShipsTotal = pd.value_counts(tours['N_SHIP'], sort=False)
        nShipsNSTR = pd.crosstab(tours['N_SHIP'], tours['NSTR'])
        nShipsLS = pd.crosstab(tours['N_SHIP'], tours['LOG_SEG'])
        toursInternal = tours[ [tours['ORIG'][i]<=99999900 and tours['DEST'][i]<=99999900 for i in tours.index] ]
        nShipsTotalInternal = pd.value_counts(toursInternal['N_SHIP'], sort=False)
        nShipsNSTRInternal = pd.crosstab(toursInternal['N_SHIP'], toursInternal['NSTR'])
        nShipsLSInternal = pd.crosstab(toursInternal['N_SHIP'], toursInternal['LOG_SEG'])
        
        outfile.write("\nNumber of shipments per tour (freight; by NSTR)\n")
        outfile.write(f"{sep}{sep}NSTR\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsNSTR.columns))[1:-1]}{sep}Total\n")
        for ships in nShipsNSTR.index:
            totalShips = sum(nShipsNSTR.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsNSTR.loc[ships,:]))[1:-1]}{sep}{totalShips}\n")
        
        outfile.write("\nNumber of shipments per tour within study area (freight; by NSTR)\n")
        outfile.write(f"{sep}{sep}NSTR\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsNSTRInternal.columns))[1:-1]}{sep}Total\n")
        for ships in nShipsNSTR.index:
            totalShips = sum(nShipsNSTRInternal.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsNSTRInternal.loc[ships,:]))[1:-1]}{sep}{totalShips}\n")            
            
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(nShipsTotal.index, nShipsTotal)
        plt.title("Number of shipments per tour (freight)")
        plt.xlabel('Number of shipments in tour')
        plt.ylabel('# tours')
        plt.xticks(nShipsTotal.index)
        fig.savefig(f"{datapathOI}OUT_NumberOfShipments_{label}.png", bbox_inches='tight')
        plt.close()        
        
        fig = plt.figure()
        plt.subplot(111)
        plt.bar(nShipsTotalInternal.index, nShipsTotalInternal)
        plt.title("Number of shipments per tour (freight; only within study area)")
        plt.xlabel('Number of shipments in tour')
        plt.ylabel('# tours')
        plt.xticks(nShipsTotalInternal.index)
        fig.savefig(f"{datapathOI}OUT_NumberOfShipmentsInternal_{label}.png", bbox_inches='tight')
        plt.close()               

        outfile.write("\nNumber of shipments per tour (freight; by logistic segment)\n")
        outfile.write(f"{sep}{sep}Logistic segment\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsLS.columns))[1:-1]}{sep}Total\n")
        for ships in nShipsLS.index:
            totalShips = sum(nShipsLS.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsLS.loc[ships,:]))[1:-1]}{sep}{totalShips}\n")
        
        outfile.write("\nNumber of shipments per tour within study area (freight; by logistic segment)\n")
        outfile.write(f"{sep}{sep}Logistic segment\n")
        outfile.write(f"{sep}Number of shipments{sep}{str(list(nShipsLSInternal.columns))[1:-1]}{sep}Total\n")
        for ships in nShipsLS.index:
            totalShips = sum(nShipsLSInternal.loc[ships,:])
            outfile.write(f"{sep}{ships}{sep}{str(list(nShipsLSInternal.loc[ships,:]))[1:-1]}{sep}{totalShips}\n")   

        outfile.write("\nAverage number of trips per tour (freight/parcel; by logistic segment)\n")
        outfile.write('LS' + sep + 'Number of trips\n')
        for ls in range(nLS-1):
            nToursLS = len(np.unique(trips.loc[trips['LOG_SEG']==ls,'TOUR_ID']))
            if nToursLS > 0:
                outfile.write(lsNames[ls] + sep + str(len(trips[trips['LOG_SEG']==ls])/nToursLS) + '\n')
            else:
                outfile.write(lsNames[ls] + sep + 'n.a.' + '\n')
        outfile.write(lsNames[nLS-1] + sep + str(np.mean(parcelTours['N_trips_tour'])) + '\n')
        outfile.write('Total (freight)' + sep + str(len(trips)/len(np.unique(trips['TOUR_ID']))) + '\n')
        

        #-------------------- Tonkilometers -----------------------------------

        # Trips tonkilometers
        trips['TONKM'] = trips['DIST'] * trips['TRIP_WEIGHT']
        tonKilometersTotal     = round(sum(trips['TONKM']),2)
        tonKilometersLeaving   = round(sum(trips[trips['DEST']>99999900]['TONKM']),2)
        tonKilometersEntering  = round(sum(trips[trips['ORIG']>99999900]['TONKM']),2)
        tonKilometersIntra     =  tonKilometersTotal -  tonKilometersLeaving -  tonKilometersEntering

        outfile.write('\nTonkilometers trips (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'TonKM'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(tonKilometersIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(tonKilometersLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(tonKilometersEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(tonKilometersTotal)    + '\n')

        
        # Shipments tonkilometers
        shipments['TONKM'] = shipments['DIST'] * shipments['WEIGHT']
        tonKilometersTotal     = round(sum(shipments['TONKM']),2)
        tonKilometersLeaving   = round(sum(shipments[shipments['DEST']>99999900]['TONKM']),2)
        tonKilometersEntering  = round(sum(shipments[shipments['ORIG']>99999900]['TONKM']),2)
        tonKilometersIntra     = tonKilometersTotal -  tonKilometersLeaving -  tonKilometersEntering

        outfile.write('\nTonkilometers shipments (freight; by direction)' + sep + '\n')
        outfile.write(sep + 'Direction'             + sep + 'TonKM'   + '\n')
        outfile.write(sep + 'Inside study area'     + sep + str(tonKilometersIntra)    + '\n')
        outfile.write(sep + 'Leaving study area'    + sep + str(tonKilometersLeaving)  + '\n')
        outfile.write(sep + 'Entering study area'   + sep + str(tonKilometersEntering) + '\n')
        outfile.write(sep + 'Total'                 + sep + str(tonKilometersTotal)    + '\n')
        
        
        #---------------- Vehicle Kilometers Travelled ------------------------
        
        print("\tVehicle Kilometers Travelled"), log_file.write("\tVehicle Kilometers Travelled\n")
      
        # VKM by logistic segment and vehicle type
        outfile.write('\nVehicle kilometers (Total; based on skim) (freight/parcel; by logistic segment and vehicle type)' + sep + '\n')
        outfile.write(sep + 'LS' + sep)
        for vt in range(nVT):
            outfile.write(vtNames[vt] + sep)
        outfile.write('\n')
        for ls in range(nLS-1):
            outfile.write(sep + lsNames[ls])
            for vt in range(nVT):
                value = np.sum(trips.loc[(trips['LOG_SEG']==ls) & (trips['VEHTYPE']==vt),'DIST'])
                outfile.write(sep + str(value))
            outfile.write('\n')
        nParcelTripsVan  = np.sum(parcelTrips.loc[parcelTrips['VehType']=='Van', 'DIST'])
        nParcelTripsLEVV = np.sum(parcelTrips.loc[parcelTrips['VehType']=='LEVV','DIST'])
        outfile.write(sep + lsNames[nLS-1] + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep + '0' + sep)
        outfile.write(str(nParcelTripsVan) + sep + str(nParcelTripsLEVV) + '\n')

        # VKMs by van traffic segment
        outfile.write('\nVehicle kilometers (Total; based on skim) (service vans; by logistic segment)' + sep + '\n')
        if vanTripsFound:
            outfile.write(sep + 'Segment' + sep + 'Number of trips\n')
            outfile.write(sep + 'Service'      + sep + str(np.sum(vanTripsService      * skimDistance / 1000)) + '\n')
            outfile.write(sep + 'Construction' + sep + str(np.sum(vanTripsConstruction * skimDistance / 1000)) + '\n')
        else:
            outfile.write(sep + 'Van trips not found in outputfolder.' + '\n')
            
            
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (Total) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoaded['LENGTH'] * linksLoaded[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (Total) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoaded['LENGTH'] * linksLoaded[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")       
        
        outfile.write("\nVehicle Kilometers Travelled (NL) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(sep + 'Service'      + sep + str(np.sum(linksLoaded['LENGTH'] * linksLoaded['N_VAN_S'])) + '\n')
        outfile.write(sep + 'Construction' + sep + str(np.sum(linksLoaded['LENGTH'] * linksLoaded['N_VAN_C'])) + '\n')
            
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")    

        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            if f'N_LS8_VEH{i}' in linksLoaded.columns:
                temp = linksLoadedZH['LENGTH'] * linksLoadedZH[f'N_LS8_VEH{i}']
                VKT_VEH[i] = np.round(np.sum(temp),2)
                outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")    

        outfile.write("\nVehicle Kilometers Travelled (in Zuid-Holland) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(sep + 'Service'      + sep + str(np.sum(linksLoadedZH['LENGTH'] * linksLoadedZH['N_VAN_S'])) + '\n')
        outfile.write(sep + 'Construction' + sep + str(np.sum(linksLoadedZH['LENGTH'] * linksLoadedZH['N_VAN_C'])) + '\n')
         
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")               

        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            if f'N_LS8_VEH{i}' in linksLoaded.columns:
                temp = linksLoadedRdam['LENGTH'] * linksLoadedRdam[f'N_LS8_VEH{i}']
                VKT_VEH[i] = np.round(np.sum(temp),2)
                outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n") 

        outfile.write("\nVehicle Kilometers Travelled (in Rotterdam) (service vans; by segment)\n")
        outfile.write(f"{sep}Segment{sep}VKT\n")
        outfile.write(sep + 'Service'      + sep + str(np.sum(linksLoadedRdam['LENGTH'] * linksLoadedRdam['N_VAN_S'])) + '\n')
        outfile.write(sep + 'Construction' + sep + str(np.sum(linksLoadedRdam['LENGTH'] * linksLoadedRdam['N_VAN_C'])) + '\n')
        
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in The Hague) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedDH['LENGTH'] * linksLoadedDH[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in The Hague) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoadedDH['LENGTH'] * linksLoadedDH[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")     
        
        VKT_LS = np.zeros((nLS))
        outfile.write("\nVehicle Kilometers Travelled (in ZEZ) (freight/parcel; by logistic segment)\n")
        outfile.write(f"{sep}LS{sep}VKT\n")
        for ls in range(nLS):
            temp = linksLoadedZEZ['LENGTH'] * linksLoadedZEZ[f'N_LS{ls}']
            VKT_LS[ls] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{lsNames[ls]}{sep}{VKT_LS[ls]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_LS)}\n")
        
        VKT_VEH = np.zeros((nVT))
        outfile.write("\nVehicle Kilometers Travelled (in ZEZ) (freight/parcel; by vehicle Type)\n")
        outfile.write(f"{sep}Vehicle type{sep}VKT\n")
        for veh in vtNums:
            i = veh
            temp = linksLoadedZEZ['LENGTH'] * linksLoadedZEZ[f'N_VEH{i}']
            VKT_VEH[i] = np.round(np.sum(temp),2)
            outfile.write(f"{sep}{vtNames[i]}{sep}{VKT_VEH[i]}\n")
        outfile.write(f"{sep}Total{sep}{sum(VKT_VEH)}\n")            
        
        
        
        #------------------------ Emissions -----------------------------------
        
        print('\tEmissions'), log_file.write("\tEmissions\n")
        
        outfile.write('\nEmissions from network (freight/parcel/vans; Total)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in ['CO2','SO2','PM','NOX']:
            total = np.sum(linksLoaded[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in Zuid-Holland)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in ['CO2','SO2','PM','NOX']:
            total = np.sum(linksLoadedZH[emission])
            outfile.write(emission + sep + str(total) + '\n')
            
        outfile.write('\nEmissions from network (freight/parcel/vans; in Rotterdam)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in ['CO2','SO2','PM','NOX']:
            total = np.sum(linksLoadedRdam[emission])
            outfile.write(emission + sep + str(total) + '\n')

        outfile.write('\nEmissions from network (freight/parcel/vans; in The Hague)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in ['CO2','SO2','PM','NOX']:
            total = np.sum(linksLoadedDH[emission])
            outfile.write(emission + sep + str(total) + '\n')
            
        outfile.write('\nEmissions from network (freight/parcel/vans; in ZEZ)\n')
        outfile.write('Type' + sep + 'kg\n')
        for emission in ['CO2','SO2','PM','NOX']:
            total = np.sum(linksLoadedZEZ[emission])
            outfile.write(emission + sep + str(total) + '\n')
            
        outfile.write('\nEmissions from network (Total) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoaded[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')                     
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoaded[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')   
                
        outfile.write('\nEmissions from network (in Zuid-Holland) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedZH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedZH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  
                
        outfile.write('\nEmissions from network (in Rotterdam) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedRdam[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedRdam[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  
                
        outfile.write('\nEmissions from network (in The Hague) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedDH[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedDH[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  
                
        outfile.write('\nEmissions from network (in ZEZ) (freight/parcel/vans; by logistic segment)\n')
        outfile.write('Logistic segment' + sep + 'Type' + sep + 'kg\n')        
        for ls in range(nLS):
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedZEZ[emission + '_LS' + str(ls)])
                outfile.write(lsNames[ls] + sep + emission + sep + str(total) + '\n')   
        for ls in ['VAN_S','VAN_C']:
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(linksLoadedZEZ[emission + '_' + str(ls)])
                outfile.write(ls + sep + emission + sep + str(total) + '\n')  
                
        outfile.write('\nEmissions from network (freight/parcel/vans; by municipality)\n')
        outfile.write('Municipality' + sep + 'Type' + sep + 'kg (all)' + sep + 'kg (parcel)' + '\n')
        for gem in np.unique(linksLoaded['Gemeentena']):
            currentLinks = linksLoaded[linksLoaded['Gemeentena']==gem]
            for emission in ['CO2','SO2','PM','NOX']:
                total = np.sum(currentLinks[emission])
                totalParcel = np.sum(currentLinks[emission + '_LS8'])
                outfile.write(gem + sep + emission + sep + str(total) + sep + str(totalParcel) + '\n')                   
                
        print(f"Tables written to {datapathOI}Output_Outputindicator_{label}.csv"), log_file.write(f"Tables written to {datapathOI}Output_Outputindicator_{label}.csv\n")
        print(f"Figures written to {datapathOI}OUT_<figname>_{label}.png"), log_file.write(f"Figures written to {datapathOI}OUT_<figname>_{label}.png\n")
        
        
        #------------------------ End of script -------------------------------
        
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Output Indicators: Done")
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
                root.update_statusbar("Output Indicators: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]
 
    
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    INPUTFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2016/'
    OUTPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunREF2016/'
    PARAMFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/'
    
    SKIMTIME        = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimTijd_REF.mtx'
    SKIMDISTANCE    = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimAfstand_REF.mtx'
    LINKS		    = INPUTFOLDER + 'links_v5.shp'
    NODES           = INPUTFOLDER + 'nodes_v5.shp'
    ZONES           = INPUTFOLDER + 'Zones_v5.shp'
    SEGS            = INPUTFOLDER + 'SEGS2016.csv'
    COMMODITYMATRIX = INPUTFOLDER + 'CommodityMatrixNUTS2_2016.csv'
    PARCELNODES     = INPUTFOLDER + 'parcelNodes_v2.shp'
    
    YEARFACTOR = 193
    
    NUTSLEVEL_INPUT = 2
    
    PARCELS_PER_HH	 = 0.195
    PARCELS_PER_EMPL = 0.073
    PARCELS_MAXLOAD	 = 180
    PARCELS_DROPTIME = 120
    PARCELS_SUCCESS_B2C   = 0.75
    PARCELS_SUCCESS_B2B   = 0.95
    PARCELS_GROWTHFREIGHT = 1.0
    
    SHIPMENTS_REF = ""
    SELECTED_LINKS = ""
    
    IMPEDANCE_SPEED = 'V_FR_OS'
    
    LABEL = 'REF'
    
    MODULES = ['SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, LINKS, NODES, ZONES, SEGS, \
            COMMODITYMATRIX, PARCELNODES, PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
            PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", "LINKS", "NODES", "ZONES", "SEGS", \
                  "COMMODITYMATRIX", "PARCELNODES", "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
                  "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B",  "PARCELS_GROWTHFREIGHT", \
                  "YEARFACTOR", "NUTSLEVEL_INPUT", \
                  "IMPEDANCE_SPEED", \
                  "SHIPMENTS_REF", "SELECTED_LINKS", \
                  "LABEL", \
                  "MODULES"]
     
    varDict = {}
    for i in range(len(args)):
        varDict[varStrings[i]] = args[i]
        
    # Run the module
    main(varDict)

    
    
    
    
    
    