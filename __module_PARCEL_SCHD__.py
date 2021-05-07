# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:21:12 2020

@author: modelpc
"""
import numpy as np
import pandas as pd
import time
import datetime
from __functions__ import read_mtx, read_shape

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread



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
        self.root.title("Progress Parcel Scheduling")
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
        parcelNodesPath  = varDict['PARCELNODES']
        label            = varDict['LABEL']
        
        dropOffTimeSec = varDict['PARCELS_DROPTIME']
        maxVehicleLoad = varDict['PARCELS_MAXLOAD']
        
        exportTripMatrix = True
        
        log_file = open(datapathO + "Logfile_ParcelScheduling.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
    
        # Scheduling function can handle four different tour types = { 0: depot to HH, 1: depot to UCC, 2: UCC to HH by van, 3: UCC to HH by LEVV   
        
        
        # --------------------------- Import data----------------------------------
        print('Importing data...'), log_file.write('Importing data...\n')        
        parcels = pd.read_csv(datapathO + 'ParcelDemand_' + label + '.csv')
        
        parcelNodes, coords = read_shape(parcelNodesPath, returnGeometry=True)
        parcelNodes['X']    = [coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y']    = [coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes['id'] = parcelNodes['id'].astype(int)
        parcelNodes.index = parcelNodes['id']
        parcelNodes = parcelNodes.sort_index()
        parcelNodesCEP = {}
        for i in parcelNodes.index:
            parcelNodesCEP[parcelNodes.at[i,'id']] = parcelNodes.at[i,'CEP']
        nParcelNodes = len(parcelNodes)
        
        zones = read_shape(zonesPath)
        zones = zones.sort_values('AREANR')
        zones.index = zones['AREANR']
        supCoordinates = pd.read_csv(datapathI + 'SupCoordinatesID.csv', sep=',')
        supCoordinates.index = supCoordinates['AREANR']
        
        zonesX = {}
        zonesY = {}
        for areanr in zones.index:
            zonesX[areanr] = zones.at[areanr, 'X']
            zonesY[areanr] = zones.at[areanr, 'Y']
        for areanr in supCoordinates.index:
            zonesX[areanr] = supCoordinates.at[areanr, 'Xcoor']
            zonesY[areanr] = supCoordinates.at[areanr, 'Ycoor']                    
        
        nIntZones = len(zones)
        nSupZones = 43
        zoneDict  = dict(np.transpose(np.vstack( (np.arange(1,nIntZones+1), zones['AREANR']) )))
        zoneDict  = {int(a):int(b) for a,b in zoneDict.items()}
        for i in range(nSupZones):
            zoneDict[nIntZones+i+1] = 99999900 + i + 1
        invZoneDict = dict((v, k) for k, v in zoneDict.items())
        
        # Change zoning to skim zones which run continuously from 0
        parcels['D_zone']        = [invZoneDict[x] for x in parcels['D_zone']]
        parcels['O_zone']        = [invZoneDict[x] for x in parcels['O_zone']]
        parcelNodes['skim_zone'] = [invZoneDict[x] for x in parcelNodes['AREANR']]
        
        # System input for scheduling
        parcelDepTime = np.array(pd.read_csv(f"{datapathI}departureTimeParcelsCDF.csv").iloc[:,1])        
        dropOffTime   = dropOffTimeSec/3600    
        skimTravTime  = read_mtx(skimTravTimePath)
        skimDistance  = read_mtx(skimDistancePath)
        nZones        = int(len(skimTravTime)**0.5)
    
        # Make lists with starting points for deliveries; depots and UCCs
        depotZones = list(parcelNodes['skim_zone'])
        depotIDs   = list(parcelNodes['id'])
        
        # Make skims with travel times between parcel nodes / UCCs and all other zones
        parcelSkim = np.zeros((nZones, len(parcelNodes)))
        i = 0
        for parcelNodeZone in depotZones:
            orig = parcelNodeZone
            dest = 1 + np.arange(nZones)
            parcelSkim[:,i] = np.round( (skimTravTime[(orig-1)*nZones+(dest-1)] / 3600),4)     
            i += 1
        depotDict = dict(zip(parcelNodes['skim_zone'],parcelNodes['id']))    


        # ------- Scheduling of trips departing at depots (UCC scenario) -----------------------

        if label == 'UCC':

            # Aggregate all parcels with the same depot, origin and destination
            delivFromDepots   = pd.pivot_table(parcels[parcels['FROM_UCC']==0], values=['Parcel_ID'], index=['DepotNumber', "O_zone", 'D_zone','TO_UCC'], aggfunc = {'Parcel_ID': 'count'})           
            delivFromDepots   = delivFromDepots.rename(columns={'Parcel_ID':'Parcels'}) 
            delivFromDepots['Depot' ] = [x[0] for x in delivFromDepots.index]
            delivFromDepots['Orig'  ] = [x[1] for x in delivFromDepots.index]
            delivFromDepots['Dest'  ] = [x[2] for x in delivFromDepots.index]
            delivFromDepots['TO_UCC'] = [x[3] for x in delivFromDepots.index]
            
            delivFromDepotsToUCC = delivFromDepots[delivFromDepots['TO_UCC']==1]            
            delivFromDepotsToHH  = delivFromDepots[delivFromDepots['TO_UCC']==0]            
            delivFromDepotsToUCC.index = np.arange(len(delivFromDepotsToUCC))
            delivFromDepotsToHH.index  = np.arange(len(delivFromDepotsToHH))
            
            # Initialize a list of arrays to be filled in the scheduling procedure           
            depotListToUCC = [[np.zeros((1,1),dtype=int)[0] \
                              for ship in range(int(np.ceil(delivFromDepotsToUCC.loc[delivFromDepotsToUCC['Depot']==depotID,'Parcels'].sum()/maxVehicleLoad)))] \
                              for depotID in depotIDs]

            depotListToHH = [[np.zeros((1,1),dtype=int)[0] \
                             for ship in range(int(np.ceil(delivFromDepotsToHH.loc[delivFromDepotsToHH['Depot']==depotID,'Parcels'].sum()/maxVehicleLoad)))] \
                             for depotID in depotIDs]
    
            print('Starting scheduling procedure for parcels from depots to households...'), log_file.write('Starting scheduling procedure for parcels from depots to households...\n')    
            deliveries = parcelSched(delivFromDepotsToHH, parcelSkim, depotZones, depotIDs, depotDict, maxVehicleLoad, dropOffTime, depotListToHH, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, 0, label)    

            print('Starting scheduling procedure for parcels from depots to UCC...'), log_file.write('Starting scheduling procedure for parcels from depots to UCC...\n')    
            deliveries1 = parcelSched(delivFromDepotsToUCC, parcelSkim, depotZones, depotIDs, depotDict, maxVehicleLoad, dropOffTime, depotListToUCC, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, 1, label)    

    
        # ------- Scheduling of trips departing at depots (REF scenario) -----------------------

        else:
            # Aggregate all parcels with the same depot, origin and destination
            delivFromDepots   = pd.pivot_table(parcels, values=['Parcel_ID'], index=['DepotNumber', "O_zone", 'D_zone'], aggfunc = {'Parcel_ID': 'count'})           
            delivFromDepots   = delivFromDepots.rename(columns={'Parcel_ID':'Parcels'}) 
            delivFromDepots['Depot'] = [x[0] for x in delivFromDepots.index]
            delivFromDepots['Orig' ] = [x[1] for x in delivFromDepots.index]
            delivFromDepots['Dest' ] = [x[2] for x in delivFromDepots.index]
            delivFromDepots.index = np.arange(len(delivFromDepots))
            
            # Initialize a list of arrays to be filled in the scheduling procedure           
            depotList   = [[np.zeros((1,1),dtype=int)[0]  \
                           for ship in range(int(np.ceil(delivFromDepots.loc[delivFromDepots['Depot']==depotID,'Parcels'].sum()/maxVehicleLoad)))] \
                           for depotID in depotIDs]
    
            print('Starting scheduling procedure for parcels...'), log_file.write('Starting tour formation procedure for parcels...\n')    
            deliveries = parcelSched(delivFromDepots, parcelSkim, depotZones, depotIDs, depotDict, maxVehicleLoad, dropOffTime, depotList, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, 0, label)


        # ------- Additional scheduling in UCC scenario: Trips departing from UCCs -----------------------
        if label == 'UCC':
            print('Scheduling delivery tours from UCC to ZEZ...'), log_file.write('Scheduling delivery tours from UCC into ZEZ...\n')
            
            uccZones   = [int(i) for i in list(parcels['O_zone'][parcels['FROM_UCC']==1].unique())]
            uccZonesDict = dict(zip(uccZones, np.arange(len(parcelNodes)+1,len(parcelNodes)+1+len(uccZones))))
            parcelSkimUCC = np.zeros((nZones, len(uccZones)))
            i = 0
            for uccZone in uccZones:
                orig = uccZone
                dest = 1 + np.arange(nZones)
                parcelSkimUCC[:,i] = np.round( (skimTravTime[(orig-1)*nZones+(dest-1)] / 3600),4)     
                i += 1

            delivFromUCC   = pd.pivot_table(parcels[parcels['FROM_UCC']==1], values=['Parcel_ID'], index=["O_zone", 'D_zone','VEHTYPE'], aggfunc = {'Parcel_ID': 'count'})           
            delivFromUCC   = delivFromUCC.rename(columns={'Parcel_ID':'Parcels'}) 
            delivFromUCC['Depot'  ] = [uccZonesDict[x[0]] for x in delivFromUCC.index]
            delivFromUCC['Orig'   ] = [x[0] for x in delivFromUCC.index]
            delivFromUCC['Dest'   ] = [x[1] for x in delivFromUCC.index]
            delivFromUCC['VEHTYPE'] = [x[2] for x in delivFromUCC.index]
            
            delivUCCVans = delivFromUCC[delivFromUCC['VEHTYPE']==7]
            delivUCCLEVV = delivFromUCC[delivFromUCC['VEHTYPE']==8]
            delivUCCVans.index = np.arange(len(delivUCCVans))
            delivUCCLEVV.index = np.arange(len(delivUCCLEVV))
            
            # Initialize a list of arrays to be filled in the scheduling procedure
            uccDepotIDs = [x + 1 + nParcelNodes for x in range(len(uccZones))]
            uccListVans = [[np.zeros((1,1),dtype=int)[0] \
                              for ship in range(int(np.ceil(delivUCCVans.loc[delivUCCVans['Depot']==uccDepotID,'Parcels'].sum()/maxVehicleLoad)))] \
                              for uccDepotID in uccDepotIDs]

            uccListLEVV = [[np.zeros((1,1),dtype=int)[0] \
                             for ship in range(int(np.ceil(delivUCCVans.loc[delivUCCVans['Depot']==uccDepotID,'Parcels'].sum()/int(maxVehicleLoad/5))))] \
                             for uccDepotID in uccDepotIDs]
            
            deliveries2  = parcelSched(delivUCCVans, parcelSkimUCC, uccZones, uccDepotIDs, uccZonesDict,     maxVehicleLoad,    dropOffTime, uccListVans, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, 2, label)
            deliveries3  = parcelSched(delivUCCLEVV, parcelSkimUCC, uccZones, uccDepotIDs, uccZonesDict, int(maxVehicleLoad/5), dropOffTime, uccListLEVV, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, 3, label)    
    
            deliveries = pd.concat([deliveries, deliveries1, deliveries2, deliveries3])
            deliveries.index = np.arange(len(deliveries))
        

        # ------------------ Export output table to CSV and SHP -------------------
        
        # Transform to MRDH zone numbers and export
        deliveries['O_zone']  =  [zoneDict[x] for x in deliveries['O_zone']]
        deliveries['D_zone']  =  [zoneDict[x] for x in deliveries['D_zone']]
        deliveries['TripDepTime'] = [round(deliveries['TripDepTime'][i], 3) for i in deliveries.index]
        deliveries['TripEndTime'] = [round(deliveries['TripEndTime'][i], 3) for i in deliveries.index]
        
        print(f"Writing scheduled trips to {datapathO}ParcelSchedule_{label}.csv")
        log_file.write(f"Writing scheduled trips to {datapathO}ParcelSchedule_{label}.csv\n")
        deliveries.to_csv(f"{datapathO}ParcelSchedule_{label}.csv", index=False)  
        
        print('Writing GeoJSON...'), log_file.write('Writing GeoJSON...\n')

        # Initialize arrays with coordinates        
        Ax = np.zeros(len(deliveries), dtype=int)
        Ay = np.zeros(len(deliveries), dtype=int)
        Bx = np.zeros(len(deliveries), dtype=int)
        By = np.zeros(len(deliveries), dtype=int)
        
        # Determine coordinates of LineString for each trip
        tripIDs  = [x.split('_')[-1] for x in deliveries['Trip_ID']]
        tourTypes = np.array(deliveries['TourType'], dtype=int)
        depotIDs = np.array(deliveries['Depot_ID'])
        for i in deliveries.index[:-1]:
            # First trip of tour
            if tripIDs[i] == '0' and tourTypes[i]<=1:
                Ax[i] = parcelNodes['X'][depotIDs[i]]
                Ay[i] = parcelNodes['Y'][depotIDs[i]]
                Bx[i] = zonesX[deliveries['D_zone'][i]]
                By[i] = zonesY[deliveries['D_zone'][i]]
            # Last trip of tour
            elif tripIDs[i+1] == '0' and tourTypes[i]<=1:
                Ax[i] = zonesX[deliveries['O_zone'][i]]
                Ay[i] = zonesY[deliveries['O_zone'][i]]                
                Bx[i] = parcelNodes['X'][depotIDs[i]]
                By[i] = parcelNodes['Y'][depotIDs[i]]
            # Intermediate trips of tour
            else:
                Ax[i] = zonesX[deliveries['O_zone'][i]]
                Ay[i] = zonesY[deliveries['O_zone'][i]]
                Bx[i] = zonesX[deliveries['D_zone'][i]]
                By[i] = zonesY[deliveries['D_zone'][i]]
        # Last trip of last tour
        i += 1
        if tourTypes[i]<=1:
            Ax[i] = zonesX[deliveries['O_zone'][i]]
            Ay[i] = zonesY[deliveries['O_zone'][i]]                
            Bx[i] = parcelNodes['X'][depotIDs[i]]
            By[i] = parcelNodes['Y'][depotIDs[i]]
        else:
            Ax[i] = zonesX[deliveries['O_zone'][i]]
            Ay[i] = zonesY[deliveries['O_zone'][i]]
            Bx[i] = zonesX[deliveries['D_zone'][i]]
            By[i] = zonesY[deliveries['D_zone'][i]]
                
        Ax = np.array(Ax, dtype=str)
        Ay = np.array(Ay, dtype=str)
        Bx = np.array(Bx, dtype=str)
        By = np.array(By, dtype=str)
        nTrips = len(deliveries)
        
        with open(datapathO + f"ParcelSchedule_{label}.geojson", 'w') as geoFile:
            geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
            for i in range(nTrips-1):
                outputStr = ""
                outputStr = outputStr + '{ "type": "Feature", "properties": '
                outputStr = outputStr + str(deliveries.loc[i,:].to_dict()).replace("'",'"')
                outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
                geoFile.write(outputStr)
                if i%int(nTrips/10) == 0:
                    print('\t' + str(int(round((i / nTrips)*100, 0))) + '%', end='\r')
                    
            # Bij de laatste feature moet er geen komma aan het einde
            i += 1
            outputStr = ""
            outputStr = outputStr + '{ "type": "Feature", "properties": '
            outputStr = outputStr + str(deliveries.loc[i,:].to_dict()).replace("'",'"')
            outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
            outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
            outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
            geoFile.write(outputStr)
            geoFile.write(']\n')
            geoFile.write('}')
        
        print(f'Parcel schedules written to {datapathO}ParcelSchedule_{label}.geojson'), log_file.write(f'Parcel schedules written to {datapathO}ParcelSchedule_{label}.geojson\n')        

        
        
        # ------------------------ Create and export trip matrices ----------------
        
        if exportTripMatrix:
            print('Generating trip matrix...'), log_file.write('Generating trip matrix...\n')
            cols = ['ORIG','DEST', 'N_TOT']
            deliveries['N_TOT'] = 1
            
            # Gebruik N_TOT om het aantal ritten per HB te bepalen, voor elk logistiek segment, voertuigtype en totaal
            pivotTable = pd.pivot_table(deliveries, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
            pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
            pivotTable['DEST'] = [x[1] for x in pivotTable.index]
            pivotTable = pivotTable[cols]
            
            pivotTable.to_csv(f"{datapathO}tripmatrix_parcels_{label}.txt", index=False, sep='\t')
            print(f'Trip matrix written to {datapathO}tripmatrix_parcels_{label}.txt'), log_file.write(f'Trip matrix written to {datapathO}tripmatrix_{label}.txt\n')
    
            deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
            deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
            
            for tod in range(24):                
                print(f'\t Also generating trip matrix for TOD {tod}...'), log_file.write(f'\t Also generating trip matrix for TOD {tod}...\n')
                output = deliveries[(deliveries['TripDepTime'] >= tod) & (deliveries['TripDepTime'] < tod+1)].copy()
                output['N_TOT'] = 1
                
                if len(output) > 0:
                    # Gebruik deze dummies om het aantal ritten per HB te bepalen, voor elk logistiek segment, voertuigtype en totaal
                    pivotTable = pd.pivot_table(output, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
                    pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
                    pivotTable['DEST'] = [x[1] for x in pivotTable.index]
                    pivotTable = pivotTable[cols]
                else:
                    pivotTable = pd.DataFrame(columns=cols)
                pivotTable.to_csv(f"{datapathO}tripmatrix_parcels_{label}_TOD{tod}.txt", index=False, sep='\t')
                
    
            
        # --------------------------- End of module -------------------------------
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Parcel Scheduling: Done")
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
                root.update_statusbar("Parcel Scheduling: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]



def parcelSched(delivToSched, skim, originZones, depotIDs, depotDict, maxVehicleLoad, dropOffTime, \
                schedList, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, tourType, label):
    '''
    Create the parcel schedules and store them in a DataFrame
    '''
    toursPerDepot = []  #takes count of tours; needed when results are put into a dataframe in the end; one number per depot
    nVanTrips     = 0       
    nOrigins      = len(originZones)
    nZones        = int(len(skimTravTime)**0.5)

    print('\t0%', end='\r')
    
    for currDepot in range(nOrigins) :  #rename currDepot to currOrigin
        
        # Create DataFrame to store zones that need to be served and the remaining no of parcels; equivalent to universalChoiceSet
        availDest = pd.DataFrame(delivToSched[delivToSched['Depot']==depotIDs[currDepot]]) #take deliveries for current origin from df with all requried deliveries
        
        # Initialize the remaining number of parcels yet to deliver
        availDest['remainingParcels'] = np.array(availDest['Parcels'].copy())
     
        # Initialise list to fill with van tours; #tours is #parcels leaving the depot divided by van capacity, rounded up
        vanTours    = [ [] for x in range(int(np.ceil(availDest['Parcels'].sum()/maxVehicleLoad)))]
        nVanTours   = len(vanTours)     # Counter that keeps being updated
        toursPerDepot.append(nVanTours) # List to which number for current depot is appended every time
        orig        = originZones[currDepot]     # First origin is the zone of the current pNode, then it gets updated
             
        # Start filling array for current depot with van tours
        for vanTour in range(nVanTours) :
            nVanTours       += 1            
            orig             = originZones[currDepot] # Reset start to depot for each new tour
            currVanCapacity  = maxVehicleLoad         # Reset capacity to max for a new van; this will be the counter for the while loop: when it reaches zero, the loop ends
            currentSkim      = pd.DataFrame(np.arange(1), columns=['zone'])
            depTimeTour      = np.where(parcelDepTime > np.random.rand())[0][0]
            
            while currVanCapacity > 0: #while van tour is not finished, trips are added
           
                # Array with all destinations that still need to be served from depot
                dests = np.array(availDest.loc[availDest['remainingParcels']!=0,'Dest'], dtype=int)
    
                if len(dests) > 0:
                
                    #the depot is not completed; update skim and find new minimum travel distance
                    currentSkim       = pd.DataFrame(dests, columns=['zone'])                                 #select zones that still receive parcels
                    currentSkim['TT'] = skimDistance[(orig-1)*nZones+(dests-1)]                                       #make skim from last zone to all remaining zones
                    currentDest       = currentSkim['zone'].loc[np.argmin(np.array(currentSkim['TT']))]       #find next destination              
                    currNoParcels     = int(availDest['remainingParcels'][availDest['Dest']==currentDest])      #store no. of parcels that go to selected destination
    
                    if currVanCapacity - currNoParcels >= 0 : # If vehicle is not being filled (= if remaining van capacity is sufficient)
                        # Make trip from orig to currentDest with currNoParcels
                        vanTrip         = [orig, currentDest, currNoParcels]
                        vanTours[int(vanTour)].append(vanTrip)
                        nVanTrips       += 1          
                        availDest.at[availDest['Dest']==currentDest,'remainingParcels'] = 0 # Remove assigned parcels from available destinations
                        currVanCapacity -= currNoParcels      # Adjust van capacity
                        orig             = currentDest.copy() # Set origin for next trip to current location
        
                    elif currVanCapacity - currNoParcels < 0 :  # Van capacity is not sufficient for this delivery
    
                        # First, add delivery trip
                        vanTrip = [orig, currentDest, currVanCapacity]
                        vanTours[int(vanTour)].append(vanTrip)
    
                        # Then add return trip to depot
                        returnTrip  = [currentDest, originZones[currDepot], 0]
                        vanTours[int(vanTour)].append(returnTrip)
    
                        nVanTrips      += 2   # Two trips are added: last (incomplete) delivery and return to depot      
                        availDest.at[availDest['Dest']==currentDest,'remainingParcels'] = currNoParcels-currVanCapacity # Remove scheduled parcels from available destinations
                        currVanCapacity = 0   # Adjust capacity to zero so that no more trips are added to the tour

                # Depot is completed, return trip is initiated even if the vehicle is not full 
                else:                         
                    vanTrip = [currentDest, originZones[currDepot], 0]
                    vanTours[int(vanTour)].append(vanTrip)  
                    nVanTrips += 1
                    break
        
            # 2-opt posterior tour improvement
            currentTour = np.array(vanTours[vanTour], dtype=int)[:,0]
            currentTour = np.append(currentTour, vanTours[vanTour][-1][1])
            currentTourDist = np.sum(skimDistance[(currentTour[:-1]-1)*nZones + currentTour[1:]-1])
            nStops = len(currentTour)
            
            if nStops > 4:
                anySwaps = False
                nParcelsPerTrip = np.array(vanTours[vanTour], dtype=int)[:,2]
                
                for shiftLocA in range(1, nStops-1):
                    for shiftLocB in range(1,nStops-1):
                        if shiftLocA != shiftLocB:
                            swappedTour             = currentTour.copy()
                            swappedTour[shiftLocA]  = currentTour[shiftLocB]
                            swappedTour[shiftLocB]  = currentTour[shiftLocA]
                            swappedTourDist = np.sum(skimDistance[(swappedTour[:-1]-1)*nZones + swappedTour[1:]-1])
                            
                            if swappedTourDist < currentTourDist:
                                currentTour = swappedTour.copy()
                                temp = int(nParcelsPerTrip[shiftLocA-1])
                                nParcelsPerTrip[shiftLocA-1] = nParcelsPerTrip[shiftLocB-1]
                                nParcelsPerTrip[shiftLocB-1] = temp
                                anySwaps = True
                                currentTourDist = swappedTourDist
                                
                if anySwaps:
                    vanTours[vanTour] = []
                    for i in range(len(currentTour)-1):
                        vanTours[vanTour].append([currentTour[i], currentTour[i+1], nParcelsPerTrip[i]])
                        
            # Determine delivery times
            travTime = skimTravTime[(vanTours[vanTour][0][0]-1)*nZones + (vanTours[vanTour][0][1]-1)] / 3600
            vanTours[vanTour][0].append(travTime)
            vanTours[vanTour][0].append(depTimeTour)
            vanTours[vanTour][0].append(depTimeTour)
            vanTours[vanTour][0].append(vanTours[vanTour][0][5] + vanTours[vanTour][0][3])
            
            for i in range(1,len(vanTours[vanTour])):
                travTime = skimTravTime[(vanTours[vanTour][i][0]-1)*nZones + (vanTours[vanTour][i][1]-1)] / 3600
                vanTours[vanTour][i].append(travTime)
                vanTours[vanTour][i].append(depTimeTour)
                vanTours[vanTour][i].append(vanTours[vanTour][i-1][6])
                vanTours[vanTour][i].append(vanTours[vanTour][i][5] + vanTours[vanTour][i][3])            
        
        #add tours to schedule list before going to next depot/ucc
        schedList[currDepot] = vanTours
    
        print('\t' + str(int(round(((currDepot+1)/nOrigins)*100,0))) + '%', end='\r')
    
    # ------------------------------ Create return table ---------------------- 
    deliveriesCols = ['TourType', 'CEP','Depot_ID', 'Tour_ID', 'Trip_ID', 'Unique_ID', 'O_zone', 'D_zone', 'N_parcels', 'Traveltime', 'TourDepTime', 'TripDepTime', 'TripEndTime']
    deliveries = np.zeros((nVanTrips, len(deliveriesCols)), dtype=object)

    tripcount = 0
    for startLoc in range(len(originZones)): #iterator must be 0 to len(schedList) so that everything is read
        depot = depotIDs[startLoc]
        
        for tour in range(toursPerDepot[startLoc]):
            for trip in range(len(schedList[startLoc][tour])):
                deliveries[tripcount, 0] = tourType                                 # depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                if tourType <= 1:
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]                # Name of the couriers
                else:
                    deliveries[tripcount, 1] = 'ConsolidatedUCC'                    # Name of the couriers
                deliveries[tripcount, 2] = depot                                    # Depot_ID
                deliveries[tripcount, 3] = f'{depot}_{tour}'                        # Tour_ID
                deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'                 # Trip_ID
                deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'      # Unique ID under consideration of tour type
                deliveries[tripcount, 6] = schedList[startLoc][tour][trip][0]       # Origin
                deliveries[tripcount, 7] = schedList[startLoc][tour][trip][1]       # Destination
                deliveries[tripcount, 8] = schedList[startLoc][tour][trip][2]       # Number of parcels
                deliveries[tripcount, 9] = schedList[startLoc][tour][trip][3]       # Travel time in hrs
                deliveries[tripcount,10] = schedList[startLoc][tour][trip][4]       # Departure of tour from depot
                deliveries[tripcount,11] = schedList[startLoc][tour][trip][5]       # Departure time of trip
                deliveries[tripcount,12] = schedList[startLoc][tour][trip][6]       # End of trip/start of next trip if there is another one                
                tripcount += 1

    deliveries = pd.DataFrame(deliveries, columns=deliveriesCols)
    dtypes =  {'TourType':int, 'CEP':str,'Depot_ID':int, 'Tour_ID':str, 'Trip_ID':str, 'Unique_ID':str, \
               'O_zone':int, 'D_zone':int, 'N_parcels':int, 'Traveltime':float, 'TourDepTime':float, 'TripDepTime':float, 'TripEndTime':float}
    for col in range(len(deliveriesCols)):
        deliveries[deliveriesCols[col]] = deliveries[deliveriesCols[col]].astype(dtypes[deliveriesCols[col]])

    vehTypes  = ['Van', 'Van', 'Van', 'LEVV']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC']
    destTypes = ['HH', 'UCC', 'HH', 'HH']

    deliveries['VehType' ] = vehTypes[tourType]
    deliveries['OrigType'] = origTypes[tourType]
    deliveries['DestType'] = destTypes[tourType]

    return deliveries
 
    
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    INPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/'
    OUTPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunUCC2030H/'
    PARAMFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/'
    SKIMTIME = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2030/skimTijd_REF.mtx'
    SKIMDISTANCE = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2030/skimAfstand_REF.mtx'
    LINKS = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/links_v5_2030H.shp'
    NODES = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/nodes_v5.shp'
    ZONES = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/Zones_v5.shp'
    SEGS = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/SEGS2030H_verrijkt.csv'
    DISTRIBUTIECENTRA = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/distributieCentra.csv'
    COST_VEHTYPE = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/Cost_VehType_2030H.csv'
    COST_SOURCING = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/Cost_Sourcing_2030H.csv'
    COMMODITYMATRIX = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/CommodityMatrixNUTS3_2030H.csv'
    PARCELNODES = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/parcelNodes_v2.shp'
    CEP_SHARES = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2030/CEPshares.csv'
    MRDH_TO_NUTS3 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/MRDHtoNUTS32013.csv'
    NUTS3_TO_MRDH = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/NUTS32013toMRDH.csv'
    PARCELS_PER_HH = 0.224
    PARCELS_PER_EMPL = 0.082
    PARCELS_MAXLOAD = 180.0
    PARCELS_DROPTIME = 120.0
    PARCELS_SUCCESS_B2C = 0.75
    PARCELS_SUCCESS_B2B = 0.95
    PARCELS_GROWTHFREIGHT = 2.0
    YEARFACTOR = 209.0
    NUTSLEVEL_INPUT = 3.0
    IMPEDANCE_SPEED = 'V_FR_OS'
    N_CPU = ''
    SHIPMENTS_REF = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunUCC2030H/Shipments_REF.csv'
    SELECTED_LINKS = ''
    LABEL = 'UCC'
    
    MODULES = ['SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, \
            LINKS, NODES, ZONES, SEGS, \
            DISTRIBUTIECENTRA, COST_VEHTYPE,COST_SOURCING, CEP_SHARES, \
            COMMODITYMATRIX, PARCELNODES, MRDH_TO_NUTS3, NUTS3_TO_MRDH, \
            PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
            PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED, N_CPU, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", \
                  "LINKS", "NODES", "ZONES", "SEGS", \
                  "DISTRIBUTIECENTRA", "COST_VEHTYPE","COST_SOURCING", "CEP_SHARES",\
                  "COMMODITYMATRIX", "PARCELNODES", "MRDH_TO_NUTS3", "NUTS3_TO_MRDH", \
                  "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
                  "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B",  "PARCELS_GROWTHFREIGHT", \
                  "YEARFACTOR", "NUTSLEVEL_INPUT", \
                  "IMPEDANCE_SPEED", "N_CPU", \
                  "SHIPMENTS_REF", "SELECTED_LINKS", \
                  "LABEL", \
                  "MODULES"]
     
    varDict = {}
    for i in range(len(args)):
        varDict[varStrings[i]] = args[i]
        
    # Run the module
    main(varDict)

    
