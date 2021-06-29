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
        segsPath         = varDict['SEGS']
        label            = varDict['LABEL']
        
        dropOffTimeSec = varDict['PARCELS_DROPTIME']
        maxVehicleLoad = varDict['PARCELS_MAXLOAD']
        
        doCrowdShipping = (str(varDict['CROWDSHIPPING']).upper() == 'TRUE')
        
        exportTripMatrix = True
        
        log_file = open(datapathO + "Logfile_ParcelScheduling.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")

        if root != '':
            root.progressBar['value'] = 0.1
            
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

        if root != '':
            root.progressBar['value'] = 0.3
            
        # System input for scheduling
        parcelDepTime = np.array(pd.read_csv(f"{datapathI}departureTimeParcelsCDF.csv").iloc[:,1])        
        dropOffTime   = dropOffTimeSec/3600    
        skimTravTime  = read_mtx(skimTravTimePath)
        skimDistance  = read_mtx(skimDistancePath)
        nZones        = int(len(skimTravTime)**0.5)

        if root != '':
            root.progressBar['value'] = 0.9
            
        # Intrazonal impedances
        skimTravTime = skimTravTime.reshape(nZones,nZones)
        for i in range(nZones):
            skimTravTime[i,i] = 0.7 * np.min(skimTravTime[i,skimTravTime[i,:]>0])
        skimTravTime = skimTravTime.flatten()
        skimDistance = skimDistance.reshape(nZones,nZones)
        for i in range(nZones):
            skimDistance[i,i] = 0.7 * np.min(skimDistance[i,skimDistance[i,:]>0])
        skimDistance = skimDistance.flatten()

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

        # Input data and parameters for the crowdshipping use case
        if doCrowdShipping == True:
            nFirstZonesCS = 5925
            
            parcelShareCRW = varDict['CRW_PARCELSHARE']
            
            modeParamsCRW = pd.read_csv(varDict['CRW_MODEPARAMS'], index_col=0, sep=',')
            
            modes = {'fiets': {}, 'auto': {}, }
            modes['fiets']['willingness' ] = modeParamsCRW.at['BIKE','WILLINGNESS']
            modes['auto' ]['willingness' ] = modeParamsCRW.at['CAR', 'WILLINGNESS']
            modes['fiets']['dropoff_time'] = modeParamsCRW.at['BIKE','DROPOFFTIME']
            modes['auto' ]['dropoff_time'] = modeParamsCRW.at['CAR', 'DROPOFFTIME']
            modes['fiets']['VoT'         ] = modeParamsCRW.at['BIKE','VOT']            
            modes['auto' ]['VoT'         ] = modeParamsCRW.at['CAR', 'VOT']
            
            modes['fiets']['relative_extra_parcel_dist_threshold'] = modeParamsCRW.at['BIKE','RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD']
            modes['auto' ]['relative_extra_parcel_dist_threshold'] = modeParamsCRW.at['CAR', 'RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD']
            
            modes['fiets']['OD_path'  ] = varDict['CRW_PDEMAND_BIKE']
            modes['fiets']['skim_time'] = skimDistance / 1000 / 12 * 3600
            modes['fiets']['n_trav'   ] = 0
            modes['auto']['OD_path'  ] = varDict['CRW_PDEMAND_CAR']
            modes['auto']['skim_time'] = skimTravTime
            modes['auto']['n_trav'   ] = 0            
            
            for mode in modes:
                modes[mode]['OD_array']  = read_mtx(modes[mode]['OD_path']).reshape(nIntZones, nIntZones)
                modes[mode]['OD_array'] *= modes[mode]['willingness']
                modes[mode]['OD_array']  = np.round(modes[mode]['OD_array'], 0)
                modes[mode]['OD_array']  = np.array(modes[mode]['OD_array'], dtype=int)
                modes[mode]['OD_array']  = modes[mode]['OD_array'][:nFirstZonesCS,:]
                modes[mode]['OD_array']  = modes[mode]['OD_array'][:,:nFirstZonesCS]
                
            # Which zones are located in which municipality
            zone_gemeente_dict = dict(np.transpose(np.vstack( (np.arange(1,nIntZones+1), zones['Gemeentena']) )))
            
            segs       = pd.read_csv(segsPath)
            segs.index = segs['zone']
            segsDetail = np.array(segs['6: detail'])

        if root != '':
            root.progressBar['value'] = 1.0
            

        # ------------------- Crowdshipping use case -----------------------------------
        print('Crowdshipping use case...'), log_file.write('Crowdshipping use case...\n')
        
        if doCrowdShipping == True:
            start_time_cs = time.time()
            
            print('\tGet parcel demand...'), log_file.write('\tGet parcel demand...\n')
            nParcels = len(parcels)
            
            randSelectionCRW = ((np.random.rand(nParcels) < parcelShareCRW) & (parcels['D_zone'] < nFirstZonesCS))
            indicesCRW = np.where( randSelectionCRW)[0]
            indicesREF = np.where(~randSelectionCRW)[0]
            
            # The parcels to use for crowdshipping
            parcelsCRW = parcels.loc[indicesCRW, :]
            parcelsCRW.index = parcelsCRW['Parcel_ID']
            
            # The parcels to ship regularly by parcel couriers
            parcels = parcels.loc[indicesREF, :]
            parcels.index = parcels['Parcel_ID']
            
            # Add number of parcels to zonal data
            zones['parcels'  ]   = [np.sum(parcels[   'D_zone']==(i+1)) for i in range(nIntZones)]
            zones['parcelsCS']   = [np.sum(parcelsCRW['D_zone']==(i+1)) for i in range(nIntZones)]
            nParcels           = int(zones[:nFirstZonesCS]["parcels"].sum()) 
            nParcelsCS         = int(zones[:nFirstZonesCS]["parcelsCS"].sum())
                        
            # Dictionary of all zones per municipality
            gemeente_zone_dict = {}
            gemeente_id_dict = {}
            id_gemeente_dict = {}
            count = 0
            for gemeente in np.unique(zones[:nFirstZonesCS]['Gemeentena']):
                gemeente_zone_dict[gemeente] = np.where(zones['Gemeentena'] == gemeente)[0]
                gemeente_id_dict[gemeente] = count
                id_gemeente_dict[count] = gemeente
                count += 1
            
            # Initialize an array for the crowdshipping parcels
            parcelsCS_cols = {'id': 0,              'orig': 1,              'dest': 2,          'orig_skim': 3, 'dest_skim': 4,
                              'gemeente': 5,        'X_ORIG': 6,            'Y_ORIG': 7,        'X_DEST': 8,    'Y_DEST': 9,
                              'TravelTime_car': 10, 'TravelDistance': 11,   'vector': 12,       'status': 13,   'traveller':14,
                              'modal choice': 15,   'detour_time': 16,      'detour_dist': 17,  'compensation': 18}            
            parcelsCS_array = np.zeros((nParcelsCS,len(parcelsCS_cols)), dtype=object)
            
            # Dictionary from (key) parcel ID as used in loop below to (value) parcel ID as used in input parcel demand file
            parc_id_dict = {}
            
            # Create object for crowdshipping parcels
            count = 0
            for i in range(nFirstZonesCS):
                nParcelsZone = int(zones.at[zoneDict[i+1],'parcelsCS'])
                parc_ids     = np.array(parcelsCRW.loc[parcelsCRW['D_zone']==(i+1), 'Parcel_ID'])
                
                if nParcelsZone > 0:
                    dest_skim = i + 1
                    dest      = zoneDict[dest_skim]        
                    gemeente  = zones.at[dest, 'Gemeentena']
                    x_dest    = zonesX[dest]
                    y_dest    = zonesY[dest]
                    status    = "ordered"
                    possible_origins = gemeente_zone_dict[gemeente]
                    
                    ratio_origins  = np.zeros(nIntZones)
                    ratio_origins[possible_origins] = segsDetail[possible_origins]
                    ratio_origins  = np.cumsum(ratio_origins)
                    ratio_origins /= ratio_origins[-1]
            
                    for n in range(nParcelsZone):
                        parc_id   = n + count
                        orig_skim = np.where(ratio_origins>=np.random.rand())[0][0] + 1
                        orig      = zoneDict[orig_skim]
                        x_orig    = zonesX[orig]
                        y_orig    = zonesY[orig]
                        trav_time = skimTravTime[(orig_skim-1) * nZones + (dest_skim-1)] / 3600
                        trav_dist = skimDistance[(orig_skim-1) * nZones + (dest_skim-1)] / 1000
                        vector    = [x_dest-x_orig, y_dest-y_orig]
                        parcelsCS_array[n+count] = [parc_id, orig, dest, orig_skim, dest_skim, gemeente, 
                                                    x_orig, y_orig, x_dest, y_dest, trav_time, trav_dist, vector, status,
                                                    0,0,0,0,0]                        
                        parc_id_dict[parc_id] = parc_ids[n]
                        
                count += nParcelsZone
            
            # Recode municipalities to numberic IDs for faster checking in parcel assignment loop
            parcelsCS_array[:,5] = np.array([gemeente_id_dict[x] for x in parcelsCS_array[:,5]], dtype=int)
            
            # Place crowdshipping parcel in DataFrame with headers
            parcelsCS_df = pd.DataFrame(parcelsCS_array, columns=parcelsCS_cols)

            if root != '':
                root.progressBar['value'] = 2.0
                
            print('\tGet potential crowdshippers...'), log_file.write('\tGet potential crowdshippers...\n')
            
            # Editing OD-matrices of passenger travellers that are willing to crowdship
            trav_array_cols = {'id': 0, 'orig': 1, 'dest': 2, 'orig_skim': 3, 'dest_skim': 4, 'vector': 5, 'gemeenten': 6, 'parcel': 7, 'status': 8}
            for mode in modes:
                OD_array = modes[mode]['OD_array']
                trav_array = np.zeros((OD_array.sum().sum(),len(trav_array_cols)), dtype=object)
            
                start_id = sum(d['n_trav'] for d in modes.values() if d)
                count = 0
                       
                for i, row in enumerate(OD_array):        
                    for j in np.where(row>0)[0]:        
                        if i != j:
                            n = row[j]                
                            orig_skim = i + 1
                            dest_skim = j + 1
                            orig = int(zoneDict[orig_skim])
                            dest = int(zoneDict[dest_skim])
                            vector = [zonesX[dest] - zonesX[orig], zonesY[dest] - zonesY[orig]]
                            gemeente = [zone_gemeente_dict[orig_skim], zone_gemeente_dict[dest_skim]]
            
                            for N in range(n):
                                trav_id = N + count + start_id
                                trav_array[N + count] = [trav_id, orig, dest, orig_skim, dest_skim, vector, gemeente, 0, 0]
                
                            count += n
                            
                trav_array = trav_array[~np.all(trav_array == 0, axis=1)]
                
                # Recode municipalities to numberic IDs for faster checking in parcel assignment loop
                trav_array[:,6] = [[gemeente_id_dict[x[0]], gemeente_id_dict[x[1]]] for x in trav_array[:,6]]
                
                modes[mode]['n_trav'] = int(len(trav_array))
                modes[mode]['trav_array'] = trav_array

            if root != '':
                root.progressBar['value'] = 4.0
                
            # Assign parcels to crowdshippers
            parcelsToBeAssigned = np.array([True for i in range(nParcelsCS)])
            nParcelsAssigned = 0
            
            # Variables to keep track of progress
            nTravellersTotal = sum([len(modes[mode]['trav_array']) for mode in modes])
            travellerCount   = 0
            
            for mode in modes:
                print("\tAssigning " + str(mode) + " travellers to parcels...")
                log_file.write("\tAssigning " + str(mode) + " travellers to parcels...\n")
                
                skimTravTime = modes[mode]['skim_time']
                dropoff_time = modes[mode]['dropoff_time']
                VoT          = modes[mode]['VoT']
                
                nTravellers = len(modes[mode]['trav_array'])
                
                # Initialize variable with orig/dest municipality of previously checked traveller 
                prevMunicipality = [-1, -1]
                
                for i, traveller in enumerate(modes[mode]['trav_array']):
                    
                    # Stop in the case all crowdshipping-eligible parcels are assigned to a bringer
                    if nParcelsAssigned == nParcelsCS:
                        break
                    
                    offers_dict  = {}
                    offers2_dict = {}
                    trav_orig = traveller[3]
                    trav_dest = traveller[4]
                    trip_dist = skimDistance[(trav_orig-1)*nZones+(trav_dest-1)] / 1000
                    trip_time = skimTravTime[(trav_orig-1)*nZones+(trav_dest-1)] / 3600
                
                    # Boolean: Parcels for which no carrier has been found yet
                    checkUnassigned = parcelsToBeAssigned
                    
                    # Boolean: Parcels with a reasonable distance in relation to the traveller's trip distance
                    checkDistance = ((trip_dist / parcelsCS_array[:,11] < 4  ) & (trip_dist / parcelsCS_array[:,11] > .5 ))
                                        
                    # Boolean: Parcels within the municipality of traveller's origin / destination
                    # (Only needs to be recalculated if the traveller has different orig/dest from previous traveller)
                    if prevMunicipality != traveller[6]:
                        checkMunicipality = ((parcelsCS_array[:,5] == traveller[6][0]) | (parcelsCS_array[:,5] == traveller[6][1]))
                    
                    # Now select the parcels that comply to the above three boolean checks
                    parcelsToConsider = parcelsCS_array[(checkUnassigned & checkMunicipality & checkDistance)]
                    
                    # Determine detour due to delivering parcel
                    parc_orig = np.array(parcelsToConsider[:,3], dtype=int)
                    parc_dest = np.array(parcelsToConsider[:,4], dtype=int)
                    dist_traveller_parcel = skimDistance[(trav_orig-1)*nZones+(parc_orig-1)] / 1000
                    dist_parcel_trip      = skimDistance[(parc_orig-1)*nZones+(parc_dest-1)] / 1000
                    dist_customer_end     = skimDistance[(parc_dest-1)*nZones+(trav_dest-1)] / 1000
                    CS_trip_dist          = (dist_traveller_parcel + dist_parcel_trip + dist_customer_end)           
                    traveller_detour  = CS_trip_dist - trip_dist
                    extra_parcel_dist = traveller_detour - dist_parcel_trip
                    relative_extra_parcel_dist = extra_parcel_dist / dist_parcel_trip
                    
                    # Determine compensation offered to traveller
                    for trip in np.where(relative_extra_parcel_dist < modes[mode]['relative_extra_parcel_dist_threshold'])[0]:        
                        CS_compensation = np.log( (dist_parcel_trip[trip]) + 5)
                        offers_dict[parcelsToConsider[trip,0]] = {'distance':   dist_parcel_trip[trip], 
                                                                 'rel_detour':  relative_extra_parcel_dist[trip],
                                                                 'compensation':CS_compensation}
                    
                    # Traveller chooses the parcel to ship
                    if offers_dict:
                        offered_parcels = sorted(offers_dict, key=lambda x: (offers_dict[x]['rel_detour']))[:3]
                                 
                        # Search for best parcel
                        for parcel in offered_parcels:
                            parc_orig = parcelsCS_array[parcel,3]
                            parc_dest = parcelsCS_array[parcel,4]
                            traveller_detour_time =     ( skimTravTime[(trav_orig-1)*nZones+(parc_orig-1)] / 3600 +
                                                          skimTravTime[(parc_orig-1)*nZones+(parc_dest-1)] / 3600 +
                                                          skimTravTime[(parc_dest-1)*nZones+(trav_dest-1)] / 3600 -
                                                          trip_time )
                            CS_utility = offers_dict[parcel]['compensation'] / ( (traveller_detour_time + 2 * dropoff_time) )
                            offers2_dict[parcel] = {'utility': CS_utility}
            
                        best_parcel = offered_parcels[0]
                        
                        # Traveller chooses whether to ship this 'best' parcel based on value of time
                        if offers2_dict[best_parcel]['utility'] > VoT:
                            modes[mode]['trav_array'][i,7] = int(best_parcel)
                            modes[mode]['trav_array'][i,8] = str('shipping')
                            parcelsToBeAssigned[best_parcel] = False
                            nParcelsAssigned += 1
                            
                            parcelsCS_array[best_parcel,13] = 'carrier found'
                            parc_orig = parcelsCS_array[best_parcel,3]
                            parc_dest = parcelsCS_array[best_parcel,4]
                            traveller_detour_time = (skimTravTime[(trav_orig-1) * nZones + (parc_orig-1)] +
                                                     skimTravTime[(parc_orig-1) * nZones + (parc_dest-1)] +
                                                     skimTravTime[(parc_dest-1) * nZones + (trav_dest-1)] -
                                                     skimTravTime[(trav_orig-1) * nZones + (trav_dest-1)]) / 3600
                            traveller_detour_distance = (skimDistance[(trav_orig-1) * nZones + (parc_orig-1)] +
                                                         skimDistance[(parc_orig-1) * nZones + (parc_dest-1)] +
                                                         skimDistance[(parc_dest-1) * nZones + (trav_dest-1)] -
                                                         skimDistance[(trav_orig-1) * nZones + (trav_dest-1)]) / 1000
                            parcelsCS_array[best_parcel,14] = traveller[0]
                            parcelsCS_array[best_parcel,15] = mode
                            parcelsCS_array[best_parcel,16] = traveller_detour_time
                            parcelsCS_array[best_parcel,17] = traveller_detour_distance
                            parcelsCS_array[best_parcel,18] = offers_dict[best_parcel]['compensation']
                    
                    travellerCount += 1
                    prevMunicipality = traveller[6]
                    
                    if i % int(nTravellers/20) == 0:
                        print('\t\t' + str(int(round(i / nTravellers*100,0))) + "% ", end='\r')
                        if root != '':
                            root.progressBar['value'] = 4.0 + (52.0 - 4.0) * (travellerCount / nTravellersTotal)                       
                            
            # Recode municipalities back to string
            parcelsCS_array[:,5] = [id_gemeente_dict[x] for x in parcelsCS_array[:,5]]
            
            # Put parcels back in DataFrame again
            parcelsCS_df = pd.DataFrame(parcelsCS_array, columns=parcelsCS_cols)
            
            # Parcels that are not assigned to an occassional carrier will need to be scheduled 
            # in the regular scheduling procedure
            unassignedParcelIDs = np.array(parcelsCS_df.loc[parcelsCS_df['status']!='carrier found', 'id'])
            unassignedParcelIDs = [parc_id_dict[x] for x in unassignedParcelIDs]
            parcels = parcels.append(parcelsCRW.loc[unassignedParcelIDs, :])            
        
            print("\tWriting crowdshipping output to CSV and GeoJSON...")
            log_file.write("\tWriting crowdshipping output to CSV and GeoJSON...\n")

            # Write CSV
            parcelsCS_df.to_csv(datapathO + f'ParcelDemand_{label}_Crowdshipping.csv', index=False)
            
            # Write GeoJSON
            for mode in modes:       
                trav_array = modes[mode]['trav_array']                
                tours = pd.DataFrame()
                
                for i, traveller in enumerate(trav_array[trav_array[:,8] == 'shipping']):
                    trav_ORIG = traveller[1]
                    trav_DEST = traveller[2]
                    parc_ORIG = parcelsCS_array[int(traveller[7]),1]
                    parc_DEST = parcelsCS_array[int(traveller[7]),2]
                    trav_orig = traveller[3]
                    trav_dest = traveller[4]
                    parc_orig = parcelsCS_array[int(traveller[7]),3]
                    parc_dest = parcelsCS_array[int(traveller[7]),4]
                    
                    for j in range(3):
                        tours.at[i*3+j, 'TOUR_ID'] = i
                        tours.at[i*3+j, 'TRIP_ID'] = str(i) + "_" + str(j)
                        tours.at[i*3+j, 'traveller_ID'] = traveller[0]
                        tours.at[i*3+j, 'parcel_ID'] = traveller[7]
                        tours.at[i*3+j, 'mode'] = mode
                        if j == 0:
                            tours.at[i*3+j, 'skim_dist']  = skimDistance[(trav_orig-1) * nZones + (parc_orig-1)] / 1000
                            tours.at[i*3+j, 'ORIG']  = trav_ORIG
                            tours.at[i*3+j, 'DEST']  = parc_ORIG
                        if j == 1:
                            tours.at[i*3+j, 'skim_dist']  = skimDistance[(parc_orig-1) * nZones + (parc_dest-1)] / 1000
                            tours.at[i*3+j, 'ORIG']  = parc_ORIG
                            tours.at[i*3+j, 'DEST']  = parc_DEST
                        if j == 2:
                            tours.at[i*3+j, 'skim_dist']  = skimDistance[(parc_dest-1) * nZones + (trav_dest-1)] / 1000
                            tours.at[i*3+j, 'ORIG']  = parc_DEST
                            tours.at[i*3+j, 'DEST']  = trav_DEST
                            
                if not tours.empty:
                    for i, ORIG in enumerate(tours['ORIG']):
                        tours.at[i, 'X_ORIG'] = zones.loc[ORIG]['X']
                        tours.at[i, 'Y_ORIG'] = zones.loc[ORIG]['Y']
                    for i, DEST in enumerate(tours['DEST']):
                        tours.at[i, 'X_DEST'] = zones.loc[DEST]['X']
                        tours.at[i, 'Y_DEST'] = zones.loc[DEST]['Y']
                    
                    #----- GeoJSON ---
                    Ax = np.array(tours['X_ORIG'], dtype=str)
                    Ay = np.array(tours['Y_ORIG'], dtype=str)
                    Bx = np.array(tours['X_DEST'], dtype=str)
                    By = np.array(tours['Y_DEST'], dtype=str)
                    nTrips = len(tours)
                    
                    with open(datapathO + f'ParcelSchedule_{label}_Crowdshipping_{mode}.geojson', 'w') as geoFile:
                        geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
                        for i in range(nTrips-1):
                            outputStr = ""
                            outputStr = outputStr + '{ "type": "Feature", "properties": '
                            outputStr = outputStr + str(tours.loc[i,:].to_dict()).replace("'",'"')
                            outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                            outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                            outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
                            geoFile.write(outputStr)
                                
                        # Bij de laatste feature moet er geen komma aan het einde
                        i += 1
                        outputStr = ""
                        outputStr = outputStr + '{ "type": "Feature", "properties": '
                        outputStr = outputStr + str(tours.loc[i,:].to_dict()).replace("'",'"')
                        outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                        outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                        outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
                        geoFile.write(outputStr)
                        geoFile.write(']\n')
                        geoFile.write('}')
            
            # Print summary of crowdshipping results
            n_parcels_total = nParcels
            n_parcels_CS = len(parcelsCS_array)
            n_parcels_CS_delivered = (parcelsCS_array[:,13] == 'carrier found').sum()
            delivered_percentage =  round(n_parcels_CS_delivered/n_parcels_CS*100,2)
            avg_dist =              round(parcelsCS_array[:,11].mean(),2)
            avg_detour =            round(parcelsCS_array[:,17].sum()/n_parcels_CS_delivered,2)
            total_detour =          int(round(parcelsCS_array[:,17].sum(),-1))
            bike_parcels =          len(parcelsCS_array[parcelsCS_array[:,15]=='fiets'][:,17])
            bike_km =               int(round(parcelsCS_array[parcelsCS_array[:,15]=='fiets'][:,17].sum(),-1))
            bike_km_avg =           round(parcelsCS_array[parcelsCS_array[:,15]=='fiets'][:,17].mean(),2)
            car_parcels =           len(parcelsCS_array[parcelsCS_array[:,15]=='auto'][:,17])
            car_km =                int(round(parcelsCS_array[parcelsCS_array[:,15]=='auto'][:,17].sum(),-1))
            car_km_avg =            round(parcelsCS_array[parcelsCS_array[:,15]=='auto'][:,17].mean(),2)
            avg_compensation =      round(parcelsCS_array[parcelsCS_array[:,13]=='carrier found'][:,18].mean(),2)
            
            print("\tA total of " + str(n_parcels_total) + " parcels ordered in the system. " + str(n_parcels_CS) +" are eligible for CS of which "+ str(n_parcels_CS_delivered) +" have been delivered through CS (" + str(delivered_percentage) + "%)." + "\n" +
                  "\tThe average distance of CS parcel trips is " +str(avg_dist) + "km. For the delivered parcels, the average detour is " + str(avg_detour) +"km." + "\n" +
                  "\tFor the CS deliveries, " + str(total_detour) + " extra kilometers are driven. The detours are distributed to modes as follows" + "\n" +
                  "\tBike: " + str(bike_parcels) + " parcels, total of " + str(bike_km) + "km (" + str(bike_km_avg) + "km average) \n" + 
                  "\tCar:  " + str(car_parcels) + " parcels, total of " + str(car_km) + "km (" + str(car_km_avg) + "km average) \n" +
                  "\tThe average provided compensation for the occasional carriers is " + str(avg_compensation) + " euro.")

            totaltime_cs = round(time.time() - start_time_cs, 2)
            print("\tCrowdshipping calculations took: %s seconds." % (totaltime_cs))
            log_file.write("\tCrowdshipping calculations took: %s seconds.\n" % (totaltime_cs)) 
            
            if root != '':
                root.progressBar['value'] = 55.0        
        
        
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
                
            print('Starting scheduling procedure for parcels from depots to households...')
            log_file.write('Starting scheduling procedure for parcels from depots to households...\n')  
            startValueProgress = 56.0 if doCrowdShipping else 2.0
            endValueProgress   = 70.0
            deliveries = parcelSched(delivFromDepotsToHH, parcelSkim, depotZones, depotIDs, depotDict, 
                                     maxVehicleLoad, dropOffTime, depotListToHH, skimTravTime, skimDistance, 
                                     parcelDepTime, parcelNodesCEP, 0, label,
                                     root, startValueProgress, endValueProgress) 
                    
            print('Starting scheduling procedure for parcels from depots to UCC...')
            log_file.write('Starting scheduling procedure for parcels from depots to UCC...\n')    
            startValueProgress = 70.0
            endValueProgress   = 80.0
            deliveries1 = parcelSched(delivFromDepotsToUCC, parcelSkim, depotZones, depotIDs, depotDict, 
                                      maxVehicleLoad, dropOffTime, depotListToUCC, skimTravTime, skimDistance, 
                                      parcelDepTime, parcelNodesCEP, 1, label,
                                      root, startValueProgress, endValueProgress)
                    
                    
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
            
            startValueProgress = 56.0 if doCrowdShipping else 2.0
            endValueProgress   = 90.0
                    
            print('Starting scheduling procedure for parcels...'), log_file.write('Starting tour formation procedure for parcels...\n')    
            deliveries = parcelSched(delivFromDepots, parcelSkim, depotZones, depotIDs, depotDict, 
                                     maxVehicleLoad, dropOffTime, depotList, skimTravTime, skimDistance, 
                                     parcelDepTime, parcelNodesCEP, 0, label, 
                                     root, startValueProgress, endValueProgress)


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

            startValueProgress = 81.0
            endValueProgress   = 85.0            
            deliveries2  = parcelSched(delivUCCVans, parcelSkimUCC, uccZones, uccDepotIDs, uccZonesDict, 
                                       maxVehicleLoad, dropOffTime, uccListVans, skimTravTime, skimDistance, 
                                       parcelDepTime, parcelNodesCEP, 2, label,
                                       root, startValueProgress, endValueProgress)

            startValueProgress = 85.0
            endValueProgress   = 89.0
            deliveries3  = parcelSched(delivUCCLEVV, parcelSkimUCC, uccZones, uccDepotIDs, uccZonesDict, 
                                       int(maxVehicleLoad/5), dropOffTime, uccListLEVV, skimTravTime, skimDistance, 
                                       parcelDepTime, parcelNodesCEP, 3, label,
                                       root, startValueProgress, endValueProgress)    
    
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

        if root != '':
            root.progressBar['value'] = 91.0
                
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
                    if root != '':
                        root.progressBar['value'] = 91.0 + (98.0 - 91.0) * (i / nTrips)
                    
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



def parcelSched(delivToSched, skim, originZones, depotIDs, depotDict, maxVehicleLoad, dropOffTime,
                schedList, skimTravTime, skimDistance, parcelDepTime, parcelNodesCEP, tourType, label,
                root, startValueProgress, endValueProgress):
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
            travTime = skimTravTime[(vanTours[vanTour][0][0]-1) * nZones + (vanTours[vanTour][0][1]-1)] / 3600
            vanTours[vanTour][0].append(travTime)
            vanTours[vanTour][0].append(depTimeTour)
            vanTours[vanTour][0].append(depTimeTour)
            vanTours[vanTour][0].append(vanTours[vanTour][0][5] + vanTours[vanTour][0][3])
            
            for i in range(1,len(vanTours[vanTour])):
                travTime = skimTravTime[(vanTours[vanTour][i][0]-1) * nZones + (vanTours[vanTour][i][1]-1)] / 3600
                vanTours[vanTour][i].append(travTime)
                vanTours[vanTour][i].append(depTimeTour)
                vanTours[vanTour][i].append(vanTours[vanTour][i-1][6])
                vanTours[vanTour][i].append(vanTours[vanTour][i][5] + vanTours[vanTour][i][3])            
        
        #add tours to schedule list before going to next depot/ucc
        schedList[currDepot] = vanTours
    
        print('\t' + str(int(round(((currDepot+1)/nOrigins)*100,0))) + '%', end='\r')
        
        if root != '':
            root.progressBar['value'] = startValueProgress + (endValueProgress - startValueProgress - 1) * ((currDepot+1) / nOrigins)
    
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

    if root != '':
        root.progressBar['value'] = endValueProgress
                             
    return deliveries
 
    
    
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
    
    CROWDSHIPPING    = 'TRUE'
    CRW_PARCELSHARE  = 0.06
    CRW_MODEPARAMS   = PARAMFOLDER + 'Params_UseCase_CrowdShipping.csv'
    CRW_PDEMAND_CAR  = INPUTFOLDER + 'MRDH_2016_Auto_Etmaal.mtx'
    CRW_PDEMAND_BIKE = INPUTFOLDER + 'MRDH_2016_Fiets_Etmaal.mtx'
    
    SHIPMENTS_REF = ""
    SELECTED_LINKS = ""
    N_CPU = ""
    
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
            CROWDSHIPPING, CRW_PARCELSHARE, CRW_MODEPARAMS, CRW_PDEMAND_CAR, CRW_PDEMAND_BIKE, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED_FREIGHT, IMPEDANCE_SPEED_VAN, N_CPU, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", \
                  "LINKS", "NODES", "ZONES", "SEGS", \
                  "DISTRIBUTIECENTRA", "COST_VEHTYPE","COST_SOURCING", \
                  "COMMODITYMATRIX", "PARCELNODES", "MRDH_TO_NUTS3", "NUTS3_TO_MRDH", \
                  "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
                  "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B",  "PARCELS_GROWTHFREIGHT", \
                  "CROWDSHIPPING", "CRW_PARCELSHARE", "CRW_MODEPARAMS", "CRW_PDEMAND_CAR", "CRW_PDEMAND_BIKE", \
                  "YEARFACTOR", "NUTSLEVEL_INPUT", \
                  "IMPEDANCE_SPEED_FREIGHT", "IMPEDANCE_SPEED_VAN", "N_CPU", \
                  "SHIPMENTS_REF", "SELECTED_LINKS", \
                  "LABEL", \
                  "MODULES"]
     
    varDict = {}
    for i in range(len(args)):
        varDict[varStrings[i]] = args[i]
        
    # Run the module
    root = ''
    main(varDict)

    
