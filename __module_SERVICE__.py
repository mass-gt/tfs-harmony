import pandas as pd
import numpy as np
import time
import datetime
from __functions__ import read_shape, read_mtx, write_mtx

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
        self.root.title("Progress Service Vans")
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
        
        root    = args[0]
        varDict = args[1]
        
        datapathI         = varDict['INPUTFOLDER']
        datapathP         = varDict['PARAMFOLDER']    
        datapathO         = varDict['OUTPUTFOLDER'] 
        pathZones         = varDict['ZONES']
        pathSegs          = varDict['SEGS']
        pathParcelNodes   = varDict['PARCELNODES']
        pathSkimTravTime  = varDict['SKIMTIME']
        pathSkimDistance  = varDict['SKIMDISTANCE']
        pathCostVehType   = varDict['COST_VEHTYPE'] 
        pathDistanceDecay = varDict['SERVICE_DISTANCEDECAY']
        yearFactor        = varDict['YEARFACTOR']
                
        start_time = time.time()
        
        log_file = open(datapathO + "Logfile_ServiceTrips.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
              
        nZones         = 6668
        nInternalZones = 6625
        nExternalZones = 43
        nCOROP         = 40
        
        sectorNames = ['LANDBOUW','INDUSTRIE','DETAIL','DIENSTEN','OVERIG']
        
        tolerance = 0.005
        maxIter = 25
        
               
        
        # --------------------------- Import data -----------------------------------
        
        print('Importing data...'), log_file.write('Importing data...\n')
        if root != '':
            root.update_statusbar('Importing data...')
        
        # Import cost parameters
        costParams  = pd.read_csv(pathCostVehType, index_col=0)
        costPerHour      = costParams.at['Van', 'CostPerH']
        costPerKilometer = costParams.at['Van', 'CostPerKm']
        
        # Import distance decay parameters
        distanceDecay = pd.read_csv(pathDistanceDecay, index_col=0)
        alphaService = distanceDecay.at['Service', 'ALPHA']
        betaService  = distanceDecay.at['Service', 'BETA']
        alphaBouw = distanceDecay.at['Construction', 'ALPHA']
        betaBouw  = distanceDecay.at['Construction', 'BETA']
        
        # Import zone shapefile
        zones = read_shape(pathZones)
        zoneDict = {}
        for i in range(nInternalZones):
            zoneDict[i] = zones.at[i,'AREANR']
        for i in range(nExternalZones):
            zoneDict[nInternalZones+i] = 99999901 + i
        invZoneDict = dict((v, k) for k, v in zoneDict.items())
        
        # Import socio economic data
        segs = pd.read_csv(pathSegs, sep=',')
        segs = segs.sort_values('zone')
        segs.index = segs['zone']
        
        # Import regression coefficients
        regrCoeffs = pd.read_csv(datapathP + 'Params_PA_SERVICE.csv', sep=',', index_col=[0])
        
        # Which MRDH zones form which COROP region
        nCOROP = 40
        MRDHtoCOROP = pd.read_csv(datapathP + 'MRDHtoCOROP.csv', sep=',')        
        MRDHwithinCOROP = {}
        for i in range(1,1+nCOROP):
            MRDHwithinCOROP[i] = np.array(MRDHtoCOROP.loc[MRDHtoCOROP['COROP']==i,'AREANR'])
            
        # Parcel nodes
        parcelNodes = read_shape(pathParcelNodes)    

        if root != '':
            root.progressBar['value'] = 0.5
            
        # Skim with travel times and distances
        skimTravTime = read_mtx(pathSkimTravTime)
        skimDistance = read_mtx(pathSkimDistance)

        if root != '':
            root.progressBar['value'] = 2.0
            
            
        # ---------------------- Productions and attractions -------------------------
 
        print('Calculating productions and attractions...'), log_file.write('Calculating productions and attractions...\n')
        if root != '':
            root.update_statusbar('Calculating productions and attractions...')
            
        # Surface of DCs per zone
        surfaceDC = np.array(zones['SurfaceDC'])
        
        # Surface of parcel nodes per zone
        surfaceParcelDepot = np.zeros(nZones, dtype=float)
        for i in range(len(parcelNodes)):
            zone    = parcelNodes.at[i, 'AREANR']
            surface = parcelNodes.at[i, 'Surface']
            surfaceParcelDepot[invZoneDict[int(zone)]] += surface
        
        # Jobs per sector
        jobs = {}
        for sector in sectorNames:
            jobs[sector] = np.zeros(nZones, dtype=float)
            jobs[sector][:nInternalZones] = segs.loc[[zoneDict[x] for x in range(nInternalZones)], sector]
            
            for i in range(nCOROP):
                jobs[sector][nInternalZones+i] = np.sum(segs.loc[MRDHwithinCOROP[i+1], sector])
        population = np.array(segs.loc[zones['AREANR'].values, '2: inwoners'])        
                
        # Determine produced trips per zone for service and construction
        prodService = np.zeros(nZones, dtype=int)
        prodBouw    = np.zeros(nZones, dtype=int)
        
        # For the zones in the study area (ZH)
        for i in range(nInternalZones):
            prodService[i] = regrCoeffs.at['Service','DC_OPP'    ] * surfaceDC[i]          + \
                             regrCoeffs.at['Service','PARCEL_OPP'] * surfaceParcelDepot[i] + \
                             regrCoeffs.at['Service','LANDBOUW'  ] * jobs['LANDBOUW' ][i]  + \
                             regrCoeffs.at['Service','INDUSTRIE' ] * jobs['INDUSTRIE'][i]  + \
                             regrCoeffs.at['Service','DETAIL'    ] * jobs['DETAIL'   ][i]  + \
                             regrCoeffs.at['Service','DIENSTEN'  ] * jobs['DIENSTEN' ][i]  + \
                             regrCoeffs.at['Service','OVERIG'    ] * jobs['OVERIG'   ][i]  + \
                             regrCoeffs.at['Service','INWONERS'  ] * population[i]

            prodBouw[i]    = regrCoeffs.at['Construction','DC_OPP'    ] * surfaceDC[i]          + \
                             regrCoeffs.at['Construction','PARCEL_OPP'] * surfaceParcelDepot[i] + \
                             regrCoeffs.at['Construction','LANDBOUW'  ] * jobs['LANDBOUW' ][i]  + \
                             regrCoeffs.at['Construction','INDUSTRIE' ] * jobs['INDUSTRIE'][i]  + \
                             regrCoeffs.at['Construction','DETAIL'    ] * jobs['DETAIL'   ][i]  + \
                             regrCoeffs.at['Construction','DIENSTEN'  ] * jobs['DIENSTEN' ][i]  + \
                             regrCoeffs.at['Construction','OVERIG'    ] * jobs['OVERIG'   ][i]  + \
                             regrCoeffs.at['Construction','INWONERS'  ] * population[i]

        if root != '':
            root.progressBar['value'] = 3.0
            
        # For the external zones                 
        for i in range(nCOROP):
            prodService[nInternalZones+i] = regrCoeffs.at['Service','PARCEL_OPP'] * surfaceParcelDepot[nInternalZones+i]        + \
                                            regrCoeffs.at['Service','LANDBOUW'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'LANDBOUW'   ]) + \
                                            regrCoeffs.at['Service','INDUSTRIE' ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'INDUSTRIE'  ]) + \
                                            regrCoeffs.at['Service','DETAIL'    ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'DETAIL'     ]) + \
                                            regrCoeffs.at['Service','DIENSTEN'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'DIENSTEN'   ]) + \
                                            regrCoeffs.at['Service','OVERIG'    ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'OVERIG'     ]) + \
                                            regrCoeffs.at['Service','INWONERS'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'2: inwoners'])

            prodBouw[nInternalZones+i] =    regrCoeffs.at['Construction','PARCEL_OPP'] * surfaceParcelDepot[nInternalZones+i]        + \
                                            regrCoeffs.at['Construction','LANDBOUW'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'LANDBOUW'   ]) + \
                                            regrCoeffs.at['Construction','INDUSTRIE' ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'INDUSTRIE'  ]) + \
                                            regrCoeffs.at['Construction','DETAIL'    ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'DETAIL'     ]) + \
                                            regrCoeffs.at['Construction','DIENSTEN'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'DIENSTEN'   ]) + \
                                            regrCoeffs.at['Construction','OVERIG'    ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'OVERIG'     ]) + \
                                            regrCoeffs.at['Construction','INWONERS'  ] * np.sum(segs.loc[MRDHwithinCOROP[i+1],'2: inwoners'])

        if root != '':
            root.progressBar['value'] = 4.0
            
        
        # ------------------------- Trip distribution -------------------------------

        print('Trip distribution...'), log_file.write('Trip distribution...\n')
        if root != '':
            root.update_statusbar('Trip distribution...')
                        
        print('\tConstructing initial matrix...'), log_file.write('\tConstructing initial matrix...\n')
        
        # Travel costs
        skimCost = costPerHour * (skimTravTime / 3600) + costPerKilometer * (skimDistance / 1000)
        skimCost = skimCost.reshape(nZones,nZones)
        
        # Intrazonal costs (half of costs to nearest zone in terms of travel costs)
        for i in range(nZones):
            skimCost[i,i]    = 0.5 * np.min(skimCost[i,skimCost[i,:]>0])
            
        # Travel resistance
        matrixService = 100 / (1 + (np.exp(alphaService) * skimCost ** betaService))
        matrixBouw    = 100 / (1 + (np.exp(alphaBouw)    * skimCost ** betaBouw))
        
        # Multiply by productions and then attractions to get start matrix (assumed: productions = attractions)
        matrixService *= np.tile(prodService, (len(prodService), 1))
        matrixBouw    *= np.tile(prodBouw,    (len(prodBouw   ), 1))
        
        matrixService *= np.tile(prodService, (len(prodService), 1)).transpose()
        matrixBouw    *= np.tile(prodBouw,    (len(prodBouw   ), 1)).transpose()
                
        if root != '':
            root.progressBar['value'] = 6.0
            
        print('\tDistributing service trips...'), log_file.write('\tDistributing service trips...\n')
        itern = 0
        conv  = tolerance + 100
        
        while (itern < maxIter) and (conv > tolerance):
            itern += 1
            print('\t\tIteration ' + str(itern)), log_file.write('\t\tIteration ' + str(itern) + '\n')
            maxColScaleFac = 0
            totalRows = np.sum(matrixService, axis = 0)
            
            for j in range(nZones):
                total = totalRows[j]
        
                if total > 0:
                    scaleFacCol = prodService[j]/total
        
                    if abs(scaleFacCol)> abs(maxColScaleFac):
                        maxColScaleFac = scaleFacCol
        
                    matrixService[:,j] *= scaleFacCol
        
            maxRowScaleFac = 0
            totalCols = np.sum(matrixService, axis=1)
            
            for i in range(nZones):        
                total = totalCols[i]
        
                if total > 0:
                    scaleFacRow = prodService[i]/total
        
                    if abs(scaleFacRow)> abs(maxRowScaleFac):
                        maxRowScaleFac = scaleFacRow
        
                    matrixService[i,:] *= scaleFacRow
        
            conv = max(abs(maxColScaleFac-1),abs(maxRowScaleFac-1))            
            print('\t\tConvergence ' + str(round(conv,4))), log_file.write('\t\tConvergence ' + str(round(conv,4)) + '\n')
            
            if root != '':
                root.progressBar['value'] = 6.0 + (46.0 - 6.0) * itern / maxIter
                
        if conv > tolerance:
            print('Warning! Convergence is lower than the tolerance criteroin, more iterations might be needed.')
            log_file.write('Warning! Convergence is lower than the tolerance criteroin, more iterations might be needed.' + '\n')


        print('\tDistributing construction trips...'), log_file.write('\tDistributing construction trips...\n')
        itern = 0
        conv  = tolerance + 100
        
        while (itern < maxIter) and (conv > tolerance):
            itern += 1
            print('\t\tIteration ' + str(itern)), log_file.write('\t\tIteration ' + str(itern) + '\n')
            maxColScaleFac = 0
            totalRows = np.sum(matrixBouw, axis = 0)
            
            for j in range(nZones):
                total = totalRows[j]
        
                if total > 0:
                    scaleFacCol = prodBouw[j]/total
        
                    if abs(scaleFacCol)> abs(maxColScaleFac):
                        maxColScaleFac = scaleFacCol
        
                    matrixBouw[:,j] *= scaleFacCol
        
            maxRowScaleFac = 0
            totalCols = np.sum(matrixBouw, axis=1)
            
            for i in range(nZones):        
                total = totalCols[i]
        
                if total > 0:
                    scaleFacRow = prodBouw[i]/total
        
                    if abs(scaleFacRow)> abs(maxRowScaleFac):
                        maxRowScaleFac = scaleFacRow
        
                    matrixBouw[i,:] *= scaleFacRow
        
            conv = max(abs(maxColScaleFac-1),abs(maxRowScaleFac-1))            
            print('\t\tConvergence ' + str(round(conv,4))), log_file.write('\t\tConvergence ' + str(round(conv,4)) + '\n')

            if root != '':
                root.progressBar['value'] = 46.0 + (86.0 - 46.0) * itern / maxIter
                
        if conv > tolerance:
            print('Warning! Convergence is lower than the tolerance criteroin, more iterations might be needed.')
            log_file.write('Warning! Convergence is lower than the tolerance criteroin, more iterations might be needed.' + '\n')
        
        # Trips van extern naar extern eruithalen (voor consistentie met vracht)
        for i in range(nExternalZones):
            matrixService[nInternalZones+i, nInternalZones:] = 0
            matrixBouw[   nInternalZones+i, nInternalZones:] = 0
            
        # Van jaar naar dag brengen
        matrixService = np.round(matrixService.flatten() / yearFactor, 3)
        matrixBouw    = np.round(matrixBouw.flatten()    / yearFactor, 3)
        
        
        # ------------------------ Writing OD trip matrices -------------------------
               
        print('\tWriting trip matrices...'), log_file.write('Writing trip matrices...\n')
        if root != '':
            root.update_statusbar('Writing trip matrices...')
                    
        write_mtx(datapathO + 'TripsVanService.mtx',      matrixService, nZones)
        
        if root != '':
            root.progressBar['value'] = 93.0
                
        write_mtx(datapathO + 'TripsVanConstruction.mtx', matrixBouw   , nZones)  
        
        if root != '':
            root.progressBar['value'] = 100.0
            
            
        # --------------------------- End of module ---------------------------------
            
        totaltime = round(time.time() - start_time, 2)
        print('Finished. Run time: ' + str(round(totaltime,2)) + ' seconds')
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Service Vans: Done")
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
                root.update_statusbar("Service Vans: Execution failed!")
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
    SEGS            = INPUTFOLDER + 'SEGS2016_verrijkt.csv'
    COMMODITYMATRIX = INPUTFOLDER + 'CommodityMatrixNUTS2_2016.csv'
    PARCELNODES     = INPUTFOLDER + 'parcelNodes_v2.shp'
    MRDH_TO_NUTS3   = PARAMFOLDER + 'MRDHtoNUTS32013.csv'
    NUTS3_TO_MRDH   = PARAMFOLDER + 'NUTS32013toMRDH.csv'
    SERVICE_DISTANCEDECAY = PARAMFOLDER + 'Params_DistanceDecay_SERVICE.csv'
    DISTRIBUTIECENTRA   = INPUTFOLDER + 'distributieCentra.csv'
    COST_VEHTYPE        = PARAMFOLDER + 'Cost_VehType_2016.csv'
    COST_SOURCING       = PARAMFOLDER + 'Cost_Sourcing_2016.csv'
    
    YEARFACTOR = 209
    
    NUTSLEVEL_INPUT = 3
    
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
    
    MODULES = ['FS', 'SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, LINKS, NODES, ZONES, SEGS, \
            COMMODITYMATRIX, PARCELNODES, MRDH_TO_NUTS3, NUTS3_TO_MRDH, \
            DISTRIBUTIECENTRA, COST_VEHTYPE, COST_SOURCING, \
            SERVICE_DISTANCEDECAY, \
            PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
            PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", "LINKS", "NODES", "ZONES", "SEGS", \
                  "COMMODITYMATRIX", "PARCELNODES", "MRDH_TO_NUTS3", "NUTS3_TO_MRDH", \
                  "DISTRIBUTIECENTRA", "COST_VEHTYPE", "COST_SOURCING",\
                  "SERVICE_DISTANCEDECAY", \
                  "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
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
    # main(varDict)

        
        

