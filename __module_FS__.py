# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:07:15 2020

@author: LEG
"""

import pandas as pd
import numpy as np
from scipy import stats
import time
import datetime
from shapely.geometry import Point, Polygon, MultiPolygon
from __functions__ import read_shape

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
        self.root.title("Progress Firm Synthesis")
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
        
        datapathI  = varDict['INPUTFOLDER']
        datapathO  = varDict['OUTPUTFOLDER']        
        pathZones  = varDict['ZONES']
        pathSegs   = varDict['SEGS']
        pathDC     = varDict['DISTRIBUTIECENTRA']
        
        doValidationChecks = True

        maxZoneNumberZH    = 7400
        minEmplLevelOutput = 3
        
        start_time = time.time()
        
        log_file = open(datapathO + "Logfile_FirmSynthesis.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        

        # --------------------------- Import data -----------------------------------
        
        print('Importing data...'), log_file.write('Importing data...\n')
        if root != '':
            root.update_statusbar('Importing data...')
            
        #Shapefile of study area
        zones, zonesGeometry = read_shape(pathZones, returnGeometry=True)
        zones          = zones.sort_values('AREANR')
        zonesGeometry  = [zonesGeometry[i] for i in zones.index]
        zones.index    = zones['AREANR']
        nInternalZones = len(zones)
        
        if root != '':
            root.progressBar['value'] = 1

        # Get zones as shapely MultiPolygon/Polygon objects
        shapelyZones = []
        for x in zonesGeometry:
            if x['type'] == 'MultiPolygon':
                temp = []
                for i in range(len(x['coordinates'])):
                    if len(x['coordinates'][i]) > 1:
                        temp.append(Polygon(x['coordinates'][i][0], [x['coordinates'][i][j] for j in range(1,len(x['coordinates'][i]))]))
                    else:
                        temp.append(Polygon(x['coordinates'][i][0]))
                shapelyZones.append(MultiPolygon(temp))
            elif x['type'] == 'Polygon':
                if len(x['coordinates']) > 1:
                    shapelyZones.append(Polygon(x['coordinates'][0], [x['coordinates'][i] for i in range(1,len(x['coordinates']))]))
                else:
                    shapelyZones.append(Polygon(x['coordinates'][0]))
            else:
                print('Warning! Object types other than Polygon or MultiPolygon found in zones shape!')
                log_file.write('Warning! Object types other than Polygon or MultiPolygon found in zones shape!\n')

        if root != '':
            root.progressBar['value'] = 2
                
        #Zonal / Socioeconomic data
        segs       = pd.read_csv(pathSegs)
        segs.index = segs['zone']
        emplCols = ['INDUSTRIE','DETAIL','LANDBOUW','DIENSTEN','OVERHEID','OVERIG']
        segsEmpl   = segs[emplCols].copy().round()
  
        #Adjust zonal employment data for distribution centres and transshipment terminals
        #in zones with DC, the jobs in DC are known and subtracted from industry jobs
        dcs     = pd.read_csv(pathDC)   #read DC file
        dcs_wp  = dcs.pivot_table(index='AREANR', values='WP', aggfunc='sum')  #aggregate jobs (WP) per zone
                  
        #add jobs in DC to segsEmpl df
        segsEmpl['WP'] = np.zeros(len(segsEmpl))
        for x in dcs_wp.index:
            segsEmpl['WP'][x] = dcs_wp['WP'][x]     

        #update number of industrie jobs and set to 0 in case it got smaller than 0
        segsEmpl['INDUSTRIE_voorCorr'] = segsEmpl['INDUSTRIE'].copy() #make a copy to keep track of changes
        segsEmpl['INDUSTRIE'] = segsEmpl['INDUSTRIE_voorCorr'] - segsEmpl['WP']
        segsEmpl['INDUSTRIE'][segsEmpl['INDUSTRIE'] <0] = 0

        jobs_init     = segsEmpl['INDUSTRIE_voorCorr'].sum()
        jobs_zonderDC = segsEmpl['INDUSTRIE'].sum()
        
        #in zones with transshipment terminals (lognode=1), remaining industry jobs are set to zero
        segsEmpl['LOGNODE'] = zones['LOGNODE'].copy()
        segsEmpl['INDUSTRIE'][segsEmpl['LOGNODE']==1]=0
       
        jobs_final   = segsEmpl['INDUSTRIE'].sum()
        
        print('INDUSTRIE jobs in SEGs', jobs_init, '\nINDUSTRIE jobs w/o DC', jobs_zonderDC, 
              '\nINDUSTRIE jobs in DC: ', dcs_wp['WP'].sum(), '\nINDUSTRIE jobs w/o DC and TT', jobs_final)

        print('Adjusting zonal data for jobs in DCs and transshipment terminals to avoid double counting... \n',
              '\t INDUSTRIE jobs before adjustment: ', int(jobs_init), '\n',
              '\t INDUSTRIE jobs in DCs (input):    ', int(dcs_wp['WP'].sum()), '\n',
              '\t INDUSTRIE jobs removed for DC:    ', int(jobs_init - jobs_zonderDC), '\n',
              '\t INDUSTRIE jobs removed for TT:    ', int(jobs_zonderDC - jobs_final), '\n',
              '\t INDUSTRIE jobs w/o DC and TT:     ', int(jobs_final), '\n')

        log_file.write('\Adjusting zonal data for jobs in DCs and transshipment terminals to avoid double counting... \n' +
                       '\tINDUSTRIE jobs before adjustment: ' + str(int(jobs_init)) + '\n' +
                       '\tINDUSTRIE jobs in DCs (input):    ' + str(int(dcs_wp['WP'].sum())) + '\n' + 
                       '\tINDUSTRIE jobs removed for DC:    ' + str(int(jobs_init - jobs_zonderDC))  +  '\n' + 
                       '\tINDUSTRIE jobs removed for TT:    ' + str(int(jobs_zonderDC - jobs_final)) +  '\n' + 
                       '\tINDUSTRIE jobs w/o DC and TT:     '   + str(int(jobs_final)) + '\n')
        
        #Remove columns that are not needed for synthesis
        segsEmpl = segsEmpl.drop(['INDUSTRIE_voorCorr', 'WP', 'LOGNODE'], 1)

        if root != '':
            root.progressBar['value'] = 3
            
        # Koppeltabel sectoren inlezen
        koppel_sectors = pd.read_csv(datapathI + "Koppeltabel_sectoren_SBI_SEGs.csv")
        sectors_dict = dict(zip(koppel_sectors['Sector_SBI'], koppel_sectors['Sector_SEGs']))
                
        #Read firm input data
        firmSizePerSector = pd.read_csv(datapathI + "FirmSizeDistributionPerSector_6cat.csv")   #Data from CBS:        
        firmSizePerSector.index = firmSizePerSector['Sector']
        firmSizePerSector.drop('Sector', inplace=True, axis=1)
        
        firmSize_labels = firmSizePerSector.columns[1:].to_list()
        firmSize_dict  = dict(zip(firmSizePerSector.columns[1:], [(1,5),(5,10),(10,20),(20,50),(50,100),(100,150,1000)]))

        if root != '':
            root.progressBar['value'] = 4      
        
        
        # -------- Create random distribution from firm size x sector table -------------
        
        # First create random distribution for all sectors combined;
        # starting point: firmSizePerSector - joint distribution table.
        # Drop column with SEGs-sectors which is not needed here
        firmSizeALL  = firmSizePerSector.drop(['Sector_SEGs'], axis=1)
        # Create probability density function
        firmsXsectorALL_pdf = (firmSizeALL/1.0)/sum(sum(firmSizeALL.values))
        # Convert 2d shape to 1d
        firmsXsectorALL_pdf = firmsXsectorALL_pdf.values.ravel()

        #create random distributions for all SEG-sectors
        SEGs_sectors = list(firmSizePerSector['Sector_SEGs'].unique())
            
        firmSize={}
        firmsXsector={}
        firmsXsector_pdf={}
        indices={}
        distr={}
        for sect in SEGs_sectors:
            firmSize[sect]     = firmSizePerSector[firmSizePerSector['Sector_SEGs']==sect].drop(['Sector_SEGs'], axis=1)
            firmsXsector[sect] = firmSize[sect].apply(lambda x: x.index+'::'+x.name).values.ravel() 
            indices[sect]      = range(0, len(firmsXsector[sect]))  
            firmsXsector_pdf[sect] = (firmSize[sect]/1.0)/sum(sum(firmSize[sect].values))
            firmsXsector_pdf[sect] = firmsXsector_pdf[sect].values.ravel()
            distr[sect]        = stats.rv_discrete(name=f'distr_{sect}', values=(indices[sect], firmsXsector_pdf[sect]))
        
        if root != '':
            root.progressBar['value'] = 5
            
        # --------------------------- Synthesize firms ------------------------------
        # Turn number of employees per zone into firms (drawn randomly from sectorXsize table)
               
        print('Synthesizing firms...'), log_file.write('Synthesizing firms...\n')
        if root != '':
            root.update_statusbar('Synthesizing firms...')
            
        firmZones   = []
        firmSectors = []
        firmSizes   = [] 
        firmEmpl    = []
        zoneEmpl    = [] #number of employees in current zone and sector
        
        # Select TAZ
        for zone in segsEmpl[:maxZoneNumberZH].index:
        # for zone in segsEmpl.index:
        # for zone in segsEmpl.index:
            assigned_jobs=0
        
            #Select industry sector I
            for sect in segsEmpl.columns:
                jobs_to_assign = segsEmpl.loc[zone][sect].copy()
        
                while jobs_to_assign > 0:
                    #draw a firm for current sector
                    firm = np.array([x.split('::') for x in firmsXsector[sect][distr[sect].rvs(size=1)]])
        
        #Determine size of the firm
                    #if we draw a firm from the largest category, draw from triangular distribution            
                    if firm[0][1] == firmSize_labels[-1]: 
                        left  = 100
                        mode  = max(100, 0.5*jobs_to_assign) 
                        right = max(150, 1.5*jobs_to_assign)
                        size = int(np.random.triangular(left, mode, right, size=1))
        
                    #if the firm is from a closed size category, draw size from uniform distribution
                    else:
                        low  = firmSize_dict[firm[0][1]][0]
                        high = firmSize_dict[firm[0][1]][1]
                        size = np.random.randint(low=low, high=high, size=1, dtype=int)
        
        #check if size fits in zone                         
                    # if firm is not way too large, accept it and add it to list of firms
                    if size <= 1.5 * jobs_to_assign:
                        # print(firm)
                        assigned_jobs += size
                        
                        firmZones.append(zone)
                        firmSectors.append(sectors_dict[str(firm[0][0])])
                        firmSizes.append(str(firm[0][1]))
                        firmEmpl.append(size)
                        # zoneEmpl.append(jobs_to_assign)
                        zoneEmpl.append(assigned_jobs)      
                        
                        jobs_to_assign -= size
                        
                    else: 
                        #determine firm size based on remaining jobs to assign
                        # size = np.random.randint(8,12)/10*jobs_to_assign
                        size = jobs_to_assign
                        assigned_jobs += int(size)

                        if size >= 100:
                            sizeClass = 'Groot'
                        elif size >= 50: 
                            sizeClass = 'Middelgroot'
                        elif size >= 20: 
                            sizeClass = 'Middel'
                        elif size >= 10: 
                            sizeClass = 'Middelklein'
                        elif size >= 5: 
                            sizeClass = 'Klein'
                        else: 
                            sizeClass = 'Micro'
                        
                        firmZones.append(zone)
                        firmSectors.append(sectors_dict[str(firm[0][0])])
                        firmSizes.append(sizeClass)
                        firmEmpl.append(size)
                        zoneEmpl.append(assigned_jobs)      
                        
                        jobs_to_assign -= size

            if (zone-1)%int((maxZoneNumberZH-1)/20) == 0:
                print('\t' + str(int(round(((zone-1) / (maxZoneNumberZH-1))*100, 0))) + '%')
                if root != '':
                    root.progressBar['value'] = 5 + (60-5) * (zone-1) / (maxZoneNumberZH-1)
            
        
        # Put firms in Numpy Array
        sizeX=len(firmZones)
        sizeY=5
        allFirms = np.empty((sizeX,sizeY), dtype=object)
        allFirms[:,0] = np.arange(sizeX)
        allFirms[:,1] = firmZones
        allFirms[:,2] = firmSectors
        allFirms[:,3] = firmSizes
        allFirms[:,4] = firmEmpl
        allFirms[:,4] = allFirms[:,4].astype('int64')

        #put results in a dataframe
        firms_df = pd.DataFrame(allFirms, columns=('FIRM_ID','MRDH_ZONE','SECTOR','SIZE','EMPL')) 

        if root != '':
            root.progressBar['value'] = 65   
            
        # --------------------------- Optional validation checks ------------------------------
        
        if doValidationChecks:
            unique_emplSize, counts_emplSize = np.unique(allFirms[:,1],return_counts=True)
            emplSize_counts = np.asarray((unique_emplSize, counts_emplSize, np.round(counts_emplSize/np.sum(counts_emplSize), 5))).T
            print(np.sum(counts_emplSize))
            print(emplSize_counts)
            
            unique_sector, counts_sector = np.unique(allFirms[:,2],return_counts=True)
            sector_counts = np.asarray((unique_sector, counts_sector, np.round(counts_sector/np.sum(counts_sector), 5))).T
            print(sector_counts)            
            
            #this one is used
            zones_jobs = firms_df.pivot_table(index='MRDH_ZONE', columns='SECTOR', values='EMPL', aggfunc='sum')
            zones_jobs['TOTAAL'] = zones_jobs.sum(axis=1, skipna=True)
            zones_jobs = zones_jobs.fillna(0)
            
            #merge with input segs to better enable comparison
            emplIn = segsEmpl.copy()
            emplIn ['TOTAAL'] = emplIn.sum(axis=1, skipna=True)
            emplIn = emplIn[emplIn.index<=maxZoneNumberZH].copy()
            
            emplAbsDiff = zones_jobs-emplIn
            emplAbsDiff.columns = ['DETAIL_absDiff', 'DIENSTEN_absDiff', 'INDUSTRIE_absDiff', 'LANDBOUW_absDiff', 'OVERHEID_absDiff', 'OVERIG_absDiff', 'TOTAAL_absDiff']
            
            emplRelDiff = (zones_jobs-emplIn)/emplIn
            emplRelDiff.columns = ['DETAIL_relDiff', 'DIENSTEN_relDiff', 'INDUSTRIE_relDiff', 'LANDBOUW_relDiff', 'OVERHEID_relDiff', 'OVERIG_relDiff', 'TOTAAL_relDiff']
                     
            df_out = emplIn.merge(zones_jobs.iloc[:, -7:], how='left', left_on=emplIn.index, right_index=True, suffixes=('_in', '_out'))
            df_out = df_out.merge(emplAbsDiff, how='left', left_on=emplIn.index, right_index=True, suffixes=('', '_absDiff')) 
            df_out = df_out.merge(emplRelDiff, how='left', left_on=emplIn.index, right_index=True, suffixes=('', '_relDiff')) 
            
            df_out = df_out.fillna(0)
            
            synthFirmsLabel= 'run_14'            
            df_out.to_csv(f'{datapathO}SynthJobsPerZone_{synthFirmsLabel}.csv')    
                    
            #make crosstab sector X firm size for comparison with input
            sectorsXsize_out = firms_df.pivot_table(index='SECTOR', columns='SIZE', values='MRDH_ZONE', aggfunc='count')
            sectorsXsize_out = sectorsXsize_out.fillna(0)
            sectorsXsize_out.to_csv(f'{datapathO}SynthFirms_{synthFirmsLabel}_sectorsXsize.csv')    
        

        # ------------------------- Draw coordinates  ------------------------------

        # Remove small firms
        firms_df = firms_df[firms_df['EMPL'] > minEmplLevelOutput]
        
        # New firm IDs after filtering
        firms_df['FIRM_ID'] = np.arange(len(firms_df))
        firms_df.index = np.arange(len(firms_df))
        nFirms = len(firms_df)
        
        print('Generating coordinates for ' + str(nFirms) + ' firms...')
        log_file.write('Generating coordinates for ' + str(nFirms) + ' firms...\n')
        if root != '':
            root.update_statusbar('Generating coordinates...')

        # Dictionary with zone number (1-6625) to corresponding zone number (1-7400)
        zoneDict    = dict(np.transpose(np.vstack((np.arange(nInternalZones), zones['AREANR']))))
        zoneDict    = {int(a):int(b) for a,b in zoneDict.items()}
        invZoneDict = dict((v, k) for k, v in zoneDict.items())   
        
        firmZones = np.array(firms_df['MRDH_ZONE'], dtype=int)
        
        # Initialize two dictionaries in which we'll store the drawn x- and y-coordinate for each firm
        X = {}
        Y = {}
        
        nTriesAllowed = 500
        nTimesNumberOfTriesReached = 0
        centroidZones = []
        
        for i in range(nFirms):
            # Get the zone polygon and its boundaries
            polygon = shapelyZones[invZoneDict[firmZones[i]]]
            minX, minY, maxX, maxY = polygon.bounds 
            
            pointInPolygon = False
            numberOfTries = 0
            
            while pointInPolygon == False and numberOfTries <= nTriesAllowed:
                # Generate a random point within the zone boundaries
                x = minX + (maxX - minX) * np.random.rand()
                y = minY + (maxY - minY) * np.random.rand()
                point = Point(x,y)
                
                # Check if the random point is actually contained by the zone polygon
                pointInPolygon = polygon.contains(point)
                numberOfTries += 1
            
            # If we haven't generated a point contained by the zone polygon yet after {nTriesAllowed} times trying,
            # then we just take the centroid of the zone polygon
            if pointInPolygon == False:
                x, y = polygon.centroid.coords[0]
                nTimesNumberOfTriesReached += 1
                centroidZones.append(firmZones[i])
                
            X[i], Y[i] = x, y
                
            if i%int(nFirms/20) == 0:
                print('\t' + str(int(round((i / nFirms)*100, 0))) + '%')
                if root != '':
                    root.progressBar['value'] = 65 + (95-65) * i / nFirms
                    
        firms_df['X'] = X.values()
        firms_df['Y'] = Y.values()
        
        print('\tHad to use the centroid coordinates for ' + str(nTimesNumberOfTriesReached) + ' of ' + str(nFirms) + ' firms')
        log_file.write('\tHad to use the centroid coordinates for ' + str(nTimesNumberOfTriesReached) + ' of ' + str(nFirms) + ' firms\n')
        
        if nTimesNumberOfTriesReached > 0:
            print('\t(This is the case for the following zone(s): ' + str(np.unique(centroidZones)) + ')')
            log_file.write('\t(This is the case for the following zone(s): ' + str(np.unique(centroidZones)) + ')\n')
        

        # ------------------------ Export firms to CSV  ----------------------------
        
        print('Exporting firms to: ' + datapathO + 'Firms.csv'), log_file.write('Exporting firms to: ' + datapathO + 'Firms.csv\n')
        if root != '':
            root.update_statusbar('Exporting firms...')
            
        firms_df.to_csv(datapathO + 'Firms.csv', index=False)   
        
        if doValidationChecks:
            firms_df.to_csv(datapathO + f'Firms_{synthFirmsLabel}.csv', index=False)   
            
        
            
        # --------------------------- End of module ---------------------------------
            
        totaltime = round(time.time() - start_time, 2)
        print('Finished. Run time: ' + str(round(totaltime,2)) + ' seconds')
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        if root != '':
            root.update_statusbar("Firm Synthesis: Done")
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
                root.update_statusbar("Firm Synthesis: Execution failed!")
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
    
    SKIMTIME            = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimTijd_REF.mtx'
    SKIMDISTANCE        = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimAfstand_REF.mtx'
    LINKS		        = INPUTFOLDER + 'links_v5.shp'
    NODES               = INPUTFOLDER + 'nodes_v5.shp'
    ZONES               = INPUTFOLDER + 'Zones_v4.shp'
    SEGS                = INPUTFOLDER + 'SEGS2016_verrijkt.csv'
    COMMODITYMATRIX     = INPUTFOLDER + 'CommodityMatrixNUTS3_2016.csv'
    PARCELNODES         = INPUTFOLDER + 'parcelNodes_v2.shp'
    DISTRIBUTIECENTRA   = INPUTFOLDER + 'distributieCentra.csv'
    COST_VEHTYPE        = PARAMFOLDER + 'Cost_VehType_2016.csv'
    COST_SOURCING       = PARAMFOLDER + 'Cost_Sourcing_2016.csv'
    MRDH_TO_NUTS3       = PARAMFOLDER + 'MRDHtoNUTS32013.csv'
    NUTS3_TO_MRDH       = PARAMFOLDER + 'NUTS32013toMRDH.csv'
    
    YEARFACTOR = 255
    
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
    N_CPU = ""
    
    IMPEDANCE_SPEED = 'V_FR_OS'
    
    LABEL = 'REF'
    
    MODULES = ['FS', 'SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, \
            LINKS, NODES, ZONES, SEGS, \
            DISTRIBUTIECENTRA, COST_VEHTYPE,COST_SOURCING,
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
                  "DISTRIBUTIECENTRA", "COST_VEHTYPE","COST_SOURCING", \
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

        
        

