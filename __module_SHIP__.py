# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:05:56 2019

@author: STH
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
        self.root.title("Progress Shipment Synthesizer")
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
        
        datapathI = varDict['INPUTFOLDER']
        datapathO = varDict['OUTPUTFOLDER']
        datapathP = varDict['PARAMFOLDER']
        yearFac   = varDict['YEARFACTOR']
        zonesPath        = varDict['ZONES']
        firmsPath        = varDict['FIRMS']
        skimTravTimePath = varDict['SKIMTIME']
        skimDistancePath = varDict['SKIMDISTANCE']
        shipmentsRef     = varDict['SHIPMENTS_REF']
        parcelDepotsPath = varDict['PARCELNODES']
        parcelsGrowthFreight = varDict['PARCELS_GROWTHFREIGHT']
        pathNUTS3toMRDH  = varDict['NUTS3_TO_MRDH']
        label            = varDict['LABEL']

        start_time = time.time()
        
        log_file = open(datapathO + "Logfile_ShipmentSynthesizer.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
                
        
        
        # ----------------------- Import data -------------------------------------
        
        root.update_statusbar("Shipment Synthesizer: Importing and preparing data")
        log_file.write("Importing and preparing data...\n")
        root.progressBar['value'] = 0
        
        nNSTR    = 10
        nLogSeg  = 8
        
        # Distance decay parameters
        alpha = -6.172
        beta  =  2.180
        
        # Which NSTR belongs to which Logistic Segments
        nstrToLogSeg  = np.array(pd.read_csv(datapathI + 'nstrToLogisticSegment.csv', header=None), dtype=float)
        for nstr in range(nNSTR):
            nstrToLogSeg[nstr,:] = nstrToLogSeg[nstr,:] / np.sum(nstrToLogSeg[nstr,:])
        
        # Which Logistic Segment belongs to which NSTRs      
        logSegToNstr  = np.array(pd.read_csv(datapathI + 'nstrToLogisticSegment.csv', header=None), dtype=float)
        for ls in range(nLogSeg):
            logSegToNstr[:,ls] = logSegToNstr[:,ls] / np.sum(logSegToNstr[:,ls])
        
        root.progressBar['value'] = 0.2
        
        # Import make/use distribution tables (by NSTR and industry sector)
        makeDistribution = np.array(pd.read_csv(f"{datapathI}MakeDistribution.csv"))
        useDistribution  = np.array(pd.read_csv(f"{datapathI}UseDistribution.csv"))
        
        root.progressBar['value'] = 0.4
        
        # Import external zones demand and coordinates
        superCommodityMatrixNSTR = pd.read_csv(datapathO + 'CommodityMatrixNUTS3.csv', sep=',')
        superCommodityMatrixNSTR['WeightDay'] = superCommodityMatrixNSTR['TonnesYear'] / yearFac
        superCommodityMatrixNSTR = np.array(superCommodityMatrixNSTR[['ORIG','DEST','NSTR','WeightDay']], dtype=object)
        superCoordinates = pd.read_csv(f'{datapathI}SupCoordinatesID.csv')
        superZoneX       = np.array(superCoordinates['Xcoor'])
        superZoneY       = np.array(superCoordinates['Ycoor'])
        nSuperZones      = len(superCoordinates)
        
        NUTS3toAREANR = pd.read_csv(pathNUTS3toMRDH, sep=',')
        NUTS3toAREANR = dict((NUTS3toAREANR.at[i,'NUTS_ID'],NUTS3toAREANR.at[i,'AREANR']) for i in NUTS3toAREANR.index)
        AREANRtoNUTS3 = {}
        for nuts3, areanr in NUTS3toAREANR.items():
            if areanr in AREANRtoNUTS3.keys():
                AREANRtoNUTS3[areanr].append(nuts3)
            else:
                AREANRtoNUTS3[areanr] = [nuts3]
                
        # Convert demand from NSTR to logistic segments
        nRows = (nSuperZones+1) * (nSuperZones+1) * nLogSeg
        superCommodityMatrix = np.zeros((nRows,4))
        superCommodityMatrix[:,0] = np.floor(np.arange(nRows)/(nSuperZones+1)/nLogSeg)
        superCommodityMatrix[:,1]= np.floor(np.arange(nRows)/nLogSeg) - superCommodityMatrix[:,0]*(nSuperZones+1)
        for logSeg in range(nLogSeg):
            superCommodityMatrix[np.arange(logSeg,nRows,nLogSeg), 2] = logSeg
        for i in range(nSuperZones+1):
            if 99999900+i in AREANRtoNUTS3.keys():
                for j in range(nSuperZones+1):
                    if 99999900+j in AREANRtoNUTS3.keys():
                        for nstr in range(nNSTR):
                            origNUTS3 = AREANRtoNUTS3[99999900+i]
                            destNUTS3 = AREANRtoNUTS3[99999900+j]
                            whereCurrent = [x for x in range(len(superCommodityMatrixNSTR)) if superCommodityMatrixNSTR[x,0] in origNUTS3 and \
                                                                                               superCommodityMatrixNSTR[x,1] in destNUTS3 and \
                                                                                               superCommodityMatrixNSTR[x,2] == nstr]
                            if len(whereCurrent) > 0:
                                weightDay = np.sum(superCommodityMatrixNSTR[whereCurrent, 3])
                                for logSeg in range(nLogSeg):
                                    currentRow = i*(nSuperZones+1)*nLogSeg + j*nLogSeg + logSeg
                                    superCommodityMatrix[currentRow, 3] += nstrToLogSeg[nstr,logSeg] * float(weightDay)
                                    
                                    # Apply growth to parcel market
                                    if logSeg == 6:
                                        superCommodityMatrix[currentRow, 3] *= parcelsGrowthFreight
        
        superCommodityMatrix = pd.DataFrame(superCommodityMatrix, columns=['From','To','LogSeg','WeightDay'])
        superCommodityMatrix.loc[(superCommodityMatrix['From']!=0) & (superCommodityMatrix['To']!=0), 'WeightDay'] = 0
        
        root.progressBar['value'] = 1
        
        # Import internal zones data
        zonesShape = read_shape(zonesPath)
        zonesShape.sort_values('AREANR')
        zonesShape.index = zonesShape['AREANR']
        zoneID      = np.array(zonesShape['AREANR'])
        zoneX       = np.array(zonesShape['X'])
        zoneY       = np.array(zonesShape['Y'])
        zoneLognode = np.array(zonesShape['LOGNODE'])
        zoneSurface = np.array(zonesShape['area'])
        nInternalZones = len(zonesShape)
        zoneDict = dict(np.transpose(np.vstack((np.arange(nInternalZones),zoneID))))
        for i in range(nSuperZones):
            zoneDict[nInternalZones+i] = superCoordinates['AREANR'][i]
        invZoneDict = dict((v, k) for k, v in zoneDict.items())
        zoneID      = np.arange(nInternalZones)
        
        root.progressBar['value'] = 1.2
        
        # Import firm data
        firms    = pd.read_csv(firmsPath)
        firmID   = np.array(firms['FIRM_ID'])
        firmZone = np.array([invZoneDict[firms['MRDH_ZONE'][i]] for i in firms.index])
        firmSize = np.array(firms['EMPL'])
        firmX    = np.array(firms['X'])
        firmY    = np.array(firms['Y'])
        
        sectorDict = {'LANDBOUW':1, 'INDUSTRIE':2, 'DETAIL':3, \
                      'DIENSTEN':4, 'OVERHEID':5,  'OVERIG':6}
        firmSector  = np.array([sectorDict[firms['SECTOR'][i]] for i in firms.index])
        
        root.progressBar['value'] = 1.5
        
        # Import logistic nodes data
        logNodes  = pd.read_csv(datapathI + 'distributieCentra.csv')
        logNodes  = logNodes[~pd.isna(logNodes['AREANR'])]
        logNodes['AREANR'] = [invZoneDict[x] for x in logNodes['AREANR']]
        logNodesX = np.array(logNodes['Xcoor'])
        logNodesY = np.array(logNodes['Ycoor'])
             
        # List of zone numbers Transshipment Terminals and Logistic Nodes
        ttZones = np.where(zoneLognode==1)[0]
        dcZones = np.array(logNodes['AREANR'])
        
        # Flowtype distribution (10 NSTRs and 12 flowtypes)
        ftShares = np.array(pd.read_csv(datapathP + 'LogFlowtype_Shares.csv', index_col=0)) 
            
        root.progressBar['value'] = 1.5
        
        # Skim with travel times and distances
        skimTravTime = read_mtx(skimTravTimePath)
        skimDistance = read_mtx(skimDistancePath)
        nZones = int(len(skimTravTime)**0.5)
        
        skimTravTime[skimTravTime<0] = 0
        skimDistance[skimDistance<0] = 0
        
        # For zero times and distances assume half the value to the nearest (non-zero) zone 
        # (otherwise we get problem in the distance decay function)
        for orig in range(nZones):
            whereZero    = np.where(skimTravTime[orig * nZones + np.arange(nZones)] == 0)[0]
            whereNonZero = np.where(skimTravTime[orig * nZones + np.arange(nZones)] != 0)[0]
            if len(whereZero) > 0:
                skimTravTime[orig * nZones + whereZero] = 0.5 * np.min(skimTravTime[orig * nZones + whereNonZero])

            whereZero    = np.where(skimDistance[orig * nZones + np.arange(nZones)] == 0)[0]
            whereNonZero = np.where(skimDistance[orig * nZones + np.arange(nZones)] != 0)[0]
            if len(whereZero) > 0:
                skimDistance[orig * nZones + whereZero] = 0.5 * np.min(skimDistance[orig * nZones + whereNonZero])
        
        root.progressBar['value'] = 2.5
        
        # Cost parameters by vehicle type with size (small/medium/large)
        costParams  = pd.read_csv(datapathP + "CostParameters.csv", index_col=0)
        costPerKm   = np.array(costParams['CostPerKm'])
        costPerHour = np.array(costParams['CostPerH'])
        
        # Cost parameters generic for sourcing (vehicle type is now known yet then)
        costParamsSourcing   = pd.read_csv(datapathP + "CostParameters_Sourcing.txt",sep='\t')
        costPerKmSourcing    = costParamsSourcing['CostPerKm'][0]
        costPerHourSourcing  = costParamsSourcing['CostPerHour'][0]     
        
        # Estimated parameters MNL for combined shipment size and vehicle type
        logitParams = pd.read_csv(datapathP + "Params_ShipSize_VehType.csv", index_col=0)
        
        # Which NSTR belongs to which goods type used in estimation of MNL
        dictNSTR = {0:'climate controlled', 1:'climate controlled', \
                    2:'heavy bulk', 3:'heavy bulk', 4:'heavy bulk', 5:'heavy bulk', 6:'heavy bulk', \
                    7:'chemicals', 8:'chemicals', 9:'manufactured'}
        
        root.progressBar['value'] = 2.6
        
        # Consolidation potential per logistic segment (for UCC scenario)
        probConsolidation = np.array(pd.read_csv(datapathI + 'ConsolidationPotential.csv', index_col='Segment'))
        
        # Vehicle/combustion shares (for UCC scenario)
        sharesUCC  = pd.read_csv(datapathI + 'ZEZscenario.csv', index_col='Segment')
        #combTypes  = ['Fuel', 'Electric', 'Hydrogen', 'Hybrid (electric)', 'Biofuel']
        vtNamesUCC = ['LEVV','Moped','Van','Truck','TractorTrailer','WasteCollection','SpecialConstruction']
        
        # Assume no consolidation potential and vehicle type switch for dangerous goods
        sharesUCC = np.array(sharesUCC)[:-1,:-1]
        
        # Only vehicle shares (summed up combustion types)
        sharesVehUCC = np.zeros((nLogSeg-1,len(vtNamesUCC)))
        for ls in range(nLogSeg-1):
            sharesVehUCC[ls,0] = np.sum(sharesUCC[ls,0:5])
            sharesVehUCC[ls,1] = np.sum(sharesUCC[ls,5:10])
            sharesVehUCC[ls,2] = np.sum(sharesUCC[ls,10:15])
            sharesVehUCC[ls,3] = np.sum(sharesUCC[ls,15:20])
            sharesVehUCC[ls,4] = np.sum(sharesUCC[ls,20:25])
            sharesVehUCC[ls,5] = np.sum(sharesUCC[ls,25:30])            
            sharesVehUCC[ls,6] = np.sum(sharesUCC[ls,30:35])
            sharesVehUCC[ls,:] = np.cumsum(sharesVehUCC[ls,:]) / np.sum(sharesVehUCC[ls,:])
        
        # Couple these vehicle types to HARMONY vehicle types
        vehUccToVeh = {0:8, 1:9, 2:7, 3:1, 4:5, 5:6, 6:6}
    
        # Depots for parcel deliveries
        parcelNodes       = read_shape(parcelDepotsPath)
        parcelNodes.index = parcelNodes['id'].astype(int)
        parcelNodes       = parcelNodes.sort_index()
        parcelNodes       = parcelNodes[parcelNodes['AREANR']<99999900] # Remove parcel nodes in external zones
        parcelNodes['AREANR'] = [invZoneDict[x] for x in parcelNodes['AREANR']]
        nParcelNodes = len(parcelNodes)
        parcelNodes.index = np.arange(nParcelNodes)
           
        # Market shares of the different parcel couriers
        cepShares = pd.read_csv(datapathI + 'CEPshares.csv', index_col=0)
        cepList   = list(cepShares.index)
        nDepots = [np.sum(parcelNodes['CEP']==str(cep)) for cep in cepList]
        cepShares['ShareInternal'] = cepShares['ShareNL']
        cepShares.iloc[np.array(nDepots)==1,-1] = 0
        cepSharesTotal    = np.cumsum(cepShares['ShareTotal'   ]) / np.sum(cepShares['ShareTotal'   ])
        cepSharesInternal = np.cumsum(cepShares['ShareInternal']) / np.sum(cepShares['ShareInternal'])            
        cepDepotZones  = [np.array(parcelNodes.loc[parcelNodes['CEP']==str(cep),'AREANR'], dtype=int) for cep in cepList]
        cepDepotShares = [np.array(parcelNodes.loc[parcelNodes['CEP']==str(cep),'Surface']) for cep in cepList]
        cepDepotShares = [np.cumsum(cepDepotShares[i]) / np.sum(cepDepotShares[i]) for i in range(len(cepList))]
        cepDepotX      = [np.array(zonesShape['X'][[zoneDict[x] for x in cepDepotZones[i]]]) for i in range(len(cepList))]
        cepDepotY      = [np.array(zonesShape['Y'][[zoneDict[x] for x in cepDepotZones[i]]]) for i in range(len(cepList))]            
        
        root.progressBar['value'] = 2.8
        
        
        
        # -------------------- Set parameters -------------------------------------
        
        nFirms = len(firms)
        nFlowTypesInternal = 9
        nFlowTypesExternal = 3
        nShipSizes = 6
        nVehTypes  = 7
        truckCapacities = np.array(pd.read_csv(f"{datapathP}CarryingCapacity.csv", index_col=0))[:,0] / 1000
        absoluteShipmentSizes = [1.5, 4.5, 8.0, 15.0, 25.0, 35.0]
        ExtArea = True        
        
        
        
        # ----------- Cumulative probability functions for allocation -------------
        
        # Cumulative probability function of firms being receiver or sender
        probReceive     = np.zeros((nFirms,nNSTR))
        probSend        = np.zeros((nFirms,nNSTR))
        cumProbReceive  = np.zeros((nFirms,nNSTR))
        cumProbSend     = np.zeros((nFirms,nNSTR))
                
        # Per goods type, determine probability based on firm size and make/use share
        for nstr in range(nNSTR):
            probReceive[:,nstr]     = firmSize * [useDistribution[nstr,sector-1]  for sector in firmSector]            
            probSend[:,nstr]        = firmSize * [makeDistribution[nstr,sector-1] for sector in firmSector]
            cumProbReceive[:,nstr]  = np.cumsum(probReceive[:,nstr])
            cumProbReceive[:,nstr] /= cumProbReceive[-1,nstr]
            cumProbSend[:,nstr]     = np.cumsum(probSend[:,nstr])
            cumProbSend[:,nstr]    /= cumProbSend[-1,nstr]
        
        # Cumulative probability function of a shipment being allocated to a particular DC/TT (based on surface)        
        probDC     = np.array(logNodes['oppervlak'])
        cumProbDC  = np.cumsum(probDC)
        cumProbDC  = cumProbDC / cumProbDC[-1]

        probTT     = zoneSurface[ttZones]        
        cumProbTT  = np.cumsum(probTT)
        cumProbTT  = cumProbTT / cumProbTT[-1]

        root.progressBar['value'] = 2.9
        
        
        
        # ----------------------- Demand by flowtype ------------------------------
        
        # Split demand by internal/export/import and goods type
        demandInternal  = np.array(superCommodityMatrix['WeightDay'][:nLogSeg],dtype=float)
        demandExport    = np.zeros((nSuperZones,nLogSeg),dtype=float)
        demandImport    = np.zeros((nSuperZones,nLogSeg),dtype=float)
        for superZone in range(nSuperZones):
            for logSeg in range(nLogSeg):
                exportIndex = (superZone+1) * (nSuperZones+1) * nLogSeg + logSeg
                demandExport[superZone][logSeg] = superCommodityMatrix['WeightDay'][exportIndex]
            
                importIndex = (superZone+1) * nLogSeg + logSeg
                demandImport[superZone][logSeg] = superCommodityMatrix['WeightDay'][importIndex]  
                
        # Then split demand by flowtype            
        demandInternalByFT  = [None] * nFlowTypesInternal
        demandExportByFT    = [None] * nFlowTypesExternal
        demandImportByFT    = [None] * nFlowTypesExternal
        for ft in range(nFlowTypesInternal):
            demandInternalByFT[ft]  = demandInternal.copy() * ftShares[ft,:]
            
        for ft in range(nFlowTypesExternal):
            demandExportByFT[ft]    = demandExport.copy() * ftShares[nFlowTypesInternal+ft,:]
            demandImportByFT[ft]    = demandImport.copy() * ftShares[nFlowTypesInternal+ft,:]

        root.progressBar['value'] = 3



        # ------------------- Shipment synthesizer procedure ----------------------
        
        if shipmentsRef == "":                
                
            # Initialize a counter for the procedure
            count = 0
            
            # Initialize shipment attributes as dictionaries
            fromFirm        = {}
            toFirm          = {}
            flowType        = {}
            goodsType       = {}
            logisticSegment = {}
            shipmentSize    = {}
            shipmentSizeCat = {}
            vehicleType     = {}
            destZone        = {}
            origZone        = {}
            origX           = {}
            origY           = {}
            destX           = {}
            destY           = {}

            root.update_statusbar("Shipment Synthesizer: Synthesizing shipments within study area")
            log_file.write("Synthesizing shipments within study area...\n")            
            percStart = 3
            percEnd   = 40
            root.progressBar['value'] = percStart

            # For progress bar
            totalWeightInternal = np.sum([np.sum(demandInternalByFT[ft]) for ft in range(nFlowTypesInternal)])
            allocatedWeightInternal  = 0
                
            for logSeg in range(nLogSeg):
                
                for nstr in range(nNSTR):
                    
                    if logSegToNstr[nstr,logSeg] > 0:
                        root.update_statusbar("Shipment Synthesizer: Synthesizing shipments within study area (LS " + str(logSeg) + " and NSTR " + str(nstr) + ")")
                        log_file.write(f"\tFor logistic segment {logSeg} (NSTR{nstr})\n")
                            
                        # Selecting the logit parameters for this NSTR group
                        logitParamsNSTR             = logitParams[dictNSTR[nstr]]
                        B_TransportCosts            = logitParamsNSTR['B_TransportCosts']
                        B_InventoryCosts            = logitParamsNSTR['B_InventoryCosts']
                        B_FromDC                    = logitParamsNSTR['B_FromDC']
                        B_ToDC                      = logitParamsNSTR['B_ToDC']
                        B_LongHaul_TruckTrailer     = logitParamsNSTR['B_LongHaul_TruckTrailer']
                        B_LongHaul_TractorTrailer   = logitParamsNSTR['B_LongHaul_TractorTrailer']
                        #ASC_SS                      = [logitParamsNSTR[f'ASC_SS_{i+1}'] for i in range(nShipSizes)]
                        ASC_VT                      = [logitParamsNSTR[f'ASC_VT_{i+1}'] for i in range(nVehTypes)]
                    
                        for ft in range(nFlowTypesInternal):
                            
                            allocatedWeight = 0
                            totalWeight     = demandInternalByFT[ft][logSeg] * logSegToNstr[nstr,logSeg]
                            
                            # While the weight of all synthesized shipment for this segment so far does not exceed the total weight for this segment
                            while allocatedWeight < totalWeight:
                                flowType[count]        = ft + 1
                                goodsType[count]       = nstr
                                logisticSegment[count] = logSeg
                                
                                rand = np.random.rand()
                                
                                if logSeg == 6:
                                    cep   = np.where(cepSharesInternal > rand)[0][0]
                                    depot = np.where(cepDepotShares[cep] > np.random.rand())[0][0]
                                    toFirm[count]   = 0
                                    destZone[count] = cepDepotZones[cep][depot]
                                    destX[count]    = cepDepotX[cep][depot]
                                    destY[count]    = cepDepotY[cep][depot]
                                    toDC            = 1
                                    
                                # Determine receiving firm for flows to consumer
                                elif (flowType[count] in (1,3,6)):
                                    toFirm[count]   = np.where(cumProbReceive[:,nstr] > rand)[0][0]
                                    destZone[count] = firmZone[toFirm[count]]
                                    destX[count]    = firmX[toFirm[count]]
                                    destY[count]    = firmY[toFirm[count]]
                                    toDC            = 0
                                    
                                # Determine receiving DC for flows to DC
                                elif (flowType[count] in (2,5,8)):
                                    toFirm[count]   = np.where(cumProbDC > rand)[0][0]
                                    destZone[count] = dcZones[toFirm[count]]
                                    destX[count]    = logNodesX[toFirm[count]]
                                    destY[count]    = logNodesY[toFirm[count]]
                                    toDC            = 1
                                    
                                # Determine receiving Transshipment Terminal for flows to TT
                                elif (flowType[count] in (4,7,9)):
                                    toFirm[count]   = ttZones[np.where(cumProbTT > rand)[0][0]]
                                    destZone[count] = toFirm[count]
                                    destX[count]    = zoneX[toFirm[count]]
                                    destY[count]    = zoneY[toFirm[count]]
                                    toDC            = 0                        
            
                                distanceDecay  = (costPerHourSourcing * skimTravTime[destZone[count]::nZones] / 3600) + \
                                                 (costPerKmSourcing   * skimDistance[destZone[count]::nZones] / 1000)
                                distanceDecay  = 1 / (1 + np.exp(alpha + beta * np.log(distanceDecay)))
                                if   (flowType[count] in (1,2,4)):
                                    distanceDecay = distanceDecay[firmZone]
                                elif (flowType[count] in (3,5,7)):
                                    distanceDecay = distanceDecay[dcZones]
                                elif (flowType[count] in (6,8,9)):
                                    distanceDecay = distanceDecay[ttZones]
                                distanceDecay /= np.sum(distanceDecay)
            
                                rand = np.random.rand()

                                if logSeg == 6:
                                    origDepot = np.where(cepDepotShares[cep] > np.random.rand())[0][0]
                                    if origDepot == depot:
                                        if depot == 0:
                                            origDepot = origDepot + 1
                                        elif depot == len(cepDepotShares[cep])-1:
                                            origDepot = origDepot - 1
                                        else:
                                            origDepot = origDepot + [-1,1][np.random.randint(2)]
                                    depot = origDepot
                                    fromFirm[count] = 0
                                    origZone[count] = cepDepotZones[cep][depot]
                                    origX[count]    = cepDepotX[cep][depot]
                                    origY[count]    = cepDepotY[cep][depot]
                                    fromDC          = 1
                                    
                                # Determine sending firm for flows from consumer
                                elif (flowType[count] in (1,2,4)):
                                    prob = probSend[:,nstr] * distanceDecay
                                    prob = np.cumsum(prob)
                                    prob /= prob[-1]
                                    fromFirm[count] = np.where(prob > rand)[0][0]
                                    origZone[count] = firmZone[fromFirm[count]]
                                    origX[count]    = firmX[fromFirm[count]]
                                    origY[count]    = firmY[fromFirm[count]]
                                    fromDC          = 0
                                    
                                # Determine sending DC for flows from DC
                                elif (flowType[count] in (3,5,7)):
                                    prob = probDC * distanceDecay
                                    prob = np.cumsum(prob)
                                    prob /= prob[-1]
                                    fromFirm[count] = np.where(prob > rand)[0][0]
                                    origZone[count] = dcZones[fromFirm[count]]
                                    origX[count]    = logNodesX[fromFirm[count]]
                                    origY[count]    = logNodesY[fromFirm[count]]
                                    fromDC          = 1
                                    
                                # Determine sending Transshipment Terminal for flows from TT
                                elif (flowType[count] in (6,8,9)):
                                    prob = probTT * distanceDecay
                                    prob = np.cumsum(prob)
                                    prob /= prob[-1]
                                    fromFirm[count] = ttZones[np.where(prob > rand)[0][0]]
                                    origZone[count] = fromFirm[count]
                                    origX[count]    = zoneX[fromFirm[count]]
                                    origY[count]    = zoneY[fromFirm[count]]
                                    fromDC          = 0
                                
                                rand = np.random.rand()
                                
                                # Determine values for attributes in the utility function of the shipment size/vehicle type MNL
                                travTime        = skimTravTime[(origZone[count])*nZones + (destZone[count])] / 3600
                                distance        = skimDistance[(origZone[count])*nZones + (destZone[count])] / 1000      
                                inventoryCosts  = absoluteShipmentSizes
                                longHaul        = (distance > 100)
                                
                                # Determine the utility and probability for each alternative
                                utilities = np.zeros((1,nVehTypes*nShipSizes))[0,:]
                                
                                for ss in range(nShipSizes):
                                    for vt in range(nVehTypes):
                                        index           = ss * nVehTypes + vt                        
                                        transportCosts  =  costPerHour[vt]*travTime + costPerKm[vt]*distance
                                        
                                        # Multiply transport costs by number of required vehicles
                                        transportCosts *= np.ceil(absoluteShipmentSizes[ss] / truckCapacities[vt])
                                        
                                        # Utility function
                                        utilities[index] =  B_TransportCosts * transportCosts + B_InventoryCosts * inventoryCosts[ss] + \
                                                            B_FromDC * fromDC * (vt==0) + \
                                                            B_ToDC * toDC * (vt in [3,4,5]) + \
                                                            B_LongHaul_TruckTrailer * longHaul * (vt in [3,4]) + \
                                                            B_LongHaul_TractorTrailer * longHaul * (vt==5) + \
                                                            ASC_VT[vt]                                          
                                                            
                                probabilities       = [np.exp(u)/np.sum(np.exp(utilities)) for u in utilities]
                                cumProbabilities    = np.cumsum(probabilities)
                                            
                                # Sample one choice based on the cumulative probability distribution
                                rand = np.random.rand()
                                
                                ssvt = np.where([cumProbabilities[i] > rand and utilities[i]!=-99999 for i in range(nVehTypes*nShipSizes)])[0][0]
                    
                                # The chosen shipment size category                  
                                ssChosen = int(np.floor(ssvt/nVehTypes))
                                shipmentSizeCat[count] = ssChosen
                                shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                                
                                # The chosen vehicle type
                                vehicleType[count] = ssvt - ssChosen*nVehTypes
                                
                                # Update weight and counter
                                allocatedWeight         += shipmentSize[count]
                                allocatedWeightInternal += shipmentSize[count]
                                count += 1
                                
                                if count%300 == 0:
                                    root.progressBar['value'] = percStart + (percEnd - percStart) * (allocatedWeightInternal / totalWeightInternal)


                          
            if ExtArea:
                
                root.update_statusbar("Shipment Synthesizer: Synthesizing shipments leaving study area")
                log_file.write("Synthesizing shipments leaving study area...\n")
                percStart = 40
                percEnd   = 65
                root.progressBar['value'] = percStart

                # For progress bar
                totalWeightExport = np.sum([np.sum(np.sum(demandExportByFT[ft])) for ft in range(nFlowTypesExternal)])
                allocatedWeightExport  = 0
                
                for logSeg in range(nLogSeg):
                    
                    for nstr in range(nNSTR):
                        
                        if logSegToNstr[nstr,logSeg] > 0:
                            root.update_statusbar("Shipment Synthesizer: Synthesizing shipments leaving study area (LS " + str(logSeg) + " and NSTR " + str(nstr) + ")")
                            log_file.write(f"\tFor logistic segment {logSeg} (NSTR{nstr})\n")
                                   
                            # Selecting the logit parameter for this NSTR group
                            logitParamsNSTR             = logitParams[dictNSTR[nstr]]
                            B_TransportCosts            = logitParamsNSTR['B_TransportCosts']
                            B_InventoryCosts            = logitParamsNSTR['B_InventoryCosts']
                            B_FromDC                    = logitParamsNSTR['B_FromDC']
                            B_ToDC                      = logitParamsNSTR['B_ToDC']
                            B_LongHaul_TruckTrailer     = logitParamsNSTR['B_LongHaul_TruckTrailer']
                            B_LongHaul_TractorTrailer   = logitParamsNSTR['B_LongHaul_TractorTrailer']
                            #ASC_SS                      = [logitParamsNSTR[f'ASC_SS_{i+1}'] for i in range(nShipSizes)]
                            ASC_VT                      = [logitParamsNSTR[f'ASC_VT_{i+1}'] for i in range(nVehTypes)]            
                            
                            for ft in range(nFlowTypesExternal):
                                
                                for dest in range(nSuperZones):                                    
                                    allocatedWeight = 0
                                    totalWeight     = demandExportByFT[ft][dest,logSeg] * logSegToNstr[nstr,logSeg]
            
                                    distanceDecay  = (costPerHourSourcing * skimTravTime[nInternalZones+dest::nZones] / 3600) + \
                                                     (costPerKmSourcing   * skimDistance[nInternalZones+dest::nZones] / 1000)
                                    distanceDecay  = 1 / (1 + np.exp(alpha + beta * np.log(distanceDecay)))
                                    if   ft + 1 + nFlowTypesInternal == 10:
                                        distanceDecay = distanceDecay[firmZone]
                                    elif ft + 1 + nFlowTypesInternal == 11:
                                        distanceDecay = distanceDecay[dcZones]
                                    elif ft + 1 + nFlowTypesInternal == 12:
                                        distanceDecay = distanceDecay[ttZones]
                                    distanceDecay /= np.sum(distanceDecay)
                                   
                                    while allocatedWeight < totalWeight:
                                        flowType[count]         = ft + 1 + nFlowTypesInternal
                                        logisticSegment[count]  = logSeg
                                        goodsType[count]        = nstr                   
                                        toFirm[count]           = 0
                                        destZone[count]         = nInternalZones + dest
                                        destX[count]            = superZoneX[dest]
                                        destY[count]            = superZoneY[dest]
                                        
                                        rand = np.random.rand()   
    
                                        if logSeg == 6:
                                            cep   = np.where(cepSharesTotal > rand)[0][0]
                                            depot = np.where(cepDepotShares[cep] > np.random.rand())[0][0]
                                            fromFirm[count] = 0
                                            origZone[count] = cepDepotZones[cep][depot]
                                            origX[count]    = cepDepotX[cep][depot]
                                            origY[count]    = cepDepotY[cep][depot]
                                            fromDC          = 1
                                        
                                        # From consumer
                                        elif flowType[count] == 10:
                                            prob = probSend[:,nstr] * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]                                            
                                            fromFirm[count] = np.where(prob > rand)[0][0]
                                            origZone[count] = firmZone[fromFirm[count]]
                                            origX[count]    = firmX[fromFirm[count]]
                                            origY[count]    = firmY[fromFirm[count]]  
                                            fromDC          = 0
                                            
                                        # From distribution center
                                        elif flowType[count] == 11:
                                            prob = probDC * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]
                                            fromFirm[count] = np.where(prob > rand)[0][0]
                                            origZone[count] = dcZones[fromFirm[count]]
                                            origX[count]    = logNodesX[fromFirm[count]]
                                            origY[count]    = logNodesY[fromFirm[count]]    
                                            fromDC          = 1
                                            
                                        # From transshipment terminal
                                        elif flowType[count] == 12:
                                            prob = probTT * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]
                                            fromFirm[count] = ttZones[np.where(prob > rand)[0][0]]
                                            origZone[count] = fromFirm[count]
                                            origX[count]    = zoneX[fromFirm[count]]
                                            origY[count]    = zoneY[fromFirm[count]]
                                            fromDC          = 0
                                            
                                        rand = np.random.rand()
                                        
                                        # Determine values for attributes in the utility function of the shipment size/vehicle type MNL                            
                                        travTime        = skimTravTime[(origZone[count])*nZones + (destZone[count])] / 3600
                                        distance        = skimDistance[(origZone[count])*nZones + (destZone[count])] / 1000      
                                        inventoryCosts  = absoluteShipmentSizes
                                        longHaul        = (distance > 100)
                                        
                                        # Determine the utility and probability for each alternative
                                        utilities = np.zeros((1,nVehTypes*nShipSizes))[0,:]
                                        
                                        for ss in range(nShipSizes):
                                            for vt in range(nVehTypes):
                                                index            = ss * nVehTypes + vt                                
                                                transportCosts   = costPerHour[vt]*travTime + costPerKm[vt]*distance  
                                                transportCosts  *= np.ceil(absoluteShipmentSizes[ss] / truckCapacities[vt])
                                                utilities[index] =  B_TransportCosts * transportCosts + B_InventoryCosts * inventoryCosts[ss] + \
                                                                    B_FromDC * fromDC * (vt==0) + \
                                                                    B_ToDC * toDC * (vt in [3,4,5]) + \
                                                                    B_LongHaul_TruckTrailer * longHaul * (vt in [3,4]) + \
                                                                    B_LongHaul_TractorTrailer * longHaul * (vt==5) + \
                                                                    ASC_VT[vt]
                                        
                                        probabilities       = [np.exp(u)/np.sum(np.exp(utilities)) for u in utilities]
                                        cumProbabilities    = np.cumsum(probabilities)
                                        
                                        # Sample one choice based on the cumulative probability distribution
                                        rand = np.random.rand()
                                        
                                        ssvt = np.where([cumProbabilities[i] > rand and utilities[i]!=-99999 for i in range(nVehTypes*nShipSizes)])[0][0]
                            
                                        # The chosen shipment size category                  
                                        ssChosen = int(np.floor(ssvt/nVehTypes))
                                        shipmentSizeCat[count] = ssChosen
                                        shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                                        
                                        # The chosen vehicle type
                                        vehicleType[count] = ssvt - ssChosen*nVehTypes
                                                        
                                        # Update weight and counter
                                        allocatedWeight         += shipmentSize[count]        
                                        allocatedWeightExport   += shipmentSize[count]  
                                        count += 1

                                        if count%300 == 0:
                                            root.progressBar['value'] = percStart + (percEnd - percStart) * (allocatedWeightExport/ totalWeightExport)                                        
                                        
                                         
                fromDC = 0
                
                root.update_statusbar("Shipment Synthesizer: Synthesizing shipments entering study area")
                log_file.write("Synthesizing shipments entering study area...\n")
                percStart = 65
                percEnd   = 90
                root.progressBar['value'] = percStart

                totalWeightImport = np.sum([np.sum(np.sum(demandImportByFT[ft])) for ft in range(nFlowTypesExternal)])
                allocatedWeightImport  = 0
                
                for logSeg in range(nLogSeg):
                    
                    for nstr in range(nNSTR):
                        
                        if logSegToNstr[nstr,logSeg] > 0:
                            root.update_statusbar("Shipment Synthesizer: Synthesizing shipments entering study area (LS " + str(logSeg) + " and NSTR " + str(nstr) + ")")
                            log_file.write(f"\tFor logistic segment {logSeg} (NSTR{nstr})\n")                        
                        
                            # Selecting the logit parameter for this NSTR group
                            logitParamsNSTR             = logitParams[dictNSTR[nstr]]
                            B_TransportCosts            = logitParamsNSTR['B_TransportCosts']
                            B_InventoryCosts            = logitParamsNSTR['B_InventoryCosts']
                            B_FromDC                    = logitParamsNSTR['B_FromDC']
                            B_ToDC                      = logitParamsNSTR['B_ToDC']
                            B_LongHaul_TruckTrailer     = logitParamsNSTR['B_LongHaul_TruckTrailer']
                            B_LongHaul_TractorTrailer   = logitParamsNSTR['B_LongHaul_TractorTrailer']
                            #ASC_SS                      = [logitParamsNSTR[f'ASC_SS_{i+1}'] for i in range(nShipSizes)]
                            ASC_VT                      = [logitParamsNSTR[f'ASC_VT_{i+1}'] for i in range(nVehTypes)]
                            
                            for ft in range(nFlowTypesExternal):
                                
                                for orig in range(nSuperZones):                                                                            
                                    allocatedWeight = 0
                                    totalWeight     = demandImportByFT[ft][orig,logSeg] * logSegToNstr[nstr,logSeg]
            
                                    distanceDecay  = (costPerHourSourcing * skimTravTime[(nInternalZones+orig)*nZones + np.arange(nZones)] / 3600) + \
                                                     (costPerKmSourcing   * skimDistance[(nInternalZones+orig)*nZones + np.arange(nZones)] / 1000)
                                    distanceDecay  = 1 / (1 + np.exp(alpha + beta * np.log(distanceDecay)))
                                    if   ft + 1 + nFlowTypesInternal == 10:
                                        distanceDecay = distanceDecay[firmZone]
                                    elif ft + 1 + nFlowTypesInternal == 11:
                                        distanceDecay = distanceDecay[dcZones]
                                    elif ft + 1 + nFlowTypesInternal == 12:
                                        distanceDecay = distanceDecay[ttZones]
                                    distanceDecay /= np.sum(distanceDecay)
                                    
                                    while allocatedWeight < totalWeight:
                                        flowType[count]         = ft + 1 + nFlowTypesInternal
                                        logisticSegment[count]  = logSeg
                                        goodsType[count]        = nstr
                                        fromFirm[count]         = 0
                                        origZone[count]         = nInternalZones + orig
                                        origX[count]            = superZoneX[orig]
                                        origY[count]            = superZoneY[orig]
                                        
                                        rand = np.random.rand()                      

                                        if logSeg == 6:
                                            cep   = np.where(cepSharesTotal > rand)[0][0]
                                            depot = np.where(cepDepotShares[cep] > np.random.rand())[0][0]
                                            toFirm[count]   = 0
                                            destZone[count] = cepDepotZones[cep][depot]
                                            destX[count]    = cepDepotX[cep][depot]
                                            destY[count]    = cepDepotY[cep][depot]
                                            toDC            = 1
                                            
                                        # To consumer
                                        elif flowType[count] == 10:
                                            prob = probReceive[:,nstr] * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]
                                            toFirm[count]   = np.where(prob > rand)[0][0]
                                            destZone[count] = firmZone[toFirm[count]]
                                            destX[count]    = firmX[toFirm[count]]
                                            destY[count]    = firmY[toFirm[count]]
                                            toDC            = 0
                                            
                                        # To distribution center
                                        elif flowType[count] == 11:
                                            prob = probDC * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]
                                            toFirm[count]   = np.where(prob > rand)[0][0]
                                            destZone[count] = dcZones[toFirm[count]]
                                            destX[count]    = logNodesX[toFirm[count]]
                                            destY[count]    = logNodesY[toFirm[count]]
                                            toDC            = 1
                                            
                                        # To transshipment terminal
                                        elif flowType[count] == 12:
                                            prob = probTT * distanceDecay
                                            prob = np.cumsum(prob)
                                            prob /= prob[-1]
                                            toFirm[count]   = ttZones[np.where(prob > rand)[0][0]]
                                            destZone[count] = toFirm[count]
                                            destX[count]    = zoneX[toFirm[count]]
                                            destY[count]    = zoneY[toFirm[count]]
                                            toDC            = 0
                                            
                                        rand = np.random.rand()
                                        
                                        # Determine values for attributes in the utility function of the shipment size/vehicle type MNL                           
                                        travTime        = skimTravTime[(origZone[count])*nZones + (destZone[count])] / 3600
                                        distance        = skimDistance[(origZone[count])*nZones + (destZone[count])] / 1000      
                                        inventoryCosts  = absoluteShipmentSizes
                                        longHaul        = (distance > 100)
                                        
                                        # Determine the utility and probability for each alternative
                                        utilities = np.zeros((1,nVehTypes*nShipSizes))[0,:]
                                        
                                        for ss in range(nShipSizes):
                                            for vt in range(nVehTypes):
                                                index = ss * nVehTypes + vt                                
                                                transportCosts   =  costPerHour[vt]*travTime + costPerKm[vt]*distance
                                                transportCosts  *= np.ceil(absoluteShipmentSizes[ss] / truckCapacities[vt])
                                                utilities[index] =  B_TransportCosts * transportCosts + B_InventoryCosts * inventoryCosts[ss] + \
                                                                    B_FromDC * fromDC * (vt==0) + \
                                                                    B_ToDC * toDC * (vt in [3,4,5]) + \
                                                                    B_LongHaul_TruckTrailer * longHaul * (vt in [3,4]) + \
                                                                    B_LongHaul_TractorTrailer * longHaul * (vt==5) + \
                                                                    ASC_VT[vt]
                                                                        
                                        probabilities    = [np.exp(u)/np.sum(np.exp(utilities)) for u in utilities]
                                        cumProbabilities = np.cumsum(probabilities)
                                                
                                        # Sample one choice based on the cumulative probability distribution
                                        rand = np.random.rand()
                                        
                                        ssvt = np.where([cumProbabilities[i] > rand and utilities[i]!=-99999 for i in range(nVehTypes*nShipSizes)])[0][0]
                            
                                        # The chosen shipment size category                  
                                        ssChosen = int(np.floor(ssvt/nVehTypes))
                                        shipmentSizeCat[count] = ssChosen
                                        shipmentSize[count] = min(absoluteShipmentSizes[ssChosen], totalWeight - allocatedWeight)
                                        
                                        # The chosen vehicle type
                                        vehicleType[count] = ssvt - ssChosen*nVehTypes
                                                        
                                        # Update weight and counter
                                        allocatedWeight         += shipmentSize[count]
                                        allocatedWeightImport   += shipmentSize[count]
                                        count += 1            

                                        if count%300 == 0:
                                            root.progressBar['value'] = percStart + (percEnd - percStart) * (allocatedWeightImport / totalWeightImport)                                        

            
            # Shipment attributes in a list instead of a dictionary        
            fromFirm        = list(fromFirm.values())
            toFirm          = list(toFirm.values())
            flowType        = list(flowType.values())
            logisticSegment = list(logisticSegment.values())
            goodsType       = list(goodsType.values())
            shipmentSize    = list(shipmentSize.values())
            shipmentSizeCat = list(shipmentSizeCat.values())
            vehicleType     = list(vehicleType.values())
            origZone        = list(origZone.values())
            destZone        = list(destZone.values())
            
            nShips = len(fromFirm)
            
            
            
            # ----------------------- Creating shipments CSV --------------------------
            
            shipCols  = ["SHIP_ID",   "ORIG",         "DEST",     "NSTR",      \
                         "WEIGHT",    "WEIGHT_CAT",   "FLOWTYPE", "LOGSEG",  "VEHTYPE",   \
                         "SEND_FIRM", "RECEIVE_FIRM", "SEND_DC",  "RECEIVE_DC"]
            shipments = pd.DataFrame(np.zeros((nShips,len(shipCols))))
            shipments.columns = shipCols
    
            shipments['SHIP_ID'     ] = np.arange(nShips)
            shipments['ORIG'        ] = [zoneDict[x] for x in origZone]
            shipments['DEST'        ] = [zoneDict[x] for x in destZone]
            shipments['NSTR'        ] = goodsType
            shipments['WEIGHT'      ] = shipmentSize
            shipments['WEIGHT_CAT'  ] = shipmentSizeCat
            shipments['FLOWTYPE'    ] = flowType
            shipments['LOGSEG'      ] = logisticSegment
            shipments['VEHTYPE'     ] = vehicleType
            shipments['SEND_FIRM'   ] = firmID[fromFirm]
            shipments['RECEIVE_FIRM'] = firmID[toFirm]
            shipments['SEND_DC'     ] = -99999
            shipments['RECEIVE_DC'  ] = -99999
            
            # For the external zones and logistical nodes there is no firm, hence firm ID -99999
            shipments.loc[(shipments['ORIG'] > 99999900), 'SEND_FIRM'   ] = -99999
            shipments.loc[(shipments['DEST'] > 99999900), 'RECEIVE_FIRM'] = -99999
            shipments.loc[(np.array(logisticSegment)== 6), ['SEND_FIRM','RECEIVE_FIRM']] = -99999
            shipments.loc[(np.array(flowType) > 10),       ['SEND_FIRM','RECEIVE_FIRM']] = -99999
            shipments.loc[(np.array(flowType)==2) | (np.array(flowType)==5) | (np.array(flowType)==8), 'RECEIVE_FIRM'] = -99999
            shipments.loc[(np.array(flowType)==4) | (np.array(flowType)==7) | (np.array(flowType)==9), 'RECEIVE_FIRM'] = -99999
            shipments.loc[(np.array(flowType)==3) | (np.array(flowType)==5) | (np.array(flowType)==7), 'SEND_FIRM'   ] = -99999
            shipments.loc[(np.array(flowType)==6) | (np.array(flowType)==8) | (np.array(flowType)==9), 'SEND_FIRM'   ] = -99999        
    
            # Only fill in DC ID for shipments to and from DC
            whereToDC   =   (shipments['FLOWTYPE']==2) | (shipments['FLOWTYPE']==5) | (shipments['FLOWTYPE']==8) \
                          | ((shipments['FLOWTYPE']==11) & (shipments['ORIG']>99999900))
            whereFromDC =   (shipments['FLOWTYPE']==3) | (shipments['FLOWTYPE']==5) | (shipments['FLOWTYPE']==7) \
                          | ((shipments['FLOWTYPE']==11) & (shipments['DEST']>99999900))                    
            shipments.loc[whereToDC,  'RECEIVE_DC'] = np.array(toFirm)[whereToDC]
            shipments.loc[whereFromDC,'SEND_DC'   ] = np.array(fromFirm)[whereFromDC]
        
        else:
            # Import the reference shipments
            shipments = pd.read_csv(shipmentsRef, index_col=0)
        
        # Get the datatypes right
        intCols  =  ["SHIP_ID",    "ORIG",         "DEST",    "NSTR",       \
                     "WEIGHT_CAT", "FLOWTYPE",     "LOGSEG",  "VEHTYPE",    \
                     "SEND_FIRM",  "RECEIVE_FIRM", "SEND_DC", "RECEIVE_DC"]
        floatCols = ['WEIGHT']
        shipments[intCols  ] = shipments[intCols].astype(int)
        shipments[floatCols] = shipments[floatCols].astype(float)
        
        # Redirect shipments via UCCs and change vehicle type
        if label == 'UCC':

            root.update_statusbar("Shipment Synthesizer: Exporting REF shipments to CSV")
            log_file.write('Exporting REF shipments to ' + datapathO + "Shipments_REF.csv\n")
            root.progressBar['value'] = 90
            
            if shipmentsRef == "":
                shipments.to_csv(datapathO + 'Shipments_REF.csv')

            root.update_statusbar("Shipment Synthesizer: Redirecting shipments via UCC")
            log_file.write("Redirecting shipments via UCC...\n")
            root.progressBar['value'] = 91
            
            shipments['FROM_UCC'] = 0
            shipments['TO_UCC'  ] = 0
          
            whereOrigZEZ = [i for i in shipments[shipments['ORIG']<99999900].index if zonesShape['ZEZ'][shipments['ORIG'][i]]==1]
            whereOrigZEZ = np.array(whereOrigZEZ, dtype=int)
            whereDestZEZ = [i for i in shipments[shipments['DEST']<99999900].index if zonesShape['ZEZ'][shipments['DEST'][i]]==1]
            whereDestZEZ = np.array(whereDestZEZ, dtype=int)
            setWhereOrigZEZ = set(whereOrigZEZ)
            setWhereDestZEZ = set(whereDestZEZ)
            
            whereBothZEZ = [i for i in shipments.index if i in setWhereOrigZEZ and i in setWhereDestZEZ]
            
            newShipments = pd.DataFrame(np.zeros(shipments.shape))
            newShipments.columns = shipments.columns
            newShipments[intCols  ] = newShipments[intCols].astype(int)
            newShipments[floatCols] = newShipments[floatCols].astype(float)
            
            count = 0
            
            for i in whereOrigZEZ:
                
                if i not in setWhereDestZEZ:
                    ls = int(shipments['LOGSEG'][i])
                    
                    if probConsolidation[ls][0] > np.random.rand():                   
                        trueOrigin = int(shipments['ORIG'][i])
                        newOrigin  = zonesShape['UCC_zone'][trueOrigin]            
                        
                        # Redirect to UCC
                        shipments.at[i,'ORIG'    ] = newOrigin
                        shipments.at[i,'FROM_UCC'] = 1
                        if shipmentsRef == "":
                            origX[i] = zoneX[invZoneDict[newOrigin]]
                            origY[i] = zoneY[invZoneDict[newOrigin]]                        
                        
                        # Add shipment from ZEZ to UCC
                        newShipments.loc[count,:] = list(shipments.loc[i,:].copy())
                        newShipments.at[count,'ORIG'    ] = trueOrigin
                        newShipments.at[count,'DEST'    ] = newOrigin
                        newShipments.at[count,'FROM_UCC'] = 0
                        newShipments.at[count,'TO_UCC'  ] = 1
                        newShipments.at[count,'VEHTYPE' ] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]]
                        if shipmentsRef == "":
                            origX[nShips+count] = zoneX[invZoneDict[trueOrigin]]
                            origY[nShips+count] = zoneY[invZoneDict[trueOrigin]]
                            destX[nShips+count] = zoneX[invZoneDict[newOrigin]]
                            destY[nShips+count] = zoneY[invZoneDict[newOrigin]]
                        
                        count += 1
                        
            for i in whereDestZEZ:
                
                if i not in setWhereOrigZEZ:
                    ls = int(shipments['LOGSEG'][i])
                    
                    if probConsolidation[ls][0] > np.random.rand():                   
                        trueDest = int(shipments['DEST'][i])
                        newDest  = zonesShape['UCC_zone'][trueDest]
                        
                        # Redirect to UCC
                        shipments.at[i,'DEST'  ] = newDest
                        shipments.at[i,'TO_UCC'] = 1
                        if shipmentsRef == "":
                            destX[i] = zoneX[invZoneDict[newDest]]
                            destY[i] = zoneY[invZoneDict[newDest]]   
                        
                        # Add shipment to ZEZ from UCC
                        newShipments.loc[count,:] = list(shipments.loc[i,:].copy())
                        newShipments.at[count,'ORIG'    ] = newDest
                        newShipments.at[count,'DEST'    ] = trueDest
                        newShipments.at[count,'FROM_UCC'] = 1
                        newShipments.at[count,'TO_UCC'  ] = 0
                        newShipments.at[count,'VEHTYPE' ] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]]
                        if shipmentsRef == "":
                            origX[nShips+count] = zoneX[invZoneDict[newDest]]
                            origY[nShips+count] = zoneY[invZoneDict[newDest]]
                            destX[nShips+count] = zoneX[invZoneDict[trueDest]]
                            destY[nShips+count] = zoneY[invZoneDict[trueDest]]
                        
                        count += 1
    
            # Also change vehicle type and rerouting for shipments that go from a ZEZ area to a ZEZ area
            for i in whereBothZEZ:
                ls = int(shipments['LOGSEG'][i])
                
                # Als het binnen dezelfde gemeente (i.e. dezelfde ZEZ) blijft, dan hoeven we alleen maar het voertuigtype aan te passen
                # Assume dangerous goods keep the same vehicle type
                if zonesShape['Gemeentena'][shipments['ORIG'][i]] == zonesShape['Gemeentena'][shipments['DEST'][i]]:                    
                    if ls != 7:
                        shipments.at[i,'VEHTYPE'] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]]
                
                # Als het van de ene ZEZ naar de andere ZEZ gaat, maken we 3 legs: ZEZ1--> UCC1, UCC1-->UCC2, UCC2-->ZEZ2
                else:
                    if probConsolidation[ls][0] > np.random.rand():
                        trueOrigin = int(shipments['ORIG'][i])
                        trueDest   = int(shipments['DEST'][i])
                        newOrigin  = zonesShape['UCC_zone'][trueOrigin]            
                        newDest    = zonesShape['UCC_zone'][trueDest]
                        
                        # Redirect to UCC
                        shipments.at[i,'ORIG'    ] = newOrigin
                        shipments.at[i,'FROM_UCC'] = 1
                        if shipmentsRef == "":
                            origX[i] = zoneX[invZoneDict[newOrigin]]
                            origY[i] = zoneY[invZoneDict[newOrigin]]                        
                        
                        # Add shipment from ZEZ1 to UCC1
                        newShipments.loc[count,:] = list(shipments.loc[i,:].copy())
                        newShipments.at[count,'ORIG'    ] = trueOrigin
                        newShipments.at[count,'DEST'    ] = newOrigin
                        newShipments.at[count,'FROM_UCC'] = 0
                        newShipments.at[count,'TO_UCC'  ] = 1
                        newShipments.at[count,'VEHTYPE' ] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]]
                        if shipmentsRef == "":
                            origX[nShips+count] = zoneX[invZoneDict[trueOrigin]]
                            origY[nShips+count] = zoneY[invZoneDict[trueOrigin]]
                            destX[nShips+count] = zoneX[invZoneDict[newOrigin]]
                            destY[nShips+count] = zoneY[invZoneDict[newOrigin]]
                        
                        count += 1                        
                        
                        # Redirect to UCC
                        shipments.at[i,'DEST'  ] = newDest
                        shipments.at[i,'TO_UCC'] = 1
                        if shipmentsRef == "":
                            destX[i] = zoneX[invZoneDict[newDest]]
                            destY[i] = zoneY[invZoneDict[newDest]]   
                        
                        # Add shipment from UCC2 to ZEZ2
                        newShipments.loc[count,:] = list(shipments.loc[i,:].copy())
                        newShipments.at[count,'ORIG'    ] = newDest
                        newShipments.at[count,'DEST'    ] = trueDest
                        newShipments.at[count,'FROM_UCC'] = 1
                        newShipments.at[count,'TO_UCC'  ] = 0
                        newShipments.at[count,'VEHTYPE' ] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]]
                        if shipmentsRef == "":
                            origX[nShips+count] = zoneX[invZoneDict[newDest]]
                            origY[nShips+count] = zoneY[invZoneDict[newDest]]
                            destX[nShips+count] = zoneX[invZoneDict[trueDest]]
                            destY[nShips+count] = zoneY[invZoneDict[trueDest]]
                        
                        count += 1            

            newShipments = newShipments.iloc[np.arange(count),:]
            
            shipments = shipments.append(newShipments)
            nShips = len(shipments)
            shipments['SHIP_ID'] = np.arange(nShips)
            shipments.index      = np.arange(nShips)                        

        root.update_statusbar("Shipment Synthesizer: Exporting shipments to CSV")
        log_file.write('Exporting ' + str(label) + ' shipments to ' + datapathO + f"Shipments_{label}.csv\n")
        root.progressBar['value'] = 92               

        dtypes = {'SHIP_ID':int,  'ORIG':int,       'DEST':int,         'NSTR':int, \
                  'WEIGHT':float, 'WEIGHT_CAT':int, 'FLOWTYPE':int,     'LOGSEG':int, \
                  'VEHTYPE':int,  'SEND_FIRM':int,  'RECEIVE_FIRM':int, 'SEND_DC':int, 'RECEIVE_DC':int}
        for col in dtypes.keys():
            shipments[col] = shipments[col].astype(dtypes[col])
            
        shipments.to_csv(datapathO + f"Shipments_{label}.csv", index=False)  
              
        
        if shipmentsRef == "":
            
            # ---------------------- Zonal productions and attractions ----------------
            root.update_statusbar("Shipment Synthesizer: Writing zonal productions/attractions")
            log_file.write("Writing zonal productions/attractions...\n")
            root.progressBar['value'] = 93
            
            prodWeight = pd.pivot_table(shipments, values=['WEIGHT'], index=['ORIG','LOGSEG'], aggfunc=np.sum)
            attrWeight = pd.pivot_table(shipments, values=['WEIGHT'], index=['DEST','LOGSEG'], aggfunc=np.sum)
            zonalProductions = np.zeros((nInternalZones+nSuperZones,nLogSeg))
            zonalAttractions = np.zeros((nInternalZones+nSuperZones,nLogSeg))
            
            for x in prodWeight.index:
                orig = invZoneDict[x[0]]
                ls   = x[1]
                zonalProductions[orig, ls] += prodWeight['WEIGHT'][x]
            for x in attrWeight.index:
                orig = invZoneDict[x[0]]
                ls   = x[1]
                zonalAttractions[orig, ls] += attrWeight['WEIGHT'][x]                    
            
            cols = ['LS0','LS1','LS2','LS3','LS4','LS5','LS6','LS7']
            zonalProductions = pd.DataFrame(zonalProductions, columns=cols)
            zonalAttractions = pd.DataFrame(zonalAttractions, columns=cols)
            zonalProductions['ZONE'] = list(zoneDict.values())
            zonalAttractions['ZONE'] = list(zoneDict.values())
            zonalProductions['TOT_WEIGHT'] = np.sum(zonalProductions[cols], axis=1)
            zonalAttractions['TOT_WEIGHT'] = np.sum(zonalAttractions[cols], axis=1)
            
            cols = ['ZONE','LS0','LS1','LS2','LS3','LS4','LS5','LS6','LS7','TOT_WEIGHT']
            zonalProductions = zonalProductions[cols]
            zonalAttractions = zonalAttractions[cols]
            
            # Export to csv
            zonalProductions.to_csv(datapathO + f'zonal_productions_{label}.csv', index=False)
            zonalAttractions.to_csv(datapathO + f'zonal_attractions_{label}.csv', index=False)            
            
            
            
            # ------------------------- Creating shipments SHP ------------------------
            
            # Write into a geopandas dataframe and export as shapefile
            root.update_statusbar("Shipment Synthesizer: Writing GeoJSON")
            log_file.write("Writing GeoJSON...\n")
            percStart = 94
            percEnd   = 100
            root.progressBar['value'] = percStart            

            Ax = np.array(list(origX.values()), dtype=str)
            Ay = np.array(list(origY.values()), dtype=str)
            Bx = np.array(list(destX.values()), dtype=str)
            By = np.array(list(destY.values()), dtype=str)
            
            with open(datapathO + f"Shipments_{label}.geojson", 'w') as geoFile:
                geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
                for i in range(nShips-1):
                    outputStr = ""
                    outputStr = outputStr + '{ "type": "Feature", "properties": '
                    outputStr = outputStr + str(shipments.loc[i,:].to_dict()).replace("'",'"')
                    outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                    outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                    outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
                    geoFile.write(outputStr)
                    if i%1000==0:
                        root.progressBar['value'] = percStart + (percEnd - percStart) * (i / nShips)  
                        
                # Bij de laatste feature moet er geen komma aan het einde
                i += 1
                outputStr = ""
                outputStr = outputStr + '{ "type": "Feature", "properties": '
                outputStr = outputStr + str(shipments.loc[i,:].to_dict()).replace("'",'"')
                outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
                outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
                outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
                geoFile.write(outputStr)
                geoFile.write(']\n')
                geoFile.write('}')
            
            
            
        # --------------------------- End of module -------------------------------
            
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    

        root.update_statusbar("Shipment Synthesizer: Done")
        root.progressBar['value'] = 100
        
        # 0 means no errors in execution
        root.returnInfo = [0, [0,0]]
        
        return root.returnInfo
        
        
        
    except BaseException:
        import sys
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        import traceback
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()
        
        # Use this information to display as error message in GUI
        root.returnInfo = [1, [sys.exc_info()[0], traceback.format_exc()]]
        
        if __name__ == '__main__':
            root.update_statusbar("Shipment Synthesizer: Execution failed!")
            errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
            root.error_screen(text=errorMessage, size=[900,350])                
        
        else:
            return root.returnInfo




#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
        
if __name__ == '__main__':
    
    INPUTFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2016/'
    OUTPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunREF2016temp/'
    PARAMFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/'
    
    SKIMTIME        = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimTijd_REF.mtx'
    SKIMDISTANCE    = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimAfstand_REF.mtx'
    LINKS		    = INPUTFOLDER + 'links_v5.shp'
    NODES           = INPUTFOLDER + 'nodes_v5.shp'
    ZONES           = INPUTFOLDER + 'Zones_v4.shp'
    SEGS            = INPUTFOLDER + 'SEGS2016.csv'
    FIRMS           = INPUTFOLDER + 'SynthFirms_v15_1.csv'
    COMMODITYMATRIX = INPUTFOLDER + 'CommodityMatrixNUTS3_2016.csv'
    PARCELNODES     = INPUTFOLDER + 'parcelNodes_v2.shp'
    MRDH_TO_NUTS3   = PARAMFOLDER + 'MRDHtoNUTS32013.csv'
    NUTS3_TO_MRDH   = PARAMFOLDER + 'NUTS32013toMRDH.csv'
    
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
    
    MODULES = ['SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
    
    args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, \
            LINKS, NODES, ZONES, SEGS, FIRMS, \
            COMMODITYMATRIX, PARCELNODES, MRDH_TO_NUTS3, NUTS3_TO_MRDH, \
            PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
            PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
            YEARFACTOR, NUTSLEVEL_INPUT, \
            IMPEDANCE_SPEED, N_CPU, \
            SHIPMENTS_REF, SELECTED_LINKS,\
            LABEL, \
            MODULES]

    varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", \
                  "LINKS", "NODES", "ZONES", "SEGS", "FIRMS", \
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





    






    
