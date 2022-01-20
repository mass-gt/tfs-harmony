# tfs-harmony

## Setting up a run
To calculate a scenario, run the MASS_GT_GUI.py script and enter the path to the .ini-file with configuration settings. 
An example of such an .ini-file is shown at the bottom. In this file you specify parameters and the paths to the input files to be used in the model run. 

The Tactical Freight Simulator has a large set of input files required for its calculations. To obtain the input files of the implementation in Zuid-Holland, the Netherlands, contact Sebastiaan Thoen (`@sebastiaanth` on GitHub). 

Besides the Python Standard Library, make sure you have the following libraries installed:
- numpy==1.19.1
- pandas==1.0.5
- scipy==1.5.0
- pyshp==2.1.0
- shapely==1.7.0
- numba==0.53.0

Finally, when you are using the Spyder IDE for running your Python scripts, make sure to have selected `Execute in an external system terminal` under `Tools-->Preferences-->Run-->Console`. This is necessary to make the scripts work that use parallelization of processes (tour formation module and traffic assignment module). 

## Further information
For more information on MASS-GT and the Tactical Freight Simulator, see: 
https://www.tudelft.nl/transport/onderzoeksthemas/goederenvervoer-logistiek/sleutelprojecten/mass-gt

## License
Please note that this code is made available under the GNU General Public License v2.0. 

## References
de Bok, M, L Tavasszy, I Kourounioti, S Thoen, L Eggers, V Mayland Nielsen, J Streng (2021) Application of the HARMONY tactical freight simulator to a case study for zero emission zones in Rotterdam, Transportation Research Records, in press

Thoen, S, L Tavasszy, M de Bok, G Correia, R van Duin (2020) Descriptive modeling of freight tour formation: A shipment-based approach, Transportation Research Part E, Volume 140, Pages XX – XX (https://doi.org/10.1016/j.tre.2020.101989)

de Bok, M, I Bal, L Tavasszy, T Tillema (2020) Exploring the impacts of an emission based truck charge in the Netherlands, Case Studies on Transport Policy, Volume 8, Pages 887 – 894. (https://doi.org/10.1016/j.cstp.2020.05.013)

Thoen, S, M de Bok and L Tavasszy (2020) Shipment-based urban freight emission calculation. 2020 Forum on Integrated and Sustainable Transportation Systems (FISTS) in Delft. (DOI: 10.1109/FISTS46898.2020.9264858)

de Bok, M, L Tavasszy, S Thoen (2020) Application of an empirical multi-agent model for urban goods transport to analyze impacts of zero emission zones in The Netherlands, Transport Policy, Volume XX, Pages XX – XX. (in press). (https://doi.org/10.1016/j.tranpol.2020.07.010)

de Bok, M, L Tavasszy (2018) "An empirical agent-based simulation system for urban goods transport (MASS-GT)." Procedia Computer Science, 130: 8. (https://doi.org/10.1016/j.procs.2018.04.021)


## Example .ini-file
```
# -------------- Which modules to run (separated by commas) ---------------------
MODULES=FS,SIF,SHIP,TOUR,PARCEL_DMND,PARCEL_SCHD,SERVICE,TRAF,OUTP

# ------------------- Scenario name ----------------------------------------------
LABEL = REF
#Current options are: REF, UCC

# -------------- Input and output folders ----------------------------------------
INPUTFOLDER  = C:\...\data\2016\
PARAMFOLDER  = C:\...\parameters\
OUTPUTFOLDER = C:\...\RunREF2016\

# ------------------- Input files ------------------------------------------------
SKIMTIME     = C:\...\data\LOS\2016\skimTijd_REF.mtx
SKIMDISTANCE = C:\...\data\LOS\2016\skimAfstand_REF.mtx
LINKS  = <<INPUTFOLDER>>links_v5.shp
NODES  = <<INPUTFOLDER>>nodes_v5.shp
ZONES  = <<INPUTFOLDER>>Zones_v5.shp
SEGS   = <<INPUTFOLDER>>SEGS2016_verrijkt.csv
COMMODITYMATRIX    = <<INPUTFOLDER>>CommodityMatrixNUTS3_2016.csv
PARCELNODES        = <<INPUTFOLDER>>parcelNodes_v2.shp
CEP_SHARES         = <<INPUTFOLDER>>CEPshares.csv
DISTRIBUTIECENTRA  = <<INPUTFOLDER>>distributieCentra.csv
DC_OPP_NUTS3       = <<INPUTFOLDER>>DC_OPP_NUTS3.csv
NSTR_TO_LS         = <<INPUTFOLDER>>nstrToLogisticSegment.csv
MAKE_DISTRIBUTION  = <<INPUTFOLDER>>MakeDistribution.csv
USE_DISTRIBUTION   = <<INPUTFOLDER>>UseDistribution.csv
SUP_COORDINATES_ID = <<INPUTFOLDER>>SupCoordinatesID.csv
CORRECTIONS_TONNES = <<INPUTFOLDER>>CorrectionsTonnes2016.csv
DEPTIME_PARCELS = <<INPUTFOLDER>>departureTimeParcelsCDF.csv
FIRMSIZE        = <<INPUTFOLDER>>FirmSizeDistributionPerSector_6cat.csv
SBI_TO_SEGS     = <<INPUTFOLDER>>Koppeltabel_sectoren_SBI_SEGs.csv

COST_VEHTYPE   = <<PARAMFOLDER>>Cost_VehType_2016.csv
COST_SOURCING  = <<PARAMFOLDER>>Cost_Sourcing_2016.csv
MRDH_TO_NUTS3  = <<PARAMFOLDER>>MRDHtoNUTS32013.csv
MRDH_TO_COROP  = <<PARAMFOLDER>>MRDHtoCOROP.csv
NUTS3_TO_MRDH  = <<PARAMFOLDER>>NUTS32013toMRDH.csv
SERVICE_DISTANCEDECAY = <<PARAMFOLDER>>Params_DistanceDecay_SERVICE.csv
SERVICE_PA            = <<PARAMFOLDER>>Params_PA_SERVICE.csv
VEHICLE_CAPACITY      = <<PARAMFOLDER>>CarryingCapacity.csv
LOGISTIC_FLOWTYPES    = <<PARAMFOLDER>>LogFlowtype_Shares.csv
PARAMS_TOD  = <<PARAMFOLDER>>Params_TOD.csv
PARAMS_SSVT = <<PARAMFOLDER>>Params_ShipSize_VehType.csv
PARAMS_ET_FIRST = <<PARAMFOLDER>>Params_EndTourFirst.csv
PARAMS_ET_LATER = <<PARAMFOLDER>>Params_EndTourLater.csv
PARAMS_SIF_PROD = <<PARAMFOLDER>>Params_PA_PROD.csv
PARAMS_SIF_ATTR = <<PARAMFOLDER>>Params_PA_ATTR.csv

EMISSIONFACS_BUITENWEG_LEEG = <<INPUTFOLDER>>EmissieFactoren_BUITENWEG_LEEG.csv
EMISSIONFACS_BUITENWEG_VOL  = <<INPUTFOLDER>>EmissieFactoren_BUITENWEG_VOL.csv
EMISSIONFACS_SNELWEG_LEEG = <<INPUTFOLDER>>EmissieFactoren_SNELWEG_LEEG.csv
EMISSIONFACS_SNELWEG_VOL  = <<INPUTFOLDER>>EmissieFactoren_SNELWEG_VOL.csv
EMISSIONFACS_STAD_LEEG = <<INPUTFOLDER>>EmissieFactoren_STAD_LEEG.csv
EMISSIONFACS_STAD_VOL  = <<INPUTFOLDER>>EmissieFactoren_STAD_VOL.csv

ZEZ_CONSOLIDATION = <<INPUTFOLDER>>ConsolidationPotential.csv
ZEZ_SCENARIO      = <<INPUTFOLDER>>ZEZscenario.csv

# ------------------- SIF parameters ---------------------------------------------
NUTSLEVEL_INPUT = 3

# ------------------- SHIP parameters --------------------------------------------
YEARFACTOR = 209

# ------------------ PARCEL parameters -------------------------------------------
PARCELS_PER_HH	 = 0.112
PARCELS_PER_EMPL = 0.041
PARCELS_MAXLOAD	 = 180
PARCELS_DROPTIME = 120
PARCELS_SUCCESS_B2C   = 0.75
PARCELS_SUCCESS_B2B   = 0.95
PARCELS_GROWTHFREIGHT = 1.0

MICROHUBS    = <<INPUTFOLDER>>Microhubs.csv
VEHICLETYPES = <<INPUTFOLDER>>Microhubs_vehicleTypes.csv

CROWDSHIPPING    = FALSE
#CRW_PARCELSHARE  = 0.06
#CRW_MODEPARAMS   = <<PARAMFOLDER>>Params_UseCase_CrowdShipping.csv
#CRW_PDEMAND_CAR  = <<INPUTFOLDER>>MRDH_2016_Auto_Etmaal.mtx
#CRW_PDEMAND_BIKE = <<INPUTFOLDER>>MRDH_2016_Fiets_Etmaal.mtx

# ---------------------- TRAF parameters -----------------------------------------
IMPEDANCE_SPEED_FREIGHT = V_FR_OS
IMPEDANCE_SPEED_VAN     = V_PA_OS

# ------------------- Optional settings ------------------------------------------
#SELECTED_LINKS = 
#SHIPMENTS_REF =
#FIRMS_REF =
#N_CPU = 
#NEAREST_DC =
#SHIFT_FREIGHT_TO_COMB1 =
#SHIFT_FREIGHT_TO_COMB2 =
#SHIFT_VAN_TO_COMB1
```
  
