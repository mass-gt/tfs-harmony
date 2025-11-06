# tfs-harmony

## Setting up a run
To calculate a scenario, run the MASS_GT_GUI.py script and enter the path to the .ini-file with configuration settings. 
An example of such an .ini-file is shown at the bottom. In this file you specify parameters and the paths to the input files to be used in the model run. 

The Tactical Freight Simulator has a large set of input files required for its calculations. To obtain the input files of the implementation in Zuid-Holland, the Netherlands, contact Sebastiaan Thoen (`@sebastiaanth` on GitHub). 

Besides the Python Standard Library, make sure you have the following libraries installed:
- numpy==2.1.3
- pandas==2.2.3
- scipy==1.14.1
- pyshp==2.3.1
- shapely==2.0.6
- numba==0.61.0
- tqdm==4.67.1
- psutil==7.1.3
The environment needs to have Python version 3.10.

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
DIMFOLDER = C:\...\dimensions\


# ------------------- Input files ------------------------------------------------
SKIMTIME     = <<INPUTFOLDER>>skimTijd_REF.mtx
SKIMDISTANCE = <<INPUTFOLDER>>skimAfstand_REF.mtx
SKIMDISTANCE_MIC_FIRSTLEG = <<INPUTFOLDER>>skimAfstand_MIC_FIRSTLEG.mtx
SKIMDISTANCE_MIC_LASTLEG = <<INPUTFOLDER>>skimAfstand_MIC_LASTLEG.mtx
LINKS  = <<INPUTFOLDER>>links_v13.shp
NODES  = <<INPUTFOLDER>>nodes_v13.shp
ZONES  = <<INPUTFOLDER>>areas_validated4.shp
SEGS   = <<INPUTFOLDER>>SEGS_2020_Verrijkt_v2.csv
COMMODITYMATRIX    = <<INPUTFOLDER>>commodity_matrix_v4.txt
PARCELNODES        = <<INPUTFOLDER>>parcelNodes_v2.shp
CEP_SHARES         = <<INPUTFOLDER>>courier_shares.txt
DISTRIBUTIECENTRA  = <<INPUTFOLDER>>distribution_centers.txt
DC_OPP_NUTS3       = <<INPUTFOLDER>>distribution_centers_surface_nuts3.txt
NSTR_TO_LS         = <<INPUTFOLDER>>nstr_to_logistic_segment.txt
MAKE_DISTRIBUTION  = <<INPUTFOLDER>>make_distribution.txt
USE_DISTRIBUTION   = <<INPUTFOLDER>>use_distribution.txt
SUP_COORDINATES_ID = <<INPUTFOLDER>>corop_coordinates.txt
CORRECTIONS_TONNES = <<INPUTFOLDER>>local_corrections.txt
DEPTIME_PARCELS = <<INPUTFOLDER>>departure_time_parcels.txt
FIRMSIZE        = <<INPUTFOLDER>>firm_size_distribution.txt
SBI_TO_SEGS     = <<INPUTFOLDER>>industry_sector_to_employment_sector.txt

COST_VEHTYPE   = <<PARAMFOLDER>>cost_figures_vehicle_type (2020).txt
COST_SOURCING  = <<PARAMFOLDER>>cost_figures_sourcing (2020).txt
MRDH_TO_NUTS3  = <<PARAMFOLDER>>mrdh_to_nuts3_2020.txt
MRDH_TO_COROP  = <<PARAMFOLDER>>mrdh_to_corop.txt
NUTS3_TO_MRDH  = <<PARAMFOLDER>>nuts3_2020_to_mrdh.txt
FREIGHT_DISTANCEDECAY = <<PARAMFOLDER>>coeffs_distance_decay_freight.txt
SERVICE_DISTANCEDECAY = <<PARAMFOLDER>>coeffs_distance_decay_service.txt
SERVICE_PA            = <<PARAMFOLDER>>Params_PA_SERVICE.csv
VEHICLE_CAPACITY      = <<PARAMFOLDER>>vehicle_capacity.txt
LOGISTIC_FLOWTYPES    = <<PARAMFOLDER>>flow_type_distribution.txt
PARAMS_TOD  = <<PARAMFOLDER>>Params_TOD.csv
PARAMS_SSVT = <<PARAMFOLDER>>coeffs_shipment_size_vehicle_type.txt
PARAMS_ET_FIRST = <<PARAMFOLDER>>coeffs_end_tour_first.txt
PARAMS_ET_LATER = <<PARAMFOLDER>>coeffs_end_tour_later.txt
PARAMS_SIF_PROD = <<PARAMFOLDER>>coeffs_freight_attr.txt
PARAMS_SIF_ATTR = <<PARAMFOLDER>>coeffs_freight_prod.txt
PARAMS_ECOMMERCE = <<PARAMFOLDER>>Params_EcommerceDemand.csv

EMISSIONFACS = <<INPUTFOLDER>>emission_factors.txt
ZEZ_CONSOLIDATION = <<INPUTFOLDER>>zez_consolidation_potential.txt
ZEZ_SCENARIO      = <<INPUTFOLDER>>zez_transition.txt
SEEDS = <<INPUTFOLDER>>seeds.txt

BIKE_LINKS = <<INPUTFOLDER>>bike_links_v11.shp
BIKE_NODES = <<INPUTFOLDER>>bike_nodes_v10.shp
CENTROIDS = <<INPUTFOLDER>>centroids_v10.shp
BIKE_MAT_OS = <<INPUTFOLDER>>Fiets-OS.MTX
BIKE_MAT_RD = <<INPUTFOLDER>>Fiets-RD.MTX
BIKE_MAT_AS = <<INPUTFOLDER>>Fiets-AS.MTX
MICRONODES = <<INPUTFOLDER>>micro_nodes.shp
TRAFFIC_LIGHTS = <<INPUTFOLDER>>traffic_lights.shp

# ------------------- SIF parameters ---------------------------------------------
NUTSLEVEL_INPUT = 3

# ------------------- SHIP parameters --------------------------------------------
YEARFACTOR = 209

# ------------------ PARCEL parameters -------------------------------------------
PARCELS_PER_PERSON = 0.1125
PARCELS_PER_EMPL = 0.0655
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
N_MULTIROUTE = 1

# -------------------- TRAF_BIKE parameters --------------------------------------
BIKE_USER_CLASSES = CABI
# Current options are: CABI,DIST,TIME,COMB (separated by commas)

BIKE_PERIODS = OS
# Current options are OS,RD,AS (separated by commas)

BIKE_TRAFFIC_LIGHT_SECONDS = 30

# ------------------- Optional settings ------------------------------------------
#SELECTED_LINKS = 
#SHIPMENTS_REF =
#FIRMS_REF =
#N_CPU = 
DAY_TO_WEEK_FACTOR = 1.0
#NEAREST_DC =
#SHIFT_FREIGHT_TO_COMB1 =
#SHIFT_FREIGHT_TO_COMB2 =
#SHIFT_VAN_TO_COMB1
```
  
