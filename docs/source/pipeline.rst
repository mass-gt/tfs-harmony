Pipeline & Module Architecture
==============================

tfs-harmony implements a sequential, modular freight simulation pipeline.
Each module exposes an `actually_run_module(root, varDict, dims)` function
that receives the full configuration dictionary and returns `[0, [0, 0]]` on
success or `[1, [exception, traceback]]` on failure.

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Module
     - Abbreviation
     - Responsibility
   * - Freight Generation & Distribution
     - FS
     - Produces commodity matrices per NUTS3 zone
   * - Spatial Interlining Formation
     - SIF
     - Allocates freight to logistic segments
   * - Shipment Generation
     - SHIP
     - Builds shipment lists from firm & commodity data
   * - Tour Formation
     - TOUR
     - Assigns shipments to vehicle tours (2-opt, savings)
   * - Parcel Demand
     - PARCEL\_DMND
     - Generates B2C/B2B parcel demand
   * - Parcel Scheduling
     - PARCEL\_SCHD
     - Schedules parcels onto vehicles & depots
   * - Service Provision
     - SERVICE
     - Models last-mile delivery services & micro-hubs
   * - Traffic Assignment & Emissions
     - TRAF
     - Assigns freight traffic and computes emissions
   * - Output
     - OUTP
     - Writes outputs (shapefiles, matrices, JSON summaries)

Data flow
---------

1. **FS** reads zonal shapefiles and commodity data → produces NUTS3 freight matrices.
2. **SIF** enriches with logistic-segment shares → per-segment shipment potential.
3. **SHIP** generates individual shipments with departure times and destinations.
4. **TOUR** solves vehicle-routing problems using 2-opt and savings heuristics.
5. **PARCEL\_DMND / PARCEL\_SCHD** model the growing e-commerce/parcel segment.
6. **SERVICE** adds micro-hub and crowd-shipping options.
7. **TRAF** performs traffic assignment and calculates emissions using factors.
8. **OUTP** consolidates all outputs into the `OUTPUTFOLDER`.

Each module's implementation lives in `src/mass_gt/calculation/<module>/`
with the pair `module_<x>.py` + `support_<x>.py`.
