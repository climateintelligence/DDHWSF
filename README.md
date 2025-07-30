# DDHWSF - Data-Driven Heatwave Season Forecasting 

DDHWSF is a two-step framework for data-driven seasonal forecatsing of heatwaves. The framework consists of: (1) an optimisaiton-based feature selection, which identifies key predictors of heatwaves in a paleo-climate simulation and (2) ML regressors traine don the paleo-climate predictors and applied to ERA5 (e.g. "real-world") predictors. 


This directory contains the:
- Predictor data from past2k and ERA5 (area-averaged time series of clusters) files.
- Target data (number of summer HW days) for past2k and ERA5
- Feature selection and forecast skill optimiser (optimisation_past2k_NRMSE.py)
- Seasonal forecasting based on random forest (other models from ML_models_regressors.py can be used)

(1) OPTIMISATION-BASED FEATURE SELECTION

Run optimisation_past2k_NRMSE.py for each coordinate pairing x/longitude and y/latitude
(job_optimiser is the file used to run on the DKRZ Levante system - this will need to be changed for other systems)

Inputs: 2D Heatwave data from model world "past2k_tasmax_HWs_EUR_MJJA_period70018850_clim88218850.nc" / Predictor data "Predictors_dataset_past2k_weekly.csv"
Outputs: Optimised solutions "opt_MayMJJ_past2k_y_x.csv"

(2) SEASONAL FORECASTS

Uses optimised solutions to forecast seasonal HW indicators over test period
Inputs: 2D heatwave data from real world "ERA5_tmax_HWs_EUR_NDQ90_period19402022_clim19932016.nc" / Predictor data "Predictors_dataset_ERA5_weekly.csv"
Output: "Forecasts_initMay_targMJJ_RF_19502022.nc"

RF - Random Forecast
MJJ - May-June-July
CRO - Coral Reef Optimization
past2k - 1850-year paleo-climate simulation / "model world"
ERA5 - reanalysis / "real world"

LICENSE

This project is licensed under the European Union Public Licence v.1.2 (EUPL-1.2).  
See the [LICENSE](./LICENSE) file for full terms.

DEPENDENCIES AND ACKNOWLEDGMENTS

This project makes use of pyCRO-SL – a Python implementation of the Coral Reef Optimization with Substrate Layers algorithm (https://pypi.org/project/PyCROSL/)
Licensed under: MIT

CMIP6 DATA ATTRIBUTION & LICENSE

Some datasets in this repository are derived from CMIP6 model simulations provided bu the Max Planck Institute for Meterology (MPI-M), using the model MPI-ESM1.2-LR. The original simulation data are avilable via the Earth System Grid Federation (ESGF).
- Original data license: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- CMIP6 Terms of Use: https://pcmdi.llnl.gov/CMIP6/TermsOfUse
- Original model output DOI or ESGF link: https://pcmdi.llnl.gov/CMIP6/ArchiveStatistics/esgf_data_holdings/PMIP/index.html (MPI-ESM1-2-LR, "past2k")
Please cite the data following CMIP6 guidelines and attribute the MPI-M as the original source.

ERA5 DATA ATTRIBUTION & LICENSE

Some datasets in this repository are derived from the ERA5 reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) and made available via the Copernicus Climate Change Service (C3S).

Source: ERA5 Reanalysis, Copernicus Climate Data Store
URL: https://cds.climate.copernicus.eu/
License: Copernicus Licence to Use
Derived data license: MIT License
Please credit the original source as follows:
“Contains modified Copernicus Climate Change Service information 2025”
