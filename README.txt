This directory contains the:
- Predictor data from past2k and ERA5 (area-averaged time series of clusters) files.
- Target data (number of summer HW days) for past2k and ERA5
- Feature selection and forecast skill optimiser (optimisation_past2k_NRMSE.py)
- Seasonal forecasting based on random forest (other models from ML_models_regressors.py can be used)

(1) OPTIMISATION-BASED FEATURE SELECTION

Run optimisation_past2k_NRMSE.py for each coordinate pairing x/longitude [0..40] and y/latitude [0..25]
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