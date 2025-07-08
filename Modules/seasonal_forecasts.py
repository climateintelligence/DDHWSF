import numpy as np
from netCDF4 import Dataset
import glob
import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import cos, asin, sqrt, pi
import os

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(1, '../Modules/')
from ML_models_regressors import clf_LR, clf_SVR, clf_DT, clf_RF, clf_neigh, clf_AB, clf_MLP, clf_LGBM

### Reformat optimal predictors ###

def create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel):
    board = np.zeros((n_rows, n_cols))
    
    for i in range(n_cols):
        start_index = int(final_sequence[i]) 
        end_index = int(final_sequence[i])  + int(sequence_length[i])
        if feat_sel[i] != 0:
            board[start_index:end_index, i] = 1
    
    return board


### Produce forecasts based on optimal predictors ###

# Inputs: p - grid point index, mod - ML model, pred_dataframe - full list of predcitors, remove-co2 - boolean to remove CO2 as predictor
def forecast(target_past2k, target_era5, period, sol, mod, pred_dataframe, remove_co2):
    first_train = "7002-04-30"
    last_train = "1993-04-30"
    target_dates=[] # dummy date for summer HWMI
    train_years_past2k=range(7001,8851,1)
    train_years_era5=range(period[0],period[1]+1,1)
    for year in train_years_past2k:
        target_dates.append(str(year).zfill(4)+"-04-30") # Days in June #
    for year in train_years_era5:
        target_dates.append(str(year).zfill(4)+"-04-30") # Days in June #
    
    target=np.concatenate((target_past2k, target_era5))
    array_best=sol

    df=pd.DataFrame(target,columns=['Target'])
    df.index = target_dates
    target_dataset=df

    split=int(array_best.shape[0]/3)
    
    sequence_length_best = array_best[0:split]
    final_sequence_best = array_best[split:split*2]
    feat_sel_best = array_best[split*2:]

    if remove_co2:
        sequence_length_best=np.delete(sequence_length_best, -2)
        final_sequence_best=np.delete(final_sequence_best, -2)
        feat_sel_best=np.delete(feat_sel_best, -2)
        n_cols = split-1
    else:
        n_cols = split
    n_rows = int((sequence_length_best + final_sequence_best).max())+10
    
    board_best = create_board(n_rows, n_cols, final_sequence_best, sequence_length_best, feat_sel_best)

    time_lags = np.repeat(0,pred_dataframe.shape[1])
    time_sequences = np.repeat(n_rows,n_cols)
    variable_selection = np.repeat(1,pred_dataframe.shape[1])
    
    dataset_opt = target_dataset.copy()
    for i,col in enumerate(pred_dataframe.columns):      
        if remove_co2:
            if col=="data_CO2":
                continue
        if col=="weekofyear":
            i=i-1
        if variable_selection[i] == 0:
            continue
        for j in range(time_sequences[i]):
            dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = pred_dataframe[col].shift(time_lags[i]+j)

        
    first_train_index=int(np.argwhere(df.index==first_train))
    last_train_index=int(np.argwhere(df.index==last_train))
    
    train_dataset_opt = dataset_opt[first_train_index:last_train_index]
    test_dataset_opt = dataset_opt[last_train_index:]

    Y_column = 'Target' 
           
    X_train=train_dataset_opt[train_dataset_opt.columns.drop([Y_column]) ]
    Y_train=train_dataset_opt[Y_column]
    X_test=test_dataset_opt[test_dataset_opt.columns.drop([Y_column]) ]
    Y_test=test_dataset_opt[Y_column]

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    X_std_train = scaler.fit(X_train)

    X_std_train = scaler.transform(X_train)
    X_std_test = scaler.transform(X_test)

    X_train=pd.DataFrame(X_std_train,columns=X_train.columns,index=X_train.index)
    X_test=pd.DataFrame(X_std_test,columns=X_test.columns,index=X_test.index)
    
    X_train_new = np.array(X_train)[:,board_best.T.reshape(1,-1)[0].astype(bool)]
    X_test_new = np.array(X_test)[:,board_best.T.reshape(1,-1)[0].astype(bool)]

    clf = mod
    clf.fit(X_train_new, Y_train)
    predictions = clf.predict(X_test_new)
    Y_pred = pd.DataFrame(predictions, columns=['Y_pred'], index=Y_test.index)
    
    return clf.__class__.__name__,predictions # full model name, 1D output forecast (year)


### Save output of forecast/predout ###

def saver(output_file, preds, l):
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Create DataFrame
    df = pd.DataFrame({
        'year': range(1979, 1979 + l),
        'pred': np.array(preds)
    })

    # Write metadata as comments at the top of the file
    with open(output_file, 'w') as f:
        f.write(F"# DDHWSF Forecasts of NDQ90")

        # Append the DataFrame
        df.to_csv(f, index=False, header=['year', 'NDQ90_predictions'])
    
    print(f"Saved predictions with metadata to {output_file}")
