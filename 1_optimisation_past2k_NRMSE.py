import sys
sys.path.insert(0, '/home/b/b382634/UAH_Repository/ML2EE/')
from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *
# from skelm import ELMClassifier
# from input_data import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, root_mean_squared_error
import warnings
import sys
warnings.filterwarnings('ignore')
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
#===============================#
# INPUT PARAMETERS #
num_eval=15000 

init="May"
targ="MJJ"

y=sys.argv[1] # Grid point x index
x=sys.argv[2] # Grid point y index
#===============================#
# OUTPUT FILES #
sol_data = pd.DataFrame(columns=['CV','Test','Sol'])
indiv_file = 'opt_{0}{1}_past2k_{2}_{3}.csv'.format(init,targ,str(y).zfill(2),str(x).zfill(2))
solution_file = 'opt_{0}{1}_past2k_{2}_{3}_sol.csv'.format(init,targ,init,str(y).zfill(2),str(x).zfill(2))
sol_data.to_csv(indiv_file,sep=' ',header=sol_data.columns,index=None)
#===============================#
# PREDICTORS - past2k #
pred_dataframe = pd.read_csv('Predictors_dataset_past2k_weekly.csv', index_col=0)
#pred_dataframe=pred_dataframe.drop(columns=['dataCO2])
#===============================#
# TARGET DATA #
# Number of HW days per month in past2k period, threshold = 90th percentile of 8821-8850
dataset=Dataset("past2k_tasmax_HWs_EUR_MJJA_period70018850_clim88218850.nc",'r')
NDQ90=dataset['NDQ90_May'][:,0,y,x]+dataset['NDQ90_Jun'][:,0,y,x]+dataset['NDQ90_Jul'][:,0,y,x]

target_dates_past2k=[] # dummy date for target

train_years_past2k=range(7001,8851,1)

for year in train_years_past2k:
    target_dates_past2k.append(str(year).zfill(4)+"-04-30")

target_dates=target_dates_past2k

df_NDQ90=pd.DataFrame(NDQ90,columns=['NDQ90'])
df_NDQ90.index = target_dates
target_dataset=df_NDQ90

first_train = "7002-04-30" # training period 1600 years #
last_train = "8600-04-30" # test period 250 years

first_train_index=int(np.argwhere(df_NDQ90.index==first_train))
last_train_index=int(np.argwhere(df_NDQ90.index==last_train))

#===============================#
# OPTIMISATION #
class ml_prediction(AbsObjectiveFunc):

#    This is the constructor of the class, here is where the objective function can be setted up.
#    In this case we will only add the size of the vector as a parameter.

    def __init__(self, size):
        self.size = size
        self.opt = "min" # it can be "max" or "min"

        # self.sup_lim = np.full(size, 6301) # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim = np.full(size, 1) # array where each component indicates the minimum value of the component of the vector

        # self.sup_lim = np.repeat(30, 35)  # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim =np.repeat(1, 35) # array where each component indicates the minimum value of the component of the vector


        # self.sup_lim = np.append(np.repeat(60, 35),np.repeat(180, 35))  # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim = np.append(np.repeat(0, 35),np.repeat(0, 35)) # array where each component indicates the minimum value of the component of the vector

        self.sup_lim = np.append(np.append(np.repeat(8, pred_dataframe.shape[1]),np.repeat(26, pred_dataframe.shape[1])),np.repeat(1, pred_dataframe.shape[1]))  # array where each component indicates the maximum value of the component of the vector
        self.inf_lim = np.append(np.append(np.repeat(1, pred_dataframe.shape[1]),np.repeat(0, pred_dataframe.shape[1])),np.repeat(0, pred_dataframe.shape[1])) # array where each component indicates the minimum value of the component of the vector
        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)

    def objective(self, solution):
        # print(solution)
        # Read data
        sol_file = pd.read_csv(indiv_file,sep=' ',header=0)
        # pred_dataframe_opt = pd.read_csv('./Predictors/pred_dataframe_lowclusters_SeasonalSmooth.csv', index_col=0)
        # pred_dataframe_opt.index = pd.to_datetime(pred_dataframe_opt.index)
        # target_dataset_opt = pd.read_csv('./Predictors/target_dataset_2.csv', index_col=0)
        # target_dataset_opt.index = pd.to_datetime(target_dataset_opt.index)


        # train_indices_opt = np.array(pd.read_csv('./Predictors/train_indices_opt.csv'))
        # val_indices_opt = np.array(pd.read_csv('./Predictors/val_indices_opt.csv'))
        # train_indices_opt = np.array(pd.read_csv('./Predictors/train_indices_2.csv')).reshape(-1)
        # val_indices_opt = np.array(pd.read_csv('./Predictors/val_indices_2.csv')).reshape(-1)
        # test_indices = np.array(pd.read_csv('./Predictors/test_indices_2.csv')).reshape(-1)
        # train_indices_opt = np.array(pd.read_csv('./Predictors/train_indices_opt__.csv'))
        # test_ind = np.array(pd.read_csv('./Predictors/test_indices__.csv'))


        # Read solution
        time_sequences = np.array(solution[:pred_dataframe.shape[1]]).astype(int)
        # time_sequences = np.repeat(7,pred_dataframe.shape[1])
        # time_lags = np.repeat(30,pred_dataframe_opt.shape[1])

        time_lags = np.array(solution[pred_dataframe.shape[1]:2*pred_dataframe.shape[1]]).astype(int)
        # time_sequences = np.repeat(180,pred_dataframe_opt.shape[1])
        variable_selection = np.array(solution[2*pred_dataframe.shape[1]:]).astype(int)
        # # print(solution)
        if sum(variable_selection) == 0:
            return 100000


        # # Create dataset according to solution
        dataset_opt = target_dataset.copy()
        for i,col in enumerate(pred_dataframe.columns):
            if variable_selection[i] == 0 or time_sequences[i] == 0:
                continue
            for j in range(time_sequences[i]):
                dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = pred_dataframe[col].shift(time_lags[i]+j)
            # dataset_opt[str(col)+'av'] = np.average(dataset_opt[[str(col)+'_lag'+str(time_lags[i]+j) for j in range(time_sequences[i])]], axis=1)
            # dataset_opt.drop([str(col)+'_lag'+str(time_lags[i]+j) for j in range(time_sequences[i])], axis=1, inplace=True)
        
        
        # dataset_opt = pd.read_pickle('./Predictors/dataset_opt.pkl')
        # dataset_opt = dataset_opt.iloc[:, np.append(0,solution.astype(int))]
            
        # Split dataset into train and test

        train_dataset = dataset_opt[first_train_index:last_train_index]
        test_dataset = dataset_opt[last_train_index:]

        # Standardize data

        Y_column = 'NDQ90' 
            
        X_train=train_dataset[train_dataset.columns.drop([Y_column]) ]
        Y_train=train_dataset[Y_column]

        X_test=test_dataset[test_dataset.columns.drop([Y_column]) ]
        Y_test=test_dataset[Y_column]
            
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_std_train = scaler.fit(X_train)

        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test)

        X_train=pd.DataFrame(X_std_train,columns=X_train.columns,index=X_train.index)
        X_test=pd.DataFrame(X_std_test,columns=X_test.columns,index=X_test.index)
        #print (X_train)
    
        # Train model
        from sklearn.metrics import f1_score, mean_absolute_error
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()

        # NORMALISED RMSE OF CROSS-VALIDATION AND TEST
        from sklearn.model_selection import cross_val_score
        iav=np.std(Y_train) # inter-annual variablity of NDQ90
        score = np.abs(cross_val_score(clf, X_train, Y_train, cv=5,scoring='neg_root_mean_squared_error'))/iav

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        # Round to nesrest integer , negative values to 0 #
        print(score.mean(), root_mean_squared_error(Y_pred,Y_test)/iav)

        sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [score.mean()], 'Test': [root_mean_squared_error(Y_pred,Y_test)/iav], 'Sol': [solution]})], ignore_index=True)
        sol_file.to_csv(indiv_file,sep=' ',header=sol_file.columns,index=None)
           
        return score.mean()
    


    def random_solution(self):
        return np.random.choice(self.sup_lim[0], self.size, replace=True)

    def repair_solution(self, solution):

        # unique = np.unique(solution)
        # if len(unique) < len(solution):
        #     pool = np.setdiff1d(np.arange(self.inf_lim[0], self.sup_lim[0]), unique)
        #     new = np.random.choice(pool, len(solution) - len(unique), replace=False)
        #     solution = np.concatenate((unique, new))
        return np.clip(solution, self.inf_lim, self.sup_lim)
objfunc = ml_prediction(3*pred_dataframe.shape[1])

params = {
    "popSize": 100,
    "rho": 0.6,
    "Fb": 0.98,
    "Fd": 0.2,
    "Pd": 0.8,
    "k": 3,
    "K": 20,
    "group_subs": True,

    "stop_cond": "Neval",
    "time_limit": 4000.0,
    "Ngen": 10000,
    "Neval": num_eval,
    "fit_target": 1000, 

    "verbose": True,
    "v_timer": 1,
    "Njobs": 1,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 10,
    "prob_amp": 0.01
}

operators = [
    SubstrateInt("BLXalpha", {"F":0.8}),
    SubstrateInt("Multipoint"),
    SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
    SubstrateInt("Xor"),
]

cro_alg = CRO_SL(objfunc, operators, params)

solution, obj_value = cro_alg.optimize()

solution.tofile(solution_file, sep=',')