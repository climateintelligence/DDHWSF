import numpy as np
import pandas as pd
from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error

class optimisation(AbsObjectiveFunc):
    """
    Class for performing machine learning prediction as an objective function
    for optimization using the CRO-SL algorithm.
    """
 
#    This is the constructor of the class, here is where the objective function can be setted up.
#    In this case we will only add the size of the vector as a parameter.

    def __init__(self, size, pred_dataframe, target_dataset, first_train_index, last_train_index, indiv_file):
        self.size = size
        self.opt = "min" # it can be "max" or "min"

        # Store necessary data references for prediction
        self.pred_dataframe = pred_dataframe
        self.target_dataset = target_dataset
        self.first_train_index = first_train_index
        self.last_train_index = last_train_index
        self.indiv_file = indiv_file
        
        # self.sup_lim = np.full(size, 6301) # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim = np.full(size, 1) # array where each component indicates the minimum value of the component of the vector

        # self.sup_lim = np.repeat(30, 35)  # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim =np.repeat(1, 35) # array where each component indicates the minimum value of the component of the vector


        # self.sup_lim = np.append(np.repeat(60, 35),np.repeat(180, 35))  # array where each component indicates the maximum value of the component of the vector
        # self.inf_lim = np.append(np.repeat(0, 35),np.repeat(0, 35)) # array where each component indicates the minimum value of the component of the vector

        self.sup_lim = np.append(np.append(np.repeat(8, self.pred_dataframe.shape[1]),np.repeat(26, self.pred_dataframe.shape[1])),np.repeat(1, self.pred_dataframe.shape[1]))  # array where each component indicates the maximum value of the component of the vector
        self.inf_lim = np.append(np.append(np.repeat(1, self.pred_dataframe.shape[1]),np.repeat(0, self.pred_dataframe.shape[1])),np.repeat(0, self.pred_dataframe.shape[1])) # array where each component indicates the minimum value of the component of the vector
        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)

    def objective(self, solution):
        # print(solution)
        # Read data
        sol_file = pd.read_csv(self.indiv_file,sep=' ',header=0)
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
        time_sequences = np.array(solution[:self.pred_dataframe.shape[1]]).astype(int)
        # time_sequences = np.repeat(7,pred_dataframe.shape[1])
        # time_lags = np.repeat(30,pred_dataframe_opt.shape[1])

        time_lags = np.array(solution[self.pred_dataframe.shape[1]:2*self.pred_dataframe.shape[1]]).astype(int)
        # time_sequences = np.repeat(180,pred_dataframe_opt.shape[1])
        variable_selection = np.array(solution[2*self.pred_dataframe.shape[1]:]).astype(int)
        # # print(solution)
        if sum(variable_selection) == 0:
            return 100000


        # # Create dataset according to solution
        dataset_opt = self.target_dataset.copy()
        for i,col in enumerate(self.pred_dataframe.columns):
            if variable_selection[i] == 0 or time_sequences[i] == 0:
                continue
            for j in range(time_sequences[i]):
                dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = self.pred_dataframe[col].shift(time_lags[i]+j)
            # dataset_opt[str(col)+'av'] = np.average(dataset_opt[[str(col)+'_lag'+str(time_lags[i]+j) for j in range(time_sequences[i])]], axis=1)
            # dataset_opt.drop([str(col)+'_lag'+str(time_lags[i]+j) for j in range(time_sequences[i])], axis=1, inplace=True)
        
        
        # dataset_opt = pd.read_pickle('./Predictors/dataset_opt.pkl')
        # dataset_opt = dataset_opt.iloc[:, np.append(0,solution.astype(int))]
            
        # Split dataset into train and test

        train_dataset = dataset_opt[self.first_train_index:self.last_train_index]
        test_dataset = dataset_opt[self.last_train_index:]

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
        sol_file.to_csv(self.indiv_file,sep=' ',header=sol_file.columns,index=None)
           
        return score.mean()

    def random_solution(self):
        """Generate a random solution within the defined limits."""
        return np.random.choice(self.sup_lim[0], self.size, replace=True)

    def repair_solution(self, solution):
        """Ensure solution values stay within bounds."""
        return np.clip(solution, self.inf_lim, self.sup_lim)