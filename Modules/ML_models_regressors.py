#LR
from sklearn.linear_model import LinearRegression
clf_LR = LinearRegression()
#SVR
from sklearn.svm import SVR
clf_SVR = SVR()
#DT
from sklearn.tree import DecisionTreeRegressor
clf_DT = DecisionTreeRegressor(max_depth=400, random_state=0)
#RF
from sklearn.ensemble import RandomForestRegressor
clf_RF = RandomForestRegressor(max_depth=400, random_state=0)
#KNeighbors
from sklearn.neighbors import KNeighborsRegressor
clf_neigh = KNeighborsRegressor(n_neighbors=20,algorithm='brute')
#AdaBoost
from sklearn.ensemble import AdaBoostRegressor
clf_AB = AdaBoostRegressor(n_estimators=50, random_state=0)
#MLP
from sklearn.neural_network import MLPRegressor
clf_MLP = MLPRegressor(hidden_layer_sizes=500,random_state=1, max_iter=300)
#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
clf_GB = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
#Light Gradient Boosting
from lightgbm import LGBMRegressor
clf_LGBM = LGBMRegressor()