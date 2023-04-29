import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def loadData() -> pd.DataFrame:
    """
    load dataset
    
    :return: returns the DataFrame with the name "new_data"
    """
    new_data = pd.read_csv("./new_data.csv")
    return new_data

def dataSplit() -> tuple:
    """
    Split data of train and test

    :return: returns a tuple with training and test features and test target variables
    """
    new_data = loadData()
    data_train, data_test = train_test_split(new_data, test_size=0.2)
    x_train = data_train.drop(['Total_Unemployment_in_State_Area'], axis=1)
    x_test = data_test.drop(['Total_Unemployment_in_State_Area'], axis=1)
    y_train = data_train[["Total_Unemployment_in_State_Area"]]
    y_test = data_test[["Total_Unemployment_in_State_Area"]]
    
    return  x_train, x_test, y_train, y_test 

def scaleData(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Use StandardScaler to scale target variables
    
    :return: returns the scaled training and test data
    """
    
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    
    return y_train_scaled, y_test_scaled

def pca(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    USe Principal Component Analysis (PCA) on the target variables
    
    :return: returns the with PCA transformed target variables 
    """
    pca = PCA(n_components=1)
    pca.fit(y_train)
    y_train_pca = pca.transform(y_train)
    y_test_pca = pca.transform(y_test)
    return y_train_pca, y_test_pca

def define_eval_metrics(y_test: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Function to define evaluation metrics (MAE, RMSE, R2)

    :return: returns the values of mae, rmse and r2
    """
    
    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    r2 = r2_score(y_test, predicted)
    return  [mae, rmse, r2]

# List of metrics to find the best output
metrics = [1, 2, 3, 4, 5, 6, 7, 8, 9]

def start_Regressor():
    """
    Starts a RandomForestRegressor model with different evaluation metrics.
    """
    for metric in metrics:
    
        with mlflow.start_run():

            x_train, x_test, y_train, y_test = dataSplit()

            y_train_scaled, y_test_scaled = scaleData(y_train, y_test)

            y_train_pca, y_test_pca = pca(y_train_scaled, y_test_scaled)

            rf = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=metric)
            rf.fit(x_train, y_train_pca)

            predicted = rf.predict(x_test)

            (mae, rmse, r2) = define_eval_metrics(y_test_pca, predicted)
        
            print("RandomForestRegressor model (metric={:f}):".format(metric))
            print("  MAE: %s" % mae)
            print("  RMSE: %s" % rmse)
            print("  R2: %s" % r2)

            mlflow.log_param("metric", metric)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2) 
            mlflow.sklearn.log_model(rf, "model")

#start_Regressor()

