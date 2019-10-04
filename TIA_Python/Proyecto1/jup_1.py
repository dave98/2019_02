from __future__ import division, print_function, unicode_literals
from CategoricalEncoder import CategoricalEncoder
from DataFrameSelector import DataFrameSelector


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from statistics import *

import pandas as pd
import numpy as np
import os


np.random.seed(42)

#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#Direccion para el almacenamiento de las figuras del plotter
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def loading_data(route):
    return pd.read_csv(route + ".csv")

#------------------------------------------------START HERE------------------------------------------------
data_container = loading_data("train")
data_test_container = loading_data("test")

train_set = data_container.drop(["Id", "SalePrice"], axis=1) #->Desechamos el valor a predecir
train_set_labels = data_container["SalePrice"].copy() # Guardando valor a predecir

test_set = data_test_container.drop(["Id"], axis = 1)


train_set_zoning = train_set['MSZoning'] #Esta caracteristica se divide en categorías
train_set_zoning_encoded, train_set_zoning_categories = train_set_zoning.factorize() #Parametrizamos las características.

cat_encoder = CategoricalEncoder() #Encodificador de categorías.
#cat_encoder = CategoricalEncoder(encoding="onehot-dense") #Encodificador de categorías.
train_set_zoning_reshaped = train_set_zoning.values.reshape(-1, 1)
train_set_zoning_1hot = cat_encoder.fit_transform(train_set_zoning_reshaped)
#print(train_set_zoning_1hot)
#print(cat_encoder.categories_) #Extrae todas las categorías dentro del margen seleccionado.

#Pre procesando los datos numéricos
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

train_set_num_set = train_set.select_dtypes(include=[np.number]) #Separanado las caracteristicas compuestas unicamente por numeros
#train_set_num_set["LotFrontage"].fillna(median, inplace=True) #
#sample_train_set_num = train_set_num_set[train_set_num_set.isnull().any(axis=1)]
#num_pipeline = Pipeline([
#            ('imputer', Imputer(strategy="median")),
#            ('std_scaler', StandardScaler()),
#            ])
#train_set_processed_numerical = num_pipeline.fit_transform(train_set_num_set) #Results from numerical atributes

#Pre procesando los datos categóricos
train_Set_atributes = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
#train_set = train_set[:].fillna('Missing') #Replacing Nan categorical values with missing to be considered as a value
train_set[train_Set_atributes] = train_set[train_Set_atributes].fillna('Miss')
test_set[train_Set_atributes] = test_set[train_Set_atributes].fillna('Miss')
#Pipelina para numeros y categorías

num_attribs = list(train_set_num_set) #Obtiene la lista de atributos basados en numeros
#cat_attribs = train_set[train_Set_atributes].fillna('Missing') #Nan values replace with missing
cat_attribs = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]


num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', Imputer(strategy="median")),
            ('std_scaler', StandardScaler()),
            ])

cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('cat_encoder', CategoricalEncoder(encoding="onehot-dense", handle_unknown='ignore')),
])


full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline),
])

#print(train_set.shape)
#print(test_set.shape)

train_set_processed = full_pipeline.fit_transform(train_set) #Prepared_Data
test_set_processed = full_pipeline.transform(test_set)

#print(train_set_processed.shape)
#print(test_set_processed.shape)

#Starting Linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_set_processed, train_set_labels)

some_data = train_set.iloc[:5]
some_labels = train_set_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

#print('Predictions: ', lin_reg.predict(some_data_prepared))
#print('Labels: ', list(some_labels))

#RootMeanSquareErrors ///////////////////////////////////////////////////////////
from sklearn.metrics import mean_squared_error
#set_predictions = lin_reg.predict(train_set_processed)
#lin_mse = mean_squared_error(train_set_labels, set_predictions)
#lin_rmse  = np.sqrt(lin_mse)
#print(lin_rmse)

#AbsoluteMeanError ////////////////////////////////////////////////////////////
#from sklearn.metrics import mean_absolute_error
#lin_mae = mean_absolute_error(train_set_labels, set_predictions)
#print(lin_mae)

#DecsionTreeRegresor /////////////////////////////////////////////////////
#from sklearn.tree import DecisionTreeRegressor

#tree_reg = DecisionTreeRegressor(random_state=42)
#tree_reg.fit(train_set_processed, train_set_labels)

#set_predictions = tree_reg.predict(train_set_processed)
#tree_mse = mean_squared_error(train_set_labels, set_predictions)
#tree_rmse = np.sqrt(tree_mse)
#print(tree_rmse)

#TUNING THE MODEL
#from sklearn.model_selection import cross_val_score
def display_scores(i_scores):
    print("List of Scores: ", i_scores)
    print("Mean: ", i_scores.mean())
    print("Standard: ", i_scores.std())

#Scores for tree_reg in cross_val_score//////////////////////////////////////
#tree_scores = cross_val_score(tree_reg, train_set_processed, train_set_labels, scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-tree_scores)
#display_scores(tree_rmse_scores)

#Scores for lin_reg in cross_val_score//////////////////////////////////////
#lin_scores = cross_val_score(lin_reg, train_set_processed, train_set_labels, scoring="neg_mean_squared_error", cv=10)
#lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores)

#Random Forest Regressor///////////////////////////////////////////////////
#from sklearn.ensemble import RandomForestRegressor
#forest_reg = RandomForestRegressor(random_state=42)
#forest_reg.fit(train_set_processed, train_set_labels)

#set_predictions = forest_reg.predict(train_set_processed)
#forest_mse = mean_squared_error(train_set_labels, set_predictions)
#forest_rmse = np.sqrt(forest_mse)
#print(forest_rmse)

#Scores for forest_reg in cross_val_score//////////////////////////////////
#forest_scores = cross_val_score(forest_reg, train_set_processed, train_set_labels, scoring="neg_mean_squared_error", cv=10)
#forest_rmse_scores = np.sqrt(-lin_scores)
#display_scores(forest_rmse_scores)

#Support Vector Regression
#from sklearn.svm import SVR
#svm_reg = SVR(kernel="linear")
#svm_reg.fit(train_set_processed, train_set_labels)

#set_predictions = svm_reg.predict(train_set_processed)
#svm_mse = mean_squared_error(train_set_labels, set_predictions)
#svm_rmse = np.sqrt(svm_mse)
#print(svm_rmse)


#Including Test Set
lin_reg_2 = LinearRegression()
lin_reg_2.fit(train_set_processed, train_set_labels)

set_predictions = lin_reg_2.predict(test_set_processed)
#print(set_predictions[:20])
A = np.arange(1461, 2920, dtype=np.int16)
records = np.rec.fromarrays((A, set_predictions), names=('Id','SalePrice'))
pd.DataFrame(records).to_csv("Predicted.csv", index=None)

#set_predictions = np.c_[A, set_predictions]
#print(set_predictions[:20])
#pd.DataFrame(set_predictions).to_csv("Predicted.csv", index=None)
