import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd                                                 # Importing package pandas (For Panel Data Analysis)
import numpy as np                                                  # Importing package numpys (For Numerical Python)
import matplotlib.pyplot as plt                                     # Importing pyplot interface to use matplotlib
import scipy as sp 
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None  
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
# for data pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import*
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline


# for prediction (machine learning models) ------------------------

from sklearn.linear_model import*
from sklearn.preprocessing import*
from sklearn.ensemble import*
from sklearn.neighbors import*
from sklearn import svm
from sklearn.naive_bayes import*
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv('avocado.csv')
data=data.drop(['Unnamed: 0'], axis=1)
X = data[['4046', '4225', '4770', 'type','region']]
Y = data['AveragePrice']
y = np.log1p(Y)
X = pd.get_dummies(X, prefix=["type","region"], columns=["type","region"], drop_first = True)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 99)
lr = LinearRegression()
lr.fit(X_train,y_train)
region_list = data['region'].unique()
type_list = data['type'].unique()

def predict(udf, selected_region,selected_type):
  udf['region_'+ selected_region][0] = 1
  udf['type_' + selected_type ][0] = 1
  ans = (np.exp(lr.predict(udf))-1)
  return ans

#streamlit   
   
st.header('Price Prediction')
lottie_coding = load_lottiefile("avocado.json")
st_lottie(
    lottie_coding,
    speed=1.5,
    reverse=False,
    loop=True,
    quality="low",height=220
)
selected_region = st.selectbox(
    "Select a region from the dropdown",
    region_list
)
selected_type = st.selectbox(
    "Select a Type from the dropdown",
    type_list
)
if st.button('Show Price'):
  dict_map = {}
  for idx, key in enumerate(X_train.columns):
    dict_map[key] = idx
  user_data = {key:[0] for key in X_train.columns}
  udf=pd.DataFrame(user_data)
  ans = predict(udf, selected_region, selected_type)
  st.write(ans)
st.write('  '
         )
st.write(' ')
