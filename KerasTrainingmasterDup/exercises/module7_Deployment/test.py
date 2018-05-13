
import keras
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

# training_data_df = pd.read_csv('./data/sales_data_training1.csv')
sp500 = pd.read_csv('./data/sales_data_testing.csv',index_col='critic_rating',usecols=['critic_rating','is_action','is_exclusive_to_us','is_portable','is_role_playing','is_sequel','is_sports','suitable_for_kids','total_earnings','unit_price'])

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_training= scaler.fit_transform(training_data_df)
# scaled_training_df = pd.DataFrame(scaled_training,columns= training_data_df.columns.values)
# scaled_training_df.to_csv('./data/sales_training_scaled.csv', index=False)