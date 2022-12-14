# # # -*- coding: utf-8 -*-
# # """
# # Created on Fri Nov  4 14:26:26 2022

# # @author: vegard
# # """

import pandas as pd
import os
import numpy as np
import tensorflow as tf
import keras
from utils import *
from regression_classifier import get_regression_classifier
from weak_learner import get_weak_learner
from keras.layers import Dense,RepeatVector,Lambda,multiply,Activation,Input, LSTM, GRU, Dropout, Flatten,BatchNormalization, Permute
from keras.models import Model
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import shuffle
from keras import backend as K

# # =============================================================================
# # Pre Processing
# # =============================================================================

nof_stocks = 88
root = "./stocknet-dataset-master"
price_dir = os.path.join(root, 'price/raw')
price_files = os.listdir(price_dir)
price_files = [file for file in price_files if file != "BRK-A.csv"] # skip the BRK-A stock  
dfs = []
for f in price_files[:nof_stocks]: 
    df = pd.read_csv(os.path.join(price_dir, f))
    df['Name'] = os.path.splitext(f)[0]
    dfs.append(df)
df_stocks_p = pd.concat(dfs, ignore_index=True)

tweets_dir = os.path.join(root, 'tweet/preprocessed/')  

# df_tweets = getTweetsRaw(tweets_dir, nof_stocks)
df_tweets = getTweetsPreprocessed(tweets_dir, nof_stocks)

df_stocks = pd.merge(df_stocks_p, df_tweets, on=['Name', 'Date'], how='outer')
df_stocks = df_stocks.dropna(subset=['Open'])

df_stocks['Diff'] = df_stocks['Adj Close'].rolling(window=2).apply(lambda x: movementPercentage(x.values[0], x.values[1]))
#filter data where difference between adjusted closing price day d and d-1 > 0.5% or < -0.5%
df_stocks = df_stocks.loc[(df_stocks['Diff'] < - 0.5) | (df_stocks['Diff'] > 0.5)]

df_stocks.loc[df_stocks['Diff'] < -0.5, 'Label'] = 0
df_stocks.loc[df_stocks['Diff'] > 0.5, 'Label'] = 1

df_stocks['text'] = df_stocks['text'].apply(lambda x: join_text(x))
# apply vader sentiment analysis to tweets
df_stocks['text'] = df_stocks['text'].apply(sentiment_score)
# # use only compound score from sentiment analyzis
df_stocks['text'] = df_stocks['text'].apply(lambda x: x['compound'])


# count number of true or false labels
print(df_stocks.groupby(['Label']).size())

# split into training and testing
training = df_stocks.groupby('Name', as_index = False).apply(lambda x: x[:int(len(x)*0.8)])
testing = df_stocks.groupby('Name', as_index = False).apply(lambda x: x[int(len(x)*0.8):])

price = training.iloc[:,1:6]
# show heatmap of correlation between stock price features
show_heatmap(price)

# normalize
maximum = np.max(np.max(training.iloc[:,1:6]).values)
minimum = np.min(np.min(training.iloc[:,1:6]).values)

mean = (training.iloc[:,1:6]).values.mean(axis=0)
std = (training.iloc[:,1:6]).values.std(axis=0)

training.iloc[:, 1:6] = training.iloc[:, 1:6].apply(lambda x: normalize2(x.values, minimum, maximum))
testing.iloc[:, 1:6] = testing.iloc[:, 1:6].apply(lambda x: normalize2(x.values, minimum, maximum))

window_size = 5
x_train = training.groupby('Name').apply(lambda x: get_windows(x, window_size)).values
x_test = testing.groupby('Name').apply(lambda x: get_windows(x, window_size)).values
# y_train = training.groupby('Name').apply(lambda x: get_labels(x, window_size+1)).values
labels_test = testing.groupby('Name').apply(lambda x: get_labels(x, window_size+1)).values
labels_train = training.groupby('Name').apply(lambda x: get_labels(x, window_size+1)).values

y_train = training.groupby('Name').apply(lambda x: x[window_size:]).values[:,5]
y_test = testing.groupby('Name').apply(lambda x: x[window_size:]).values[:,5]


# # # reshape 
x_train = np.array([tf.convert_to_tensor((val), dtype=np.float64) for sublist in x_train for val in sublist])
x_test = np.array([tf.convert_to_tensor(list(val), dtype=np.float64) for sublist in x_test for val in sublist])
labels_test = np.array([tf.convert_to_tensor([val], dtype=np.float64) for sublist in labels_test for val in sublist])
labels_train = np.array([tf.convert_to_tensor([val], dtype=np.float64) for sublist in labels_train for val in sublist])

y_train = [[i] for i in y_train]
y_train = np.array(y_train, dtype=np.float64)
y_test = np.array(y_test, dtype=np.float64)
x_train, y_train = shuffle(x_train, y_train, random_state=0)
n_features = 6

# =============================================================================
# Regression classifier implementation
# =============================================================================

clf = get_regression_classifier([window_size,n_features],128)

# using custom loss function
clf.fit(x_train, np.append(y_train, x_train[:,:,4], axis=1), epochs=500, batch_size = 32, validation_data=(x_test, y_test))

# using regular mse
# clf.fit(x_train, y_train, epochs=425, batch_size = 32, validation_data=(x_test, y_test))

pred = clf.predict(x_test)


predictions = pd.DataFrame(pred)
pred = []
# append 1 to account for not having a movement prediction on first day
pred.append(1)
for i in predictions.rolling(window=2):
    if(len(i) != 2):
        continue
    if(i.values[0]<i.values[1]):
        pred.append(1)
    else:
        pred.append(0)
acc = 0
for i in range(len(pred)):
    if(labels_test[i] == pred[i]):
        acc += 1
    
print("Accuracy: ", acc/(len(labels_test)-1))

# # =============================================================================
# # AdaBoost GRU Implementation
# # =============================================================================


regressor = get_weak_learner([window_size,n_features],128)

regressor = KerasRegressor(regressor, epochs = 60, batch_size = 32)
rgr = AdaBoostRegressor(base_estimator=regressor, learning_rate=0.001, n_estimators=50)
rgr.fit(x_train, y_train)

print("R2 score: ", rgr.score(x_test, y_test))

predictions = rgr.predict(x_test)

predictions = pd.DataFrame(predictions)

mae = tf.keras.losses.MeanAbsoluteError()

print("MAE: ", mae(predictions, y_test).numpy())

print("RMSE: ", root_mean_squared_error(predictions.values, y_test).numpy())


