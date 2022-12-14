# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:07:38 2022

@author: vegard
"""
import matplotlib as plt
import numpy as np
import os
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from keras import backend as K


#finds how much a stock moves up or down in percentage
def movementPercentage(p1, p2):
    return ((p2 - p1)/p1)*100


# normalize data between 0 and 1
def normalize(data, train_split):
    minimum = np.min(data[:train_split])
    maximum = np.max(data[:train_split])
    return (data-minimum)/(maximum - minimum)

def normalize2(data, minimum, maximum):
    return (data-minimum)/(maximum - minimum)

# standardize data between -1 and 1
def standardize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

def standardize2(data, mean, std):
    return (data - mean) / std
    
def get_windows(x, window_size):
    windows = []
    for window in x.rolling(window=window_size):
        if(len(window) != window_size):
            continue
        windows.append(window.iloc[:, [1,2,3,4,5,8]].values)
    if(len(windows) != 0):
        windows.pop(-1)
    return windows

def get_labels(x, window_size):
    labels = []
    for window in x.rolling(window=window_size):
        if(len(window) != window_size):
            continue
        labels.append(window.values[-1, -1])
    return labels

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


def getTweetsPreprocessed(path, nof_stocks):
    filenames = os.listdir(path)
    tweets = []
    for f in filenames[:nof_stocks]:
        t_dir = os.path.join(path, f+'/')
        tweet_files = os.listdir(t_dir)
        for file in tweet_files:
            for line in open(t_dir + file, 'r'):
                tweet = json.loads(line)
                tweets.append({'text': tweet['text']})
                tweets[-1]['Date'] = file
                tweets[-1]['Name'] = os.path.splitext(f)[0]
    df_t_apple = pd.DataFrame(tweets)
    df_t_grouped = df_t_apple.groupby(['Name', 'Date'])['text'].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
    return df_t_grouped


def getTweetsRaw(path, nof_stocks):
    filenames = os.listdir(path)
    tweets = []
    for f in filenames[:nof_stocks]:
        t_dir = os.path.join(path, f+'/')
        tweet_files = os.listdir(t_dir)
        for file in tweet_files:
            for line in open(t_dir + file, 'r'):
                tweet = json.loads(line)
                tweets.append({'text': tweet['text']})
                tweets[-1]['Date'] = file
                tweets[-1]['Name'] = os.path.splitext(f)[0]
    df_t_apple = pd.DataFrame(tweets)
    df_t_grouped = df_t_apple.groupby(['Name', 'Date'])['text'].apply(lambda x: [lst for lst in x]).reset_index()
    return df_t_grouped

def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def join_text(x):
    if(not isinstance(x, list)):
        if math.isnan(x):
            return ''
    return ' '.join(x)

#source https://keras.io/examples/timeseries/timeseries_weather_forecasting/
def show_heatmap(data):
    print(data.corr())
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()



