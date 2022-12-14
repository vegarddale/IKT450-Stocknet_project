# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:02:54 2022

@author: vegard
"""

from keras.layers import Dense,RepeatVector,Lambda,multiply,Activation,Input, LSTM, GRU, Dropout, Flatten,BatchNormalization, Permute
from keras.models import Model
from keras.optimizers import Adam

def get_weak_learner(input_shape,n):
    X_input = Input(shape=(input_shape[0], (input_shape[1])))
    X = GRU(n,return_sequences=False)(X_input)
    outputs = Dense(1, activation='sigmoid')(X)
    optim = Adam(lr=0.001)
    model = Model(inputs = X_input, outputs = outputs)
    model.compile(loss='mse',optimizer=optim, metrics=['mse'])
    return model