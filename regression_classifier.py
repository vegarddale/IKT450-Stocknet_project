# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:00:09 2022

@author: vegard
"""
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Dense,RepeatVector,Lambda,multiply,Activation,Input, LSTM, GRU, Dropout, Flatten,BatchNormalization, Permute
from keras import backend as K
from keras.models import Model

# adds additional penalty if predicting up or down wrong when previous day was up or down
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    labels = y_true[:,0]
    labels = tf.reshape(labels, (-1,1))
    data = y_true[:,1:]
    data = tf.reshape(data, (-1, 5))
    added_loss = 0.0
    loss = mse(labels, y_pred)
    for i in range(len(data)):
        t_val = labels[i,0]
        p_val = 1 if (y_pred[i,0] - t_val) < 0 else 0
        if(data[i,3] - data[i,4] < 0 and data[i,4] - t_val < 0): # up day 4-5 up day 5-6
            if(p_val == 0):
                added_loss += 0.001*loss
        if(data[i,3] - data[i,4] > 0 and data[i,4] - t_val > 0): # down day 4-5 down day 5-6
            if(p_val == 1):
                added_loss += 0.001*loss 
    if(y_true.shape[0] != None):
        added_loss = added_loss/(float(y_true.shape[0])) 
        
    return loss + added_loss

n_features = 6


def get_regression_classifier(input_shape,n):
    X_input = Input(shape=(input_shape[0], (input_shape[1])))
    X = LSTM(n,return_sequences=True)(X_input)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    
    X = LSTM(n,return_sequences=True)(X)
    X = Dropout(0.2)(X)
    X = BatchNormalization()(X)
    X = LSTM(n,return_sequences=False)(X)
    
    # Attention mechanism
    e = Dense(1, activation='tanh')(X)
    e = Flatten()(e)
    a = Activation('softmax')(e)
    tmp=RepeatVector(128)(a)
    tmp=Permute([2, 1])(tmp)
    output = multiply([X, tmp])
    output = Lambda(lambda values: K.sum(values, axis=1))(output)
    X = Dropout(0.2)(output)
    X = BatchNormalization()(X)
    
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.2)(output)
    outputs = Dense(1,activation='sigmoid')(X)
    
    opt = Adam(lr=0.001)
    model = Model(inputs = X_input, outputs = outputs)
    model.compile(loss=custom_loss, optimizer=opt, metrics=['mse'])
    return model