#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import pickle

import os
import numpy as np
np.random.seed(1337)  # for reproducibility
from scipy import spatial
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.layers import Input,Masking, Dense, Dropout, Activation, Flatten, Merge, Convolution1D, MaxPooling1D, LSTM, merge,  Reshape, Lambda, AveragePooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping
import speech_feature
import theano.tensor as T
from keras.engine.topology import Layer, InputSpec

#list of paremeter 
nb_filters = 50
filters_size = 5
sentene_word_num = 60
document_word_num = 620
word_vector_dim = 50
batch_size = 20
nb_epoch = 5


[mfcc_feats, anslist, count]=speech_feature.run()

train_set	=	mfcc_feats[0:len(mfcc_feats)*8/10]
dev_set		=	mfcc_feats[len(mfcc_feats)*8/10:len(mfcc_feats)*9/10]
test_set	=	mfcc_feats[len(mfcc_feats)*9/10:]
train_ans	=	anslist[0:len(anslist)*8/10]
dev_ans		=	anslist[len(mfcc_feats)*8/10:len(mfcc_feats)*9/10]
test_ans	=	anslist[len(mfcc_feats)*9/10:]

print(train_set[0])
print(train_set.shape)
model = Sequential()
model.add(Masking(mask_value=0.0,input_shape=(424,13)))

model.add(LSTM(13))
model.add(Dense(13))
model.add(Dense(1))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()
model.fit(train_set,train_ans,batch_size=1,nb_epoch=3,validation_data=(dev_set,dev_ans),shuffle=True)
output=model.predict(test_set)
for ans in output:
	print (ans)