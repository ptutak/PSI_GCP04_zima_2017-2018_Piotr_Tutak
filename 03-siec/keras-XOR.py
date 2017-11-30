# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:13:28 2017

@author: PiotrTutak
"""

import numpy as np
import keras as ks



trainingData=np.array([np.array([0,0]),
              np.array([0,1]),
              np.array([1,0]),
              np.array([1,1])])

expected=[0,1,1,0]

model=ks.models.Sequential()

model.add(ks.layers.Dense(2,input_dim=2,activation='sigmoid'))
model.add(ks.layers.Dense(1,activation='hard_sigmoid'))

adam = ks.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


model.compile(loss='mean_squared_error',optimizer=adam,metrics=['accuracy'])

model.fit(trainingData,expected,epochs=1000,batch_size=1)
scores=model.evaluate(trainingData,expected)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))