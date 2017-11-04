# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:57:07 2017

@author: Piotrek
"""



from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import numpy as np

def rastrigin(x,y):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

np.random.seed(7)

with open('training_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

with open('test_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

np.random.seed(7)

dataSet=np.loadtxt('training_data.csv',delimiter=',')
inputData = dataSet[:,0:2]
expected = dataSet[:,2]

testDataSet=np.loadtxt('test_data.csv',delimiter=',')
inputTestData = testDataSet[:,0:2]
expectedTestData = testDataSet[:,2]

model=Sequential()
model.add(Dense(20, input_dim=2, activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(1,activation='linear'))


adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

model.fit(inputData,expected,epochs=100000,batch_size=20)

scores=model.evaluate(inputTestData,expectedTestData)

print(scores)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))