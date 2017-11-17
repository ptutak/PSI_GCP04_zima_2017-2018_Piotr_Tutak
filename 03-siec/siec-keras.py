# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:57:07 2017

@author: Piotrek
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import numpy as np
import sys

"""
Rastrigin
"""

def rastrigin(x,y):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

np.random.seed(7)

"""
Generowanie danych uczących i testujących
"""

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


lr=0.1
decay=0.0
layers=[30,1]


"""
Przekierowanie wyjscia
"""
STDOUT=sys.stdout
try:
    f=open('results'+str(layers)+'-lr-'+str(lr)+'-decay-'+str(decay)+'.txt','w');
    sys.stdout=f
    
    np.random.seed(7)
    
    dataSet=np.loadtxt('training_data.csv',delimiter=',')
    inputData = dataSet[:,0:2]
    expected = dataSet[:,2]
    
    testDataSet=np.loadtxt('test_data.csv',delimiter=',')
    inputTestData = testDataSet[:,0:2]
    expectedTestData = testDataSet[:,2]
   
    """
    Dodawanie warstw do modelu
    """
    model=Sequential()
    for i in range(len(layers)):
        if i==0:
            model.add(Dense(layers[i], input_dim=2,activation='sigmoid'))
        elif i==len(layers)-1:
            model.add(Dense(layers[i],activation='linear'))
        else:
            model.add(Dense(layers[i],activation='sigmoid'))
    
    """
    Optymalizator
    """
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    
    """
    Trenowanie modelu
    """
    model.fit(inputData,expected,epochs=100000,batch_size=20)
    
    scores=model.evaluate(inputTestData,expectedTestData)
    print("scores: ",scores)
    print("metrics_names:",model.metrics_names)
    
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    print(model.summary())
    """
    zapis modelu
    """
    model.save('model_sieci-'+str(layers)+'-lr-'+str(lr)+'-decay-'+str(decay)+'.h5')
finally:
    sys.stdout=STDOUT
    f.close()
