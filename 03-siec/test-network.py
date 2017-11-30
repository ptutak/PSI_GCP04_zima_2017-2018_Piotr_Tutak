# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:38:17 2017

@author: PiotrTutak
"""
import keras

import numpy as np



dataSet=np.loadtxt('test_data.csv',delimiter=',')
inputData = dataSet[:,0:2]
expected = dataSet[:,2]

model=keras.models.load_model("model_sieci-[25, 5, 1]-lr-0.01-decay-0.0.h5")
model2=keras.models.load_model("model_sieci-[5, 20, 5, 1]-lr-0.01-decay-0.0.h5")

with open('test-results.txt','w') as f:
    print("model_sieci-[25, 5, 1]-lr-0.01-decay-0.0.h5",file=f)
    print(*list(zip(model.predict(inputData,batch_size=20),expected)),sep='\n',file=f)
    print("loss:{0}, accuracy:{1}".format(*model.evaluate(inputData,expected,batch_size=20)),file=f)
    print("\n",file=f)
    print("model_sieci-[5, 20, 5, 1]-lr-0.01-decay-0.0.h5",file=f)
    print(*list(zip(model2.predict(inputData,batch_size=20),expected)),sep='\n',file=f)
    print("loss:{0}, accuracy:{1}".format(*model2.evaluate(inputData,expected,batch_size=20)),file=f)
