# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:57:07 2017

@author: Piotrek
"""



from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def rastrigin(x,y):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

np.random.seed(7)

with open('training_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

np.random.seed(7)

dataset=np.loadtxt('training_data.csv',delimiter=',')
inputData = dataset[:,0:2]
expected = dataset[:,2]

print(inputData)
print(expected)