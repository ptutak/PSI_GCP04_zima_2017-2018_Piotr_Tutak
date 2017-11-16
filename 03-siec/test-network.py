# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:38:17 2017

@author: PiotrTutak
"""
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import numpy as np
import sys


dataSet=np.loadtxt('training_data.csv',delimiter=',')
inputData = dataSet[:,0:2]
expected = dataSet[:,2]

