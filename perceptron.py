# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:39:26 2017

@author: PiotrTutak
"""

import numpy as np

class perceptron:
    def __init__(self,weights, bias, activationFunction):
        self._weights=np.array(weights)
        self._bias=bias
        self.activationFunc=activationFunction
    def learn(self,inputValues,expectedValues):
        s=0
        if len(inputValues)!=len(self._weights) or len(inputValues)!=len(expectedValues):
            raise TypeError('Wrong values length')
        for i in range(len(self._weights)):
            
        
