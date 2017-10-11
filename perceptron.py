# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:39:26 2017

@author: PiotrTutak
"""

import numpy as np

class perceptron:
    def __init__(self,weights, bias, learnRate, activFunc):
        self._weights=np.array(weights)
        self._bias=bias
        self._learnRate=learnRate
        self._activFunc=activFunc
    def learn(self,inputValues,expectedValue):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        s=np.dot(self._weights,inputValues)
        s-=self._bias
        s=self._activFunc(s)
        if s!=expectedValue:
            for i in range(len(self._weights)):
                self._weights[i]+=self._learnRate*(expectedValue-s)*inputValues[i]
            self._bias+=self._learnRate*(expectedValue-s)

