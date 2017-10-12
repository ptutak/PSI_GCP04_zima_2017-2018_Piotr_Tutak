# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:39:26 2017

@author: PiotrTutak
"""

import numpy as np

class SigmFact:
    def __call__(self,alfa):
        def sigm(s):
            return 1/(1+np.exp(-alfa*s))
        return sigm
    def derivative(self,alfa):
        def sigmDeriv(s):
            return alfa*np.exp(-alfa*s)/((1+np.exp(-alfa*s))**2)
        return sigmDeriv


class Perceptron:
    def __init__(self, weights, learnRate, activFunc, activFuncDeriv):
        self._weights=np.array(weights)
        self._learnRate=learnRate
        self._activFunc=activFunc
        self._activFuncDeriv=activFuncDeriv
    def learn(self,inputValues,expectedValue,learnRate=False):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        if (learnRate):
            self._learnRate=learnRate
        s=np.dot(self._weights,inputValues)
        s=self._activFunc(s)
        if s!=expectedValue:
            for i in range(len(self._weights)):
                self._weights[i]+=self._learnRate*(expectedValue-s)*self._activFuncDeriv(s)*inputValues[i]
    def process(self,inputValues):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        s=np.dot(self._weights,inputValues)
        s+=self._bias
        return self._activFunc(s)
    def __getitem__(self,index):
        return self._weights[index]
    def __setitem__(self,index,value):
        self._weights[index]=value
    def __iter__(self):
        return iter(self._weights)
    def __len__(self):
        return len(self._weights)
    def __repr__(self):
        return 'Perceptron(weights:{0!r},learnRate:({1!r}),activFunc:{2!r})'.format(list(self._weights),self._learnRate,self._activFunc)

