# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:39:26 2017

@author: PiotrTutak
"""

import numpy as np

def ident(x):
    return x

def one(x):
    return 1

def signAbs(x):
    if x<0:
        return 0
    else:
        return 1

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
        self._error=expectedValue-s
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*(expectedValue-s)*inputValues[i]
    def process(self,inputValues):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        s=np.dot(self._weights,inputValues)
        self._val=self._activFunc(s)
        return self._val
    """
    def processError(self,inputValues,expectedValue,learnRate=False):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        if learnRate:
            self._learnRate=learnRate
        s=np.dot(self._weights,inputValues)
        self._val=self._activFunc(s)
        self._error=(expectedValue-self._val)*self._activFuncDeriv(self._val)
        return self._val
    """
    def propagateError(self,weights,errors,learnRate=False):
        if len(errors)!=len(weights):
            raise TypeError('Wrong values length')
        if learnRate:
            self._learnRate=learnRate
        self._error=self._activFuncDeriv(self._val)*np.dot(weights,errors)
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*self._error*self._value
        return self._error
    def __getitem__(self,index):
        if index=='error':
            return self._error
        return self._weights[index]
    def __setitem__(self,index,value):
        if index=='error':
            self._error=value
        elif index=='activFunc':
            self._activFunc=value
        elif index=='activFuncDeriv':
            self._activFuncDeriv=value
        else:
            self._weights[index]=value
    def __getattr__(self,attr):
        raise AttributeError('No such attribute %r'%attr)
    def __iter__(self):
        return iter(self._weights)
    def __len__(self):
        return len(self._weights)
    def __repr__(self):
        return 'Perceptron(weights:{0!r},learnRate:({1!r}),activFunc:{2!r})'.format(list(self._weights),self._learnRate,self._activFunc)


if __name__=='__main__':
        
    inputData=(
            ((1,0,0),0),
            ((1,0,1),0),
            ((1,1,0),0),
            ((1,1,1),1)
            )
    
    learnRates=[np.random.ranf() for _ in range(10)]
    
    p=Perceptron([np.random.ranf() for _ in range(3)],0.1,signAbs,one)
    
    for _ in range(100):
        p.learn(*inputData[np.random.choice(len(inputData))])
        
    print(p)
    
    for x in inputData:
        print(p.process(x[0]))