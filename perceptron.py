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

def naturalOne(x):
    if x<=0:
        return 0
    return 1

class Sigm:
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
        self.__dict__['_weights']=np.array(weights)
        self.__dict__['_learnRate']=learnRate
        self.__dict__['_activFunc']=activFunc
        self.__dict__['_activFuncDeriv']=activFuncDeriv
    def learn(self,inputValues,expectedValue,learnRate=False):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        if (learnRate):
            self._learnRate=learnRate
        s=np.dot(self._weights,inputValues)
        s=self._activFunc(s)
        self.__dict__['_error']=expectedValue-s
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*(expectedValue-s)*inputValues[i]
    def process(self,inputValues):
        inputValuesList=[]
        inputValues=iter(inputValues)
        for x in range(len(self._weights)):
            inputValuesList.append(next(inputValues))
        self.__dict__['_inputValues']=np.array(inputValuesList)
        if len(self._inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        self.__dict__['_val']=self._activFunc(np.dot(self._weights,self._inputValues))
        return self._val
    def propagateError(self,weights,errors,learnRate=False):
        weights=np.array(weights)
        errors=np.array(errors)
        if len(errors)!=len(weights):
            raise TypeError('Wrong values length')
        if learnRate:
            self._learnRate=learnRate
        self.__dict__['_error']=self._activFuncDeriv(self._val)*np.dot(weights,errors)
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*self._error*self._inputValues[i]
        return self._error
    def __getitem__(self,index):
        if index=='error':
            return self._error
        return self._weights[index]
    def __setitem__(self,index,value):
        if index=='activFunc':
            self._activFunc=value
        elif index=='activFuncDeriv':
            self._activFuncDeriv=value
        elif index=='learnRate':
            self._learnRate=value
        else:
            self._weights[index]=value
    def __getattr__(self,attr):
        raise AttributeError('No such attribute: %r'%attr)
    def __setattr__(self,attr,value):
        if attr=='learnRate':
            self._learnRate=value
        else:
            raise AttributeError('No such attribute: %r'%attr)
    def __iter__(self):
        return iter(self._weights)
    def __len__(self):
        return len(self._weights)
    def __repr__(self):
        w='['
        for x in self._weights:
            w+="{!r:.7},".format(x)
        w=w[:-1]
        w+=']'
        return 'Perceptron(weights:{0!r:},learnRate:({1!r:.7}),activFunc:{2!r})'.format(w,self._learnRate,self._activFunc.__name__)


class PerceptronLayer:
    def __init__(self,percepNumber,inputNumber,activFunc,activFuncDeriv,learnRate=0.1):
        self._learnRate=learnRate
        self._perceptrons=[Perceptron([np.random.ranf() for _ in range(inputNumber)],self._learnRate,activFunc,activFuncDeriv) for _ in range(percepNumber)]
    def __getitem__(self,index):
        return self._perceptrons[index]
    def __setitem__(self,index,value):
        if index=='learnRate':
            self._lernRate=value
            for x in self._perceptrons:
                x['learnRate']=value
                
class Multilayer:
    def __init__(self,perceptronLayers):
        self._perceptronLayers=perceptronLayers
    def learn(self,inputValues,expectedValues):
        inpValIter=iter(inputValues)
        for layer in self._perceptronLayers:
            for p in layer:
                p.process(inpValIter)
                


if __name__=='__main__':
        
    inputData=(
            ((1,0,0),0),
            ((1,0,1),0),
            ((1,1,0),0),
            ((1,1,1),1)
            )
    
    learnRates=[abs(np.random.ranf())*10 for _ in range(10)]
    
    p=Perceptron([abs(np.random.ranf()) for _ in range(3)],1,naturalOne,one)
    print(p)
    i=0
    while(True):
        cont=False
        i+=1
        p.learn(*inputData[np.random.choice(len(inputData))])
        for data,expected in inputData:
            r=p.process(data)
            if r!=expected:
                cont=True
                break
        if not cont:
            break
    print(i)
    print(p)
    print
    for x in inputData:
        print(p.process(x[0]))