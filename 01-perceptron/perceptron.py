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
    def __init__(self, weights, learnRate, activFunc, activFuncDeriv, bias=0):
        self.__dict__['_weights']=np.array(weights)
        self.__dict__['_learnRate']=learnRate
        self.__dict__['_activFunc']=activFunc
        self.__dict__['_activFuncDeriv']=activFuncDeriv
        self.__dict__['_bias']=bias
        self.__dict__['_error']=0
    def learn(self,inputValues,expectedValue,learnRate=False):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        if (learnRate):
            print('x')
            self._learnRate=learnRate
        s=np.dot(self._weights,inputValues)+self._bias
        s=self._activFunc(s)
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*(expectedValue-s)*inputValues[i]
        print(self._bias)
        self.__dict__['_bias']+=self._learnRate*(expectedValue-s)
    def process(self,inputValues):
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        self.__dict__['_inputValues']=np.array(inputValues)
        self.__dict__['_val']=self._activFunc(np.dot(self._weights,self._inputValues)+self._bias)
        return self._val
    def propagateError(self,weights,errors,learnRate=False):
        weights=np.array(weights)
        errors=np.array(errors)
        if len(errors)!=len(weights):
            raise TypeError('Wrong values length')
        if learnRate:
            self.__dict__['_learnRate']=learnRate
        self.__dict__['_error']=self._activFuncDeriv(self._val)*np.dot(weights,errors)
        for i in range(len(self._weights)):
            self._weights[i]+=self._learnRate*self._error*self._inputValues[i]
        self.__dict__['_bias']+=self._learnRate*self._error
        return self._error
    def __getitem__(self,index):
        if index=='error':
            return self._error
        return self._weights[index]
    def __setitem__(self,index,value):
        if index=='learnRate':
            self._learnRate=value
        else:
            self._weights[index]=value
    def __getattr__(self,attr):
        raise AttributeError('get: No such attribute: %r'%attr)
    def __setattr__(self,attr,value):
        if attr=='learnRate':
            self._learnRate=value
        else:
            raise AttributeError('set: No such attribute: %r'%attr)
    def __iter__(self):
        return iter(self._weights)
    def __len__(self):
        return len(self._weights)
    def __repr__(self):
        w='['+','.join('{:8.5f}'.format(x) for x in self._weights)+']'
        return 'Perceptron(weights:{0},learnRate:({1:.5f}),activFunc:{2!r})'.format(w,self._learnRate,self._activFunc.__name__)


class Layer:
    def __init__(self,percepNumber,inputNumber,activFunc,activFuncDeriv,learnRate=0.1):
        self._learnRate=learnRate
        self._perceptrons=[Perceptron([np.random.ranf()*np.random.choice([-1,1]) for _ in range(inputNumber)],self._learnRate,activFunc,activFuncDeriv) for _ in range(percepNumber)]
    def __getitem__(self,index):
        return self._perceptrons[index]
    def __setitem__(self,index,value):
        if index=='learnRate':
            self._learnRate=value
            for x in self._perceptrons:
                x['learnRate']=value
                
class Multilayer:
    def __init__(self,perceptronLayers):
        self._perceptronLayers=perceptronLayers
    def process(self,inputValues):
        inputValues=list(inputValues)
        values=[]
        for layer in self._perceptronLayers:
            values=[]
            for p in layer:
                values.append(p.process(inputValues[:len(p)]))
                inputValues=inputValues[len(p):]
            inputValues=values
        return values
        
    def learn(self,inputValues,expectedValues):
        inputValues=list(inputValues)
        values=[]
        for layer in self._perceptronLayers:
            values=[]
            for p in layer:
                print(inputValues[:len(p)])
                values.append(p.process(inputValues[:len(p)]))
                inputValues=inputValues[len(p):]
            inputValues=values
        results=iter(values)
        i=len(expectedValues)
        expectedValues=iter(expectedValues)
        errors=[]
        for _ in range(i):
            errors.append([next(results)-next(expectedValues)])
        weights=[[1] for x in range(len(errors))]
        for layer in reversed(self._perceptronLayers):
            newErrors=[]
            newWeights=[]
            for p in layer:
                newErrors.append(p.propagateError(weights.pop(0),errors.pop(0)))
                newWeights.append(p[:-1])
            print(newWeights)
            print(newErrors)
            weights=list(zip(*newWeights))
            print(weights)
            errors=[newErrors for x in range(len(weights))]
            print(errors)
                

if __name__=='__main__':
        
    inputData=(
            ((1,0,0),0),
            ((1,0,1),0),
            ((1,1,0),0),
            ((1,1,1),1)
            )
    
    listPercMin=[]
    listPercAver=[]
    listPercMax=[]
    RES_NUMBER=1
    while(len(listPercMin)<RES_NUMBER or len(listPercAver)<RES_NUMBER or len(listPercMax)<RES_NUMBER):
        w=[np.random.ranf()*np.random.choice([-1,1]) for _ in range(3)]
        p=Perceptron(w,np.random.ranf()*np.random.ranf()*np.random.ranf(),naturalOne,one)
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
        w='initialWeights:['+','.join('{:8.5f}'.format(x) for x in w)+']'
        if i<10 and len(listPercMin)<RES_NUMBER:
            listPercMin.append((w,p,"iterNumber: %d"%i))
            print(i)
        elif i>=10 and i<1000 and len(listPercAver)<RES_NUMBER:
            listPercAver.append((w,p,"iterNumber: %d"%i))
            print(i)
        elif i>=1000 and len(listPercMax)<RES_NUMBER:
            listPercMax.append((w,p,"iterNumber: %d"%i))
            print(i)
    print('\n------------ min iter number ------------\n')
    for x in listPercMin:
        print(x)
    print('\n------------ average iter number ------------\n')
    for x in listPercAver:
        print(x)
    print('\n------------ max iter number ------------\n')
    for x in listPercMax:
        print(x)
    print('\n\n\n\n')
    xorInputData=(
            ((0,0),[0]),
            ((0,1),[1]),
            ((1,0),[1]),
            ((1,1),[0])
            )
    w=[np.random.ranf()*np.random.choice([-1,1]) for _ in range(3)]
    firstLayer=Layer(2,1,Sigm()(1),Sigm().derivative(1))
    outputLayer=Layer(1,2,naturalOne,one)
    mult=Multilayer([firstLayer,outputLayer])
    
    i=0
    while(True):
        cont=False
        i+=1
        print(*xorInputData[0])        
        mult.learn(*xorInputData[np.random.choice(len(xorInputData))])
        for data,expected in xorInputData:
            r=mult.process(data)
            if r[0]!=expected:
                cont=True
                break
        if not cont:
            break
    for data,expected in xorInputData:
        r=mult.process(data)
        print(data,r)
    print('iter number: %d',i)
 