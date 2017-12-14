# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:39 2017

@author: PiotrTutak
"""

import numpy as np
import scipy.linalg as lg


def simpleLearnCorrection(l):
    t=-1
    def f():
        nonlocal t
        t+=1
        return np.exp(-t/l)
    return f

def distanceEuklides(input1,input2):
    s=0.0
    for i in range(len(input1)):
        s+=(input1[i]-input2[i])**2
    return np.sqrt(s)

def radiusSimple(l, distanceF):
    def f(input1, input2):
        d=distanceF(input1,input2)
        if d<=l:
            return 1.0
        return 0.0
    return f

def radiusGauss(l,distanceF):
    def f(input1,input2):
        d=distanceF(input1,input2)
        return np.exp(-(d**2)/(l**2))
    return f

class LayerKohonen:
    def __init__(self, size, inputNumber, distanceFunc, radiusFunc, learnRate=0.1, learnFunc=None):
        m,n=size
        self.size=size
        self.m=m
        self.n=n
        neurons=[]
        for i in range(m):
            neurons.append([])
            for j in range(n):
                neurons[i].append(dict())
                d=np.array([np.random.uniform(-1.0,1.0) for x in range(inputNumber)])
                neurons[i][j]['w']=d/lg.norm(d)
#                print(neurons[i][j]['w'])
#                print(lg.norm(neurons[i][j]['w']))
        self.neurons=neurons
        self.distanceFunc=distanceFunc
        self.radiusFunc=radiusFunc
        self.inputNumber=inputNumber
        self.learnRate=learnRate
        self.actualLearnRate=learnRate
        self.learnFunc=learnFunc

    def processKohonen(self, inputValues):
        minDist=self.distanceFunc([0 for x in range(self.inputNumber)],inputValues)
        minNeuron=dict()
        i=0
        for row in self.neurons:
            j=0
            for neuron in row:
                res=self.distanceFunc(neuron['w'],inputValues)
 #               print(res)
                neuron['d']=res
                if res<minDist:
                    minDist=res
                    minNeuron['i']=i
                    minNeuron['j']=j
                    minNeuron['w']=neuron['w']
                    minNeuron['d']=neuron['d']
                j+=1
            i+=1
            
#        print(minNeuron)
        return minNeuron
    
    def learnKohonen(self,inputValues):
        minNeuron=self.processKohonen(inputValues)
        i=0
        if self.learnFunc:
            self.actualLearnRate=self.learnRate*self.learnFunc()
        for row in self.neurons:
            j=0
            for neuron in row:
                newWeights=[]
                for k in range(self.inputNumber):
                    newWeights.append(neuron['w'][k]+self.actualLearnRate*self.radiusFunc((i,j),(minNeuron['i'],minNeuron['j']))*(inputValues[k]-neuron['w'][k]))
                newWeights=np.array(newWeights)
                neuron['w']=newWeights/lg.norm(newWeights)
                j+=1
            i+=1
    def __str__(self):
        ret=""
        i=0
        for row in self.neurons:
            j=0
            for neuron in row:
                ret+="{0} {1:.5f} ".format((i,j),neuron['d'])
                j+=1
            ret+="\n"
            i+=1
        return ret
    