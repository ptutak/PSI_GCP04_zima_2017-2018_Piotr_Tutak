# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:39 2017

@author: PiotrTutak
"""

import numpy as np
import scipy.linalg as lg


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
    def __init__(self, size, inputNumber, distanceFunc, radiusFunc, learnRate=0.1):
        m,n=size
        self.size=size
        self.m=m
        self.n=n
        neurons=[]
        for i in range(m):
            neurons.append([])
            for j in range(n):
                neurons[i].append(dict())
                neurons[i]['w']=list(lg.norm([np.random.uniform(-1.0,1.0) for x in inputNumber]))
        self.neurons=neurons
        self.distanceFunc=distanceFunc
        self.radiusFunc=radiusFunc
        self.inputNumber=inputNumber
        self.learnRate=learnRate

    def learnKohonen(self,inputValues):
        minDist=0
        minNeuron=dict()
        i=0
        for row in self.neurons:
            j=0
            for neuron in row:
                res=self.distanceFunc(neuron['w'],inputValues)
                neuron['d']=res
                if res<minDist:
                    minDist=res
                    minNeuron['i']=i
                    minNeuron['j']=j
                    minNeuron['n']=neuron
                j+=1
            i+=1
        i=0
        for row in self.neurons:
            j=0
            for neuron in row:
                for k in range(self.inputNumber):
                    neuron['w'][k]+=self.learnRate*self.radiusFunc((i,j),(minNeuron['i'],minNeuron['j']))(inputValues[k]-neuron['w'][k])
                neuron['w']=list(lg.norm(neuron['w']))
                j+=1
            i+=1
    