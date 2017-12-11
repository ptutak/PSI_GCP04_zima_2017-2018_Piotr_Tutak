# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:39 2017

@author: PiotrTutak
"""

import numpy as np





class LayerKohonen:
    def __init__(self, size,inputNumber, distanceFunc, radiusFunc):
        m,n=size
        self.m=m
        self.n=n
        neurons=[]
        for i in range(m):
            neurons.append([])
            for j in range(n):
                neurons[i].append(dict())
                neurons[i]['w']=[np.random.uniform(-1.0,1.0) for x in inputNumber]
        self.neurons=neurons
        self.distanceFunc=distanceFunc
        
    def learnKohonen(self,inputValues):
        minDist=0
        minNeuron=None
        for row in self.neurons:
            for neuron in row:
                res=self.distanceFunc(neuron['w'],inputValues)
                neuron['d']=res
                if res<minDist:
                    minDist=res
                    minNeuron=neuron
        