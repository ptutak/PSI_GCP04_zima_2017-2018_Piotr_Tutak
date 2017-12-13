# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:39 2017

@author: PiotrTutak
"""
import numpy as np
from layerkohonen import *
from copy import deepcopy

data=[]
expected=[]
with open('data.txt') as f:
    for line in f:
        l=line.strip().split()
        expected.append(l[-1])
        l=[float(x) for x in l[:4]]
        data.append(np.array(l))
        
print(*data,sep='\n')
print(expected)
layer=LayerKohonen((15,15),4,distanceEuklides,radiusSimple(0.0,distanceEuklides))

while(True):
    samples=deepcopy(data)
    np.random.shuffle(samples)
    while(samples):
        d=samples.pop(0)
        layer.learnKohonen(d)
    results=dict()
    for x in data:
        res=tuple(sorted(layer.processKohonen(x).items()))[1:3]
       # print(res)
        if res in results:
            results[res]+=1
        else:
            results[res]=0
    print(sorted(results.items()))
