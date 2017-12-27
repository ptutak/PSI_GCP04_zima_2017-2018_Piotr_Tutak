# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:07:39 2017

@author: PiotrTutak
"""
import numpy as np
from layerkohonen import *
from copy import deepcopy
import sys

data=[]
flowers=set()
with open('data.txt') as f:
    for line in f:
        l=line.strip().split()
        expected=l[-1]
        if expected not in flowers:
            flowers.add(expected)
        l=[float(x) for x in l[:4]]
        data.append((np.array(l),expected))
        
#LR_CORR=60
SIZE=[x for x in range(5,21,3)]
LEARN_RATE=[0.5,0.1, 0.01, 0.001]
STDOUT=sys.stdout
sys.stdout=open('results-no-corr.txt','w')
try:
    for s in SIZE:
        layerOrig=LayerKohonen((s,s),4,distanceEuklides,radiusSimple(0.0,distanceEuklides),None,None)
        for l in LEARN_RATE:
            layer=deepcopy(layerOrig)
            #layer.learnFunc=simpleLearnCorrection(LR_CORR*s*s)
            layer.learnRate=l
            layer.actualLearnRate=l
            #np.random.seed(12)
            for i in range(100):
                samples=deepcopy(data)
                np.random.shuffle(samples)
                while(samples):
                    d=samples.pop(0)
                    layer.learnKohonen(d[0])
                results=dict()
            for f in flowers:
                results[f]=dict()
            for x in data:
                res=layer.processKohonen(x[0])['uid']
                if res in results[x[1]]:
                    results[x[1]][res]+=1
                else:
                    results[x[1]][res]=1
            print(i,s,layer.actualLearnRate,*sorted(results.items()))
finally:
    sys.stdout.close()
    sys.stdout=STDOUT
