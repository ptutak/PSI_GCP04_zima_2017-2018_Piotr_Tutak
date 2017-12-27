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
letters=set()
with open('data.txt') as f:
    for line in f:
        l=line.strip().split()
        expected=l[0]
        letters.add(expected)
        letter=[float(x) for x in l[-1]]
        data.append((np.array(letter),expected))
        

LR_CORR=100
SIZE=[x for x in range(5,21,3)]
LEARN_RATE=[0.5,0.1, 0.01, 0.001]
RADIUS = [1.0,2.0,3.0,4.0]
#STDOUT=sys.stdout
#sys.stdout=open('results.txt','w')
try:
    for s in SIZE:
        layerOrig=LayerKohonen((s,s),35,distanceEuklides,None,None,None)
        for l in LEARN_RATE:
            for r in RADIUS:
                layer=deepcopy(layerOrig)
                layer.learnFunc=simpleLearnCorrection(LR_CORR*24)
                layer.learnRate=l
                layer.actualLearnRate=l
                layer.radiusFunc=radiusGauss(r,distanceEuklides)
                #np.random.seed(12)
                for i in range(1):
                    samples=deepcopy(data)
                    np.random.shuffle(samples)
                    while(samples):
                        d=samples.pop(0)
                        layer.learnKohonen(d[0])
                    results=dict()
                for letter in letters:
                    results[letter]=dict()
                for x in data:
                    res=layer.processKohonen(x[0])['uid']
                    if res in results[x[1]]:
                        results[x[1]][res]+=1
                    else:
                        results[x[1]][res]=1
                print(i,s,l,layer.actualLearnRate,layer.learnFunc.__name__,*sorted(results.items()))
finally:
#    sys.stdout.close()
#    sys.stdout=STDOUT
    pass
