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
        

LR_CORR=10
SIZE=[5,10,15]
LEARN_RATE=[0.5,0.1, 0.01]
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
                for i in range(10):
                    np.random.seed(12)
                    samples=deepcopy(data)
                    np.random.shuffle(samples)
                    while(samples):
                        d=samples.pop(0)
                        layer.learnKohonen(d[0])
                
                results=dict()
                for neuron in layer:
                    results[neuron['coords']]=set()
                maxLen=0
                for x in data:
                    res=layer.processKohonen(x[0])
                    results[res['coords']].add(x[1])
                    if len(results[res['coords']])>maxLen:
                        maxLen=len(results[res['coords']])
                print(i,s,r,l,layer.actualLearnRate,layer.learnFunc.__name__)
                
                formatStr="{0:^"+str(maxLen)+"}  "
                toPrint=""
                for res in sorted(list(results.items())):
                    let="".join(sorted(res[1]))
                    if not let:
                        let="0"
                    toPrint+=formatStr.format(let)
                    if res[0][1]==s-1:
                        toPrint+="\n"
                print(toPrint)
                print()
finally:
#    sys.stdout.close()
#    sys.stdout=STDOUT
    pass
