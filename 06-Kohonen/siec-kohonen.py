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
testData=[]
letters=set()
with open('data.txt') as f:
    for line in f:
        l=line.strip().split()
        expected=l[0]
        letters.add(expected)
        letter=[float(x) for x in l[-1]]
        testLetter=list(letter)
        change=set()
        while len(change)<5:
            change.add(np.random.randint(0,len(letter)))
        for x in change:
            if testLetter[x]:
                testLetter[x]=0.0
            else:
                testLetter[x]=1.0
        data.append((np.array(letter),expected))
        testData.append((np.array(testLetter),expected))
with open('testData.txt','w') as f:
    for x in testData:
        f.write(x[1])
        f.write(' ')
        for y in x[0]:
            f.write(str(int(y)))
        f.write('\n')

EPOCHS=100
SIZE=[5,10,15]
LEARN_RATE=[0.5,0.1,0.01]
RADIUS = [1.0,2.0,3.0]
STDOUT=sys.stdout
sys.stdout=open('results.txt','w')
resFile=open('results-results.txt','w')
try:
    for s in SIZE:
        np.random.seed(12)
        layerOrig=LayerKohonen((s,s),35,distanceEuklides,None,None,None)
        for l in LEARN_RATE:
            for r in RADIUS:
                layer=deepcopy(layerOrig)
                layer.learnFunc=simpleLearnCorrection(EPOCHS*20)
                layer.learnRate=l
                layer.actualLearnRate=l
                layer.radiusFunc=radiusGauss(r,distanceEuklides)
                np.random.seed(12)
                for i in range(EPOCHS):
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
                print('DATA:')
                formatStr="{0:^"+str(maxLen)+"}  "
                toPrint=""
                for res in sorted(list(results.items())):
                    let="".join(sorted(res[1]))
                    if not let:
                        let="."
                    toPrint+=formatStr.format(let)
                    if res[0][1]==s-1:
                        toPrint+="\n"
                print(toPrint)
                print('TEST DATA:')
                
                testResults=dict()
                for neuron in layer:
                    testResults[neuron['coords']]=set()
                maxLen=0
                acc=0
                i=0
                for x in testData:
                    res=layer.processKohonen(x[0])
                    if res['coords']==layer.processKohonen(data[i][0])['coords']:
                        acc+=1
                    i+=1
                    testResults[res['coords']].add(x[1])
                    if len(testResults[res['coords']])>maxLen:
                        maxLen=len(testResults[res['coords']])
                formatStr="{0:^"+str(maxLen)+"}  "
                toPrint=""
                for res in sorted(list(testResults.items())):
                    let="".join(sorted(res[1]))
                    if not let:
                        let="."
                    toPrint+=formatStr.format(let)
                    if res[0][1]==s-1:
                        toPrint+="\n"
                print(toPrint)
                print("size:{0}, radius:{1}, learningRate:{2}, endLearnRate:{3:.7}, accuracy:{4:.3}\n".format(s,r,l,layer.actualLearnRate,acc/20*100))
                print("{0} {1} {2} {3} {4:.3}".format(s,r,l,layer.actualLearnRate,acc/20*100),file=resFile)
                
finally:
    sys.stdout.close()
    sys.stdout=STDOUT
    resFile.close()
    pass
