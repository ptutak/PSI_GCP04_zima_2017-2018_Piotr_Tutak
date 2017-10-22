# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:20:52 2017

@author: PiotrTutak
"""

from perceptron import *
import random
import time
import numpy as np


litery=dict()
literyLow=dict()
literyHigh=dict()
with open('litery.txt','r') as f:
    for l in f:
        x=l.strip().split('=')
        if x[0].islower():
            litery[x[0]]=([float(a) for a in x[1]],[0.0])
            literyLow[x[0]]=([float(a) for a in x[1]],[0.0])
        else:
            litery[x[0]]=([float(a) for a in x[1]],[1.0])
            literyHigh[x[0]]=([float(a) for a in x[1]],[1.0])

litery20=random.sample(list(literyLow.items()),10)
litery20.extend(random.sample(list(literyHigh.items()),10))
litery20Low=[x for x in litery20 if x[0].islower()]
litery20High=[x for x in litery20 if x[0].isupper()]
print('użyte litery:')
print(*list(x[0] for x in litery20),sep='\n')


listPerc=[]
RES_NUMBER=1
HIDDEN_LAYER_PERCEP_NUMB=15
while(len(listPerc)<RES_NUMBER):
    multilayer=Multilayer(
             [35,HIDDEN_LAYER_PERCEP_NUMB,1],
             [hardOne,SignSigm()(1.0),ident],
             [zero,SignSigm().derivative(1.0),one],
             [[1.0],None,[1.0 for x in range(HIDDEN_LAYER_PERCEP_NUMB)]],
             [0.0,0.1,0.0],
             [-0.5,-0.11,0.0]
             )
    print(multilayer)
    i=0
    run=True
    start=time.clock()
    while(run):
        samples=list(litery20)
        while(run and samples):
            inp=random.sample(samples,1).pop(0)
            samples.remove(inp)
            multilayer.learn(inp[1][0],inp[1][1])
            dif=0.0
            for char,data in litery20:
                r=multilayer.process(data[0])
                dif=max(abs(data[1][0]-r[0]),dif)
                
            if dif<0.0001:
                run=False
        i+=1
        print(i,dif,sep=';;')
    listPerc.append(multilayer)

for m in listPerc:
    print(m)

example=listPerc[0]
print('letter',' result  ','expected',sep='||')
for key,value in litery20:
    r=example.process(value[0])
    print("{0:6}||{1: 8f}||{2:8}".format(key,*r,*value[1]))   
