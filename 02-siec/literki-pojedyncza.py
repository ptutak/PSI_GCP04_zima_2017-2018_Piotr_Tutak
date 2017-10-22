# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:20:52 2017

@author: PiotrTutak
"""

from perceptron import *
from operator import itemgetter
import random
import time
import numpy as np
import sys

def listWithPrec(listA,prec):
    ret="["
    formatStr="{0: "+str(int(prec+3))+"."+str(int(prec))+"f}"
    for x in listA:
        ret+=formatStr.format(x)
        ret+=","
    ret=ret[:-1]+']'
    return ret

def MSE(results,expected):
    sum=0.0
    for i in range(len(results)):
        sum+=(results[i]-expected[i])**2
    return sum/len(results)

def MAPE(results,expected):
    sum=0.0
    for i in range(len(results)):
        sum+=abs((expected[i]-results[i])/expected[i])
    return 100*sum/len(results)


STDOUT=sys.stdout
f=open('results.txt','w');
sys.stdout=f


litery=dict()
literyLow=dict()
literyHigh=dict()
with open('litery.txt','r') as f:
    for l in f:
        x=l.strip().split('=')
        if x[0].islower():
            litery[x[0]]=[float(a) for a in x[1]]
            literyLow[x[0]]=[float(a) for a in x[1]]
        else:
            litery[x[0]]=[float(a) for a in x[1]]
            literyHigh[x[0]]=[float(a) for a in x[1]]

litery20=sorted(random.sample(list(literyLow.items()),10),key=itemgetter(0))
litery20.extend(sorted(random.sample(list(literyHigh.items()),10),key=itemgetter(0)))
litery20Expected=[[1.0 if x==y else 0.0 for x in range(1,21)] for y in range(1,21)]
litery20Low=[x for x in litery20 if x[0].islower()]
litery20High=[x for x in litery20 if x[0].isupper()]
print('użyte litery:')
print(*list(x[0] for x in litery20),sep='\n')


listPerc=[]
RES_NUMBER=2
HIDDEN_LAYER_PERCEP_NUMB=20
while(len(listPerc)<RES_NUMBER):

    multilayer=Multilayer(
             [35,HIDDEN_LAYER_PERCEP_NUMB],
             [hardOne,SignSigm()(1.0)],
             [zero,SignSigm().derivative(1.0)],
             [[1.0],None],
             [0.0,0.3],
             [-0.5,-0.01]
             )
    i=0
    run=True
    start=time.clock()
    while(run):
        samples=list(litery20)
        while(run and samples):
            inp=random.sample(samples,1).pop(0)
            samples.remove(inp)
            multilayer.learn(inp[1],litery20Expected[litery20.index(inp)])
        
        dif=0.0
        for inp in litery20:
            r=multilayer.process(inp[1])
            if len(listPerc)<2:
                error=MSE(r,litery20Expected[litery20.index(inp)])
            else:
                error=MAPE(r,litery20Expected[litery20.index(inp)])
        multilayer.multiLearnRates(1.0/-np.log10(error))
        if error<0.00001:
            run=False    
        i+=1
        print(i,time.clock()-start,error,sep=';')
    listPerc.append(multilayer)

for m in listPerc:
    print(m)

example=listPerc[0]
for item in litery20:
    r=example.process(item[1])
    print("{0:6}\n{1!s}\n{2!s}".format(item[0],listWithPrec(r,5),listWithPrec(litery20Expected[litery20.index(item)],5)))   


f.close()
sys.stdout=STDOUT
