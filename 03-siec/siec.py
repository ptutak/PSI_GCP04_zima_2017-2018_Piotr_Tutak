# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:30:59 2017

@author: Piotrek
"""

from perceptron import *
import numpy as np

import copy
import time
import random

def rastrigin(x,y):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

np.random.seed(7)

with open('training_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

with open('test_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

np.random.seed(7)

dataSet=np.loadtxt('training_data.csv',delimiter=',')
inputData = dataSet[:,0:2]
expected = dataSet[:,2]

testDataSet=np.loadtxt('test_data.csv',delimiter=',')
inputTestData = testDataSet[:,0:2]
expectedTestData = testDataSet[:,2]


INTERN_LAYERS=2
LAYERS=[20,10]


ACTIV_FUNCS=[Sigm()(1) for x in range(INTERN_LAYERS)]
ACTIV_FUNC_DERIVS=[Sigm().derivative(1) for x in range(INTERN_LAYERS)]
LEARN_RATES=[0.01 for x in range(INTERN_LAYERS)]
WEIGHTS=[None for x in range(INTERN_LAYERS)]
BIASES=[0.0 for x in range(INTERN_LAYERS)]

multOrig=Multilayer(
        [2,*LAYERS,1],
        [ident,*ACTIV_FUNCS,ident],
        [zero,*ACTIV_FUNC_DERIVS,one],
        [[1.0],*WEIGHTS,[1.0 for x in range(5)]],
        [0.0,*LEARN_RATES,0.01],
        [0.0,*BIASES,0.0]        
        )

error=1.0
while error>0.0001:
    multilayer=copy.deepcopy(multOrig)
    i=0
    start=time.clock()
    run=True
    while(run):
        samples=list(zip(inputData,expected))
        while(run and samples):
            inp=samples.pop(np.random.randint(0,len(samples)))
            multilayer.learn(inp[0],[inp[1]])

        results=[]
        for inp in inputData:
            results.extend(multilayer.process(inp))
         
        error=MSE(results,expected)
        if error<0.0001:
            run=False    
        i+=1
        print("{0:9};{1: 8.5f}".format(i,time.clock()-start),error,sep=';')
