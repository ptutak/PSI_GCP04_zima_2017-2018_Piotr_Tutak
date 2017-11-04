# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:30:59 2017

@author: Piotrek
"""

from perceptron import *
import numpy as np

def rastrigin(x,y):
    return 20+x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)

np.random.seed(7)

with open('training_data.csv','w') as f:
    for i in range(1000):
        x=np.random.sample()*4-2
        y=np.random.sample()*4-2
        print('{0},{1},{2}'.format(x,y,rastrigin(x,y)),file=f)

np.random.seed(7)

dataset=np.loadtxt('training_data.csv',delimiter=',')
inputData = dataset[:,0:2]
expected = dataset[:,2]

print(inputData)
print(expected)


INTERN_LAYERS=3
LAYERS=[5 for x in range(INTERN_LAYERS)]
ACTIV_FUNCS=[SignSigm()(1) for x in range(INTERN_LAYERS)]
ACTIV_FUNC_DERIVS=[SignSigm().derivative(1) for x in range(INTERN_LAYERS)]
LEARN_RATES=[0.1 for x in range(INTERN_LAYERS)]
WEIGHTS=[None for x in range(INTERN_LAYERS)]
BIASES=[1.0 for x in range(INTERN_LAYERS)]

mult=Multilayer(
        [2,*LAYERS,1],
        [ident,*ACTIV_FUNCS,ident],
        [zero,*ACTIV_FUNC_DERIVS,zero],
        [[1.0],*WEIGHTS,[1.0 for x in range(5)]],
        [0.0,*LEARN_RATES,0.0],
        [0.0,*BIASES,0.0]        
        )

