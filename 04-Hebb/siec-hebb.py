# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:45:52 2017

@author: PiotrTutak
"""
import numpy as np

np.random.seed(7)

from neuronhebb import *
from neuron import *

import time
import os


buzki=dict()

for file in os.listdir("."):
    if file.startswith("b") and file.endswith(".txt"):
        print(os.path.join(".", file))
        buzka=[]
        with open(file,'r') as f:
            for line in f:
                buzka.extend([int(x) for x in line.strip()])
        buzki[file[2:-4]]=buzka


print(buzki)
"""
x=NeuronHebb([(0.8*np.random.ranf()+0.1)*np.random.choice([-1.0,1.0]) for _ in range(64)],ident,learnRate=0.001,forgetRate=0.051,bias=-0.8*np.random.ranf()-0.1)
while (True):
    print(x)
    for buzka in buzki.items():
        x.process(buzka[1])
        x.learn()
    print(x)
    print('\n')
"""
multilayer=MultilayerHebb(
        layers=[64,64,4],
        activFuncs=[ident,SignSigm()(1.0),Sigm()(1.0)],
        weights=[[1.0],None,None],
        learnRates=[0.0,0.05,0.05],
        forgetRates=[0.0,0.2,0.2],
        biases=[0.0,None,None]
        )

print(multilayer)

while (True):
    samples=list(buzki.items())
    while (samples):
        inp=random.sample(samples,1).pop(0)
        samples.remove(inp)
        multilayer.learn(inp[1])
    
    results=[]
    for buzka in buzki.items():
        results.append(multilayer.process(buzka[1]))
    print(*results,'\n',sep='\n')
    time.sleep(1.0)
