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
        buzki[file[2:-4]]=np.array(buzka)


print(buzki)
a=[0 for x in range(50)]
a.extend([1 for x in range(14)])
buzki['ztest']=np.array(a)
print(list(sorted(buzki.items())))
"""
x=NeuronHebb([(0.8*np.random.ranf()+0.1)*np.random.choice([-1.0,1.0]) for _ in range(64)],ident,learnRate=0.001,forgetRate=0.051,bias=-0.8*np.random.ranf()-0.1)
while (True):
    print(x)
    for buzka in buzki.items():
        x.process(buzka[1])
        x.learnHebb()
        time.sleep(0.5)
    print(x)
    print('\n')
"""
multilayer=MultilayerHebb(
        layers=[64,1],
        activFuncs=[hardSign,SignSigm()(1.0)],
        weights=[[1.0],None],
        learnRates=[0.0,0.009],
        forgetRates=[0.0,0.1],
        biases=[-0.5,None]
        )

print(multilayer)

while (True):
    samples=list(sorted(buzki.items()))[:]
    while (samples):
        inp=random.sample(samples,1).pop(0)
        samples.remove(inp)
        multilayer.learnHebb(inp[1])
    
    results=[]
    for buzka in buzki.items():
        results.append(multilayer.process(buzka[1]))
    print(*results,'\n',sep='\n')
#    time.sleep(1.0)
