# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:45:52 2017

@author: PiotrTutak
"""
import numpy as np

np.random.seed(7)

from operator import itemgetter
from neuronhebb import *
from neuron import *
import time
import os
import sys

buzki=dict()
buzkiTest=dict()
for file in os.listdir("."):
    if file.startswith("b") and file.endswith(".txt"):
        print(os.path.join(".", file))
        buzka=[]
        with open(file,'r') as f:
            for line in f:
                buzka.extend([int(x) for x in line.strip()])
        buzki[file[2:-4]]=np.array(buzka)
        buzkiTest['test-'+file[2:-4]]=np.array(buzka)


for buzka in buzkiTest:
    for x in range(10):
        index=np.random.choice(64)
        buzkiTest[buzka][index]^=1

a=[0 for x in range(50)]
a.extend([1 for x in range(14)])
buzki['test']=np.array(a)
buzkiTest['test']=np.array(a)
print(buzki)
print(buzkiTest)
#sys.exit("end")
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
        learnRates=[0.0,0.007],
        forgetRates=[0.0,0.1],
        biases=[-0.5,None]
        )

#print(multilayer)
startTime=time.clock()
print("\n\nstart learning:")

ITER_NUMBER=100

for i in range(ITER_NUMBER):
    samples=list(sorted(buzki.items()))[:]
    while (samples):
        inp=random.sample(samples,1).pop(0)
        samples.remove(inp)
        multilayer.learnHebb(inp[1])
    
    results=[]
    for buzka in sorted(buzkiTest.items(),key=itemgetter(0)):
        results.append((buzka[0],*multilayer.process(buzka[1])))
    for buzka in sorted(buzki.items(),key=itemgetter(0)):
        results.append((buzka[0],*multilayer.process(buzka[1])))
    
    print(*results,"time:{0:.5f}".format(time.clock()-startTime))
#    time.sleep(1.0)
