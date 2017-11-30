# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:45:52 2017

@author: PiotrTutak
"""
import numpy as np

np.random.seed(7)

from operator import itemgetter
from copy import deepcopy
from neuronhebb import *
from neuron import *
import time
import os
import sys



LEARN_RATE=0.007
FORGET_RATE=0.1



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
buzkiTest['test']=np.array(a)

print(*buzki.items(),sep='\n')
print(*buzkiTest.items(),sep='\n')
buzki['test']=np.array(a)
#sys.exit("end")



multilayerOrig=MultilayerHebb(
        layers=[64,1],
        activFuncs=[hardSign,SignSigm()(1.0)],
        weights=[[1.0],None],
        learnRates=[0.0,0.0],
        forgetRates=[0.0,0.0],
        biases=[-0.5,None]
        )

learnRates=[0.1,0.01,0.007,0.005,0.003,0.001,0.0001]
forgetRates=[0.0,0.1,0.3,0.5,0.9]

#print(multilayer)
for lr in learnRates:
    for fr in forgetRates:
        multilayer=deepcopy(multilayerOrig)
        multilayer.setForgetRates(fr)
        multilayer.setLearnRates(lr)
        startTime=time.clock()
        print("\n\nstart learning lr:{0:.5f} fr:{1:.5f}".format(lr,fr))
        
        STDOUT=sys.stdout
        sys.stdout=open("hebb-lr-{0}-fr-{1}.txt".format(lr,fr),"w")
        
        ITER_NUMBER=100
        np.random.seed(7)
        for i in range(ITER_NUMBER):
            samples=list(sorted(buzki.items()))[:]
            while (samples):
                inp=random.sample(samples,1).pop(0)
                samples.remove(inp)
                multilayer.learnHebb(inp[1])
            
            results=[]
            for buzka in sorted(buzkiTest.items(),key=itemgetter(0)):
                results.append(buzka[0])
                results.append("{0:.5f}".format(*multilayer.process(buzka[1])))
            for buzka in sorted(buzki.items(),key=itemgetter(0)):
                results.append(buzka[0])
                results.append("{0:.5f}".format(*multilayer.process(buzka[1])))
            
            print(*results,"time {0:.5f}".format(time.clock()-startTime))
        
        
        sys.stdout.close()
        sys.stdout=STDOUT
        print('end learning')