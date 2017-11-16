# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:29:12 2017

@author: PiotrTutak
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
data01={}
data001={}
for file in os.listdir("."):
    if file.startswith("processed") and file.endswith(".txt"):
        print(os.path.join(".", file))

        with open(file,'r') as f:
            etiq=f.readline().strip()
            if file.endswith("0.01-decay-0.0.txt"):
                data001[etiq]=[]
                dataOut=data001[etiq]
            else:
                data01[etiq]=[]
                dataOut=data01[etiq]
            for line in f:
                dataOut.append(float(line.strip()))


colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
i=0
legend=[]
plt.figure(figsize=(20,40))
plt.figure(1)
plt.subplot(211)
for k,v in data001.items():
    line,=plt.plot(list(range(len(v))),v,colors[i],linewidth=0.5,label=k)
    legend.append(mpatches.Patch(color=colors[i],label=k))
    i+=1
plt.legend(handles=legend)
plt.axis([0,100000,0,120])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Zaleznosc bledu od epoki')

legend=[]
i=0
plt.subplot(212)
for k,v in data01.items():
    line,=plt.plot(list(range(len(v))),v,colors[i],linewidth=0.5,label=k)
    legend.append(mpatches.Patch(color=colors[i],label=k))
    i+=1
plt.legend(handles=legend)
plt.axis([0,100000,0,120])
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Zaleznosc bledu od epoki')


plt.savefig("testPlot.png")
plt.show()

