# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


printData=dict()
colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

printData['0.1']=dict()
printData['0.01']=dict()
printData['0.1']['0.3']=dict([('abc',[0,1,2,3]),('cde',[0,15,25,35])])
printData['0.1']['0.5']=dict([('abc',[0,10,20,30]),('cde',[0,11,12,13])])

printData['0.01']['0.3']=dict([('abc',[0,31,32,33]),('cde',[0,19,92,39])])
printData['0.01']['0.5']=dict([('abc',[0,14,24,34]),('cde',[0,17,72,73])])

f,axarr=plt.subplots(len(printData),len(printData[list(printData.keys()).pop(0)]),sharex=True,sharey=True)

i=0
j=0
for f,lr in printData.items():
    for l,ld in lr.items():
        axarr[i, j].set_title('fr:'+f+' lr:'+l)
        for b,d in ld.items():
            nazwa=b+" fr:"+f+" lr:"+l
            print(nazwa)
            # Four axes, returned as a 2-d array
            axarr[i, j].plot(range(4),d)
        j+=1
    i+=1

plt.show()

"""
legend=[]

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
"""