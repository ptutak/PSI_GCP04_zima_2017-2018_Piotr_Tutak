# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 07:29:02 2017

@author: PiotrTutak
"""

import os
import re
import matplotlib.pyplot as plt

pattern_filename=re.compile(r"-lr-(\d\.[\d]+)-fr-(\d\.[\d]+)")
fileList=[]
colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
printDict=dict()
for file in os.listdir("."):
    if file.startswith("hebb") and file.endswith(".txt"):
        print(os.path.join(".", file))
        res=pattern_filename.search(file)
        lr=res.group(1)
        fr=res.group(2)
        resVal=dict()
        if fr not in printDict:
            printDict[fr]=dict()
        with open(file,'r') as f:
            for line in f:
                line=line.strip().split()
                for i in range(len(line)//2):
                    x=line.pop(0)
                    y=float(line.pop(0))
                    if x not in resVal:
                        resVal[x]=[]
                    else:
                        resVal[x].append(y)
        printDict[fr][lr]=resVal
print(printDict)


i=0
legend=[]
for f,_ in printDict.items():
    plt.figure(figsize=(20,40))
    plt.figure(1)
    plt.subplot(211)
    for l,d in _.items():
        for v,vl in d.items():
            
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


                