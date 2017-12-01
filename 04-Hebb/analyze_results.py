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
printData=dict()
for file in os.listdir("."):
    if file.startswith("hebb") and file.endswith(".txt"):
        print(os.path.join(".", file))
        res=pattern_filename.search(file)
        lr=res.group(1)
        fr=res.group(2)
        resVal=dict()
        if fr not in printData:
            printData[fr]=dict()
        with open(file,'r') as f:
            for line in f:
                line=line.strip().split()
                for i in range(len(line)//2-1):
                    x=line.pop(0)
                    y=float(line.pop(0))
                    if x not in resVal:
                        resVal[x]=[]
                        resVal[x].append(y)
                    else:
                        resVal[x].append(y)
        printData[fr][lr]=resVal
#print(printData)

f,axarr=plt.subplots(len(printData),len(printData[list(printData.keys()).pop(0)]))
f.set_size_inches(180,120)

i=0
j=0
for f,lr in printData.items():
    j=0
    for l,ld in lr.items():
        axarr[i,j].set_title('fr:'+f+' lr:'+l+' [i:{0},j:{1}]'.format(i,j))
        axarr[i,j].set_ylim(-1.05,1.05)
        axarr[i,j].set_xlim(-5,105)
        for b,d in ld.items():
            nazwa=b+" fr:"+f+" lr:"+l
            print('plotting:',nazwa)
            axarr[i, j].plot(range(100),d,label=b)
        legend=axarr[i,j].legend(loc="upper right")
        j+=1
    i+=1

plt.savefig("resultsPlot.png")
#plt.show()
