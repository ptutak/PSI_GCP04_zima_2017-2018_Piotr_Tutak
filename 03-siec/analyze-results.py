# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:21:32 2017

@author: PiotrTutak
"""

import os
import re

pattern=re.compile(r"- .+step - loss: \d+\.\d+ - acc: \d+\.[\de\+]+")


for file in os.listdir("."):
    if file.startswith("results") and file.endswith(".txt"):
        print(os.path.join(".", file))
        with open(file,'r') as f:
            i=1
            outputFile="processed-"+file
            with open(outputFile,"w") as fOut:
                for line in f:
                    out=pattern.search(line)
                    if out:
                        print("epoch: {0} out: {1}".format(i,out.group(0)),file=fOut)
                        i+=1
            