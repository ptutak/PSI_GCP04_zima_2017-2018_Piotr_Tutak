# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:21:32 2017

@author: PiotrTutak
"""

import os
import re
import matplotlib.pyplot as plt

pattern=re.compile(r"- .+step - loss: (\d+\.\d+) - acc: \d+\.[\de\+]+")
pattern_filename=re.compile(r"\[.+\]-lr-[\d\.]+")
fileList=[]
for file in os.listdir("."):
    if file.startswith("results") and file.endswith(".txt"):
        print(os.path.join(".", file))
        with open(file,'r') as f:
            i=1
            outputFile="processed-"+file
            fileList.append(outputFile)
            with open(outputFile,"w") as fOut:
                print(pattern_filename.search(file).group(0),file=fOut)
                for line in f:
                    out=pattern.search(line)
                    if out:
                        print("{0}".format(out.group(1)),file=fOut)
                        i+=1
