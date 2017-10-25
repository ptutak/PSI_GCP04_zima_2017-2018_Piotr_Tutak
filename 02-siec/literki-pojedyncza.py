# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:20:52 2017

@author: PiotrTutak
"""

from perceptron import *
from operator import itemgetter
import random
import time
import sys
import copy
#funkcja wypisująca zawartosc listy z zadaną precyzją
def listWithPrec(listA,prec):
    ret="["
    formatStr="{0: "+str(int(prec+3))+"."+str(int(prec))+"f}"
    for x in listA:
        ret+=formatStr.format(x)
        ret+=","
    ret=ret[:-1]+']'
    return ret

#funkcje liczace wartosci błędów
def MSE(results,expected):
    sum=0.0
    for i in range(len(results)):
        sum+=(results[i]-expected[i])**2
    return sum/len(results)

def MAPE(results,expected):
    sum=0.0
    for i in range(len(results)):
        sum+=abs((expected[i]-results[i])/expected[i])
    return 100*sum/len(results)

#przekierowanie wyjscia do pliku
STDOUT=sys.stdout
f=open('results.txt','w');
sys.stdout=f


literyLow=dict()
literyHigh=dict()
with open('litery.txt','r') as f:
    for l in f:
        x=l.strip().split('=')
        if x[0].islower():
            literyLow[x[0]]=[float(a) for a in x[1]]
        else:
            literyHigh[x[0]]=[float(a) for a in x[1]]

#utworzenie danych uczacych i testujacych
litery20=sorted(random.sample(list(literyLow.items()),10),key=itemgetter(0))
litery20.extend(sorted(random.sample(list(literyHigh.items()),10),key=itemgetter(0)))
litery20Expected=[[1.0] if x >9 else [0.0] for x in range(20)]
litery20ExpectedTest=[0.0 if x<=9 else 1.0 for x in range(20)]

litery20Stirred=copy.deepcopy(litery20)

#liczba blednych pikseli w literze
NUMBER_OF_STIRR=5

for x in litery20Stirred:
    change=random.sample(set(range(35)),NUMBER_OF_STIRR)
    for c in change:
        if x[1][c]==0.0:
            x[1][c]=1.0
        else:
            x[1][c]=0.0

print('użyte litery:')
print(*list(x[0] for x in litery20),sep=' ')
print('\n')

for x in litery20:
    for j in range(5):
        for i in range(7):
            if x[1][j*7+i]:
                print(str(int(x[1][j*7+i]))+' ',end='')
            else:
                print('  ',end='')
        print('')
    print('\n')


listPerc=[]
RES_NUMBER=4

#przyjeta liczba perceptronow w warstwie
HIDDEN_LAYER_PERCEP_NUMB=15
learnRate=0.5

#uczenie poszczegolnych warstw
while(len(listPerc)<RES_NUMBER):
    if len(listPerc)==2:
        learnRate=0.1
    if len(listPerc)%2==0:
        multilayer=Multilayer(
                 [35,HIDDEN_LAYER_PERCEP_NUMB,1],
                 [hardOne,SignSigm()(1.0),hardOne],
                 [zero,SignSigm().derivative(1.0),one],
                 [[1.0],None,[1.0 for x in range(HIDDEN_LAYER_PERCEP_NUMB)]],
                 [0.0,learnRate,0.0],
                 [-0.5,-0.05,0.0]
                 )
    else:
        multilayer=Multilayer(
                 [35,HIDDEN_LAYER_PERCEP_NUMB,1],
                 [hardOne,SignSigm()(1.0),ident],
                 [zero,SignSigm().derivative(1.0),one],
                 [[1.0],None,[1.0 for x in range(HIDDEN_LAYER_PERCEP_NUMB)]],
                 [0.0,learnRate,0.0],
                 [-0.5,-0.05,0.0]
                 )
        
    i=0
    run=True
    start=time.clock()
    print('start learning:')
    print('iteration;  time  ;  error   ; learn rate: %f;'% (learnRate))
    while(run):
        samples=list(litery20)
        while(run and samples):
            inp=random.sample(samples,1).pop(0)
            samples.remove(inp)
            multilayer.learn(inp[1],litery20Expected[litery20.index(inp)])
        
        results=[]
        for inp in litery20:
            results.extend(multilayer.process(inp[1]))
         
        error=MSE(results,litery20ExpectedTest)
        if error<0.0001:
            run=False    
        i+=1
        print("{0:9};{1: 8.5f}".format(i,time.clock()-start),error,sep=';')
    listPerc.append((multilayer,i,error,time.clock()-start))
    print("iter number:{0:8}".format(i),"; time taken[s]:{0:8}".format(time.clock()-start))
    print('')


print('\n\nTestowanie sieci:')
print('Dane zaszumione w liczbie %d pikseli' % NUMBER_OF_STIRR)

for x in litery20Stirred:
    for j in range(5):
        for i in range(7):
            if x[1][j*7+i]:
                print(str(int(x[1][j*7+i]))+' ',end='')
            else:
                print('  ',end='')
        print('')
    print('\n')

#testowanie warstw na danych zaszumionych
for m in listPerc:
    res=[]
    errorCompute=[]
    print(repr(m[0]),end='')
    print("iter number:{0:8}".format(m[1]),"; time taken[s]:{0:8}".format(m[3]))
    print('errors:\n(letter, result, expected):')
    for s in litery20Stirred:
        res.append(m[0].process(s[1]))
        errorCompute.extend(m[0].process(s[1]))
    res=[(x[0],*y,*z) for x,y,z in zip(litery20,res,litery20Expected) if z[0]!=y[0]]
    print(*res,sep='\n')
    print('number of errors:',len(res))
    print('error value MSE:',MSE(errorCompute,litery20ExpectedTest))
    print('\n')
    
    
f.close()
sys.stdout=STDOUT
