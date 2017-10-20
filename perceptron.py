# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 19:39:26 2017

@author: PiotrTutak
"""

import numpy as np
from itertools import zip_longest
from operator import itemgetter
import random
import sys

"""
Różne funkcje aktywacji używane w testowaniu perceptronu: 
"""

def ident(x):
    return float(x)

def positiveIdent(x):
    if x<0:
        return 0.0
    return float(x)

def one(x):
    return 1.0

def zero(x):
    return 0.0

class Const:
    def __call__(self,alfa):
        def const(x):
            return float(alfa)
        return const


def hardOne(x):
    if x<0:
        return 0.0
    return 1.0


def hardSign(x):
    if x<0:
        return -1.0
    return 1.0
    

def squash(x):
    if x<-1:
        return -1.0
    elif x>1:
        return 1.0
    return x


class Sigm:
    def __call__(self,alfa):
        def sigm(x):
            return 1.0/(1.0+np.exp(-alfa*x))
        return sigm
    def derivative(self,alfa):
        def sigmDeriv(x):
            return alfa*np.exp(-alfa*x)/((1.0+np.exp(-alfa*x))**2)
        return sigmDeriv

class SignSigm:
    def __call__(self,alfa):
        def sigm(x):
            return (2.0/(1.0+np.exp(-alfa*x)))-1.0
        return sigm
    def derivative(self,alfa):
        def sigmDeriv(x):
            return 2.0*alfa*np.exp(-alfa*x)/((1.0+np.exp(-alfa*x))**2)
        return sigmDeriv



class Perceptron:
    """
    Klasa Perceptron
    """
    def __init__(self, weights, activFunc, activFuncDeriv, learnRate=0.1, bias=-0.8*np.random.ranf()-0.1):
        self.__dict__['_weights']=np.array(weights)
        self.__dict__['_learnRate']=learnRate
        self.__dict__['_activFunc']=activFunc
        self.__dict__['_activFuncDeriv']=activFuncDeriv
        self.__dict__['_bias']=bias
        self.__dict__['_error']=None
        self.__dict__['_inputValues']=None
        self.__dict__['_val']=None
    
    def process(self,inputValues):
        """
        Funkcja przetwarzająca dane wejsciowe na dane wyjsciowe
        """
        if len(inputValues)!=len(self._weights):
            raise TypeError('Wrong values length')
        self.__dict__['_inputValues']=np.array(inputValues)
        self.__dict__['_val']=np.dot(self._weights,self._inputValues)+self._bias
        return self._activFunc(self._val)
    
    def propagateError(self,weights,errors):
        """
        Funkcja propagująca błąd i korygująca wagi oraz bias
        """
        weights=np.array(weights)
        errors=np.array(errors)
        if len(errors)!=len(weights):
            raise TypeError('Wrong values length')
        self.__dict__['_error']=np.dot(weights,errors)*self._activFuncDeriv(self._val)
        if (self._learnRate):
            for i in range(len(self._weights)):
                self._weights[i]+=self._learnRate*self._error*self._inputValues[i]
            self.__dict__['_bias']+=self._learnRate*self._error
        return self._error
    """
    Funkcje dostępowe:
    """
    def __setitem__(self,index,value):
        if index=='learnRate':
            self.__dict__['_learnRate']=value
    
    def __getitem__(self,index):
        if index=='error':
            return self._error
        elif index=='input':
            return self._inputValues
        elif index=='value':
            return self._val
        elif index=='learnRate':
            return self._learnRate
        return self._weights[index]
    
    def __getattr__(self,attr):
        raise AttributeError('get: No such attribute: %r'%attr)
    
    def __setattr__(self,attr,value):
        raise AttributeError('set: No such attribute: %r'%attr)
    
    def __iter__(self):
        return iter(self._weights)
    
    def __len__(self):
        return len(self._weights)
    
    def __repr__(self):
        w='['+','.join('{:8.5f}'.format(x) for x in self._weights)+']'
        return 'Perceptron(weights:{0},bias:{1:8.5f},learnRate:{2:.5f},activFunc:{3!s})'.format(w,self._bias,self._learnRate,self._activFunc.__name__)


class Layer:
    """
    Klasa wartstwy używana w wielowarstwowej sieci neuronowej.
    """
    def __init__(self,inputNumber,percepNumber,activFunc,activFuncDeriv,weights=None,learnRate=None,bias=None):
        self.__dict__['_inputNumber']=inputNumber
        self.__dict__['_percepNumber']=percepNumber
        self.__dict__['_activFunc']=activFunc
        self.__dict__['_activFuncDeriv']=activFuncDeriv
        
        if weights!=None:
            _weights=list(weights)
            if inputNumber>len(_weights):
                _weights.extend([np.random.ranf()*np.random.choice([-1.0,1.0]) for _ in range(inputNumber-len(_weights))])
        else:
            _weights=[np.random.ranf()*np.random.choice([-1.0,1.0]) for _ in range(inputNumber)]
        
        if learnRate!=None:
            self.__dict__['_learnRate']=learnRate
        else:
            self.__dict__['_learnRate']=0.1
        
        if bias!=None:
            _bias=bias
        else:
            _bias=-0.8*np.random.ranf()-0.1
        
        self.__dict__['_perceptrons']=[Perceptron(_weights[:inputNumber],activFunc,activFuncDeriv,bias=_bias,learnRate=self._learnRate) for _ in range(percepNumber)]
    """
    Funkcje dostępowe
    """
    def __len__(self):
        return len(self._perceptrons)
    def __getitem__(self,index):
        if index=='learnRate':
            return self._learnRate
        return self._perceptrons[index]
    def __iter__(self):
        return iter(self._perceptrons)
    def __setitem__(self,index,value):
        if index=='learnRate':
            self.__dict__['_learnRate']=value
            for x in self._perceptrons:
                x['learnRate']*=value
    def __getattr__(self,attr):
        raise AttributeError('get: No such attribute: %r'%attr)
    
    def __setattr__(self,attr,value):
        raise AttributeError('set: No such attribute: %r'%attr)
        
    def __repr__(self):
        result='Layer(inputNumber:{0}, perceptronNumber:{1}, activFunc:{2!s}, activFuncDeriv:{3!s}, learnRate:{4:.5f})'\
              .format(self._inputNumber,self._percepNumber,self._activFunc.__name__,self._activFuncDeriv.__name__,self._learnRate)
        return result
    def __str__(self):
        result=repr(self)+'\n'
        for p in self:
            result+='        '+str(p)+'\n'
        return result



class Multilayer:
    """
    Wielowarstwa z możliwoscią zaprogramwania indywidualnie każdej wartstwy
    """
    def __init__(self,layers,activFuncs=None,activFuncDerivs=None,weights=[], learnRates=[], biases=[]):
        if isinstance(layers[0],Layer):
            self._layers=layers
        elif isinstance(layers[0],int):
            if not all([activFuncs,activFuncDerivs]):
                raise TypeError('Missing activation functions or derivatives')
            percepNumbers=layers
            l=zip_longest(percepNumbers,activFuncs,activFuncDerivs,weights,learnRates,biases,fillvalue=None)
            prev=next(l)
            layerList=[Layer(1,*prev)]
            for x in l:
                layerList.append(Layer(prev[0],*x))
                prev=x
            self._layers=layerList
    def process(self,inputValues):
        """
        Funkcja przetwarzająca dane wejsciowe sieci na dane wyjsciowe
        """
        inputValues=list(inputValues)
        values=[]
        for p in self._layers[0]:
            values.append(p.process(inputValues[:len(p)]))
            inputValues=inputValues[len(p):]    
        for layer in self._layers[1:]:
            inputValues=values
            values=[]
            for p in layer:
                values.append(p.process(inputValues))
        return values
        
    def learn(self,inputValues,expectedValues):
        """
        Funkcja ucząca sieć neuronową po uprzednim nadaniu współczynników uczenia
        dla każdej wartwy
        """
        results=iter(self.process(inputValues))
        i=len(expectedValues)
        expectedValues=iter(expectedValues)
        errors=[]
        for _ in range(i):
            errors.append([next(expectedValues)-next(results)])
        weights=[[1] for _ in range(len(errors))]
        for layer in reversed(self._layers):
            newErrors=[]
            oldWeights=[]
            for p in layer:
                oldWeights.append(p[:])
                newErrors.append(p.propagateError(weights.pop(0),errors.pop(0)))
            weights=list(zip(*oldWeights))
            errors=[newErrors for x in range(len(weights))]
    """
    Funkcje dostępowe
    """
    def __getitem__(self,index):
        return self._layers[index]
    def __iter__(self):
        return iter(self._layers)
    def __repr__(self):
        result='Multilayer:\n'
        for layer in self._layers:
            result+='    '+repr(layer)
            result+='\n'
        return result
    def __str__(self):
        result='Multilayer:\n'
        for layer in self._layers:
            result+='    '+str(layer)
            result+='\n'
        return result
        
                

if __name__=='__main__':
    """
    Kod programu przeprowadzającego uczenie i testowanie perceptronu
    Wyjscie jest przekierowywane do pliku results.txt
    """
    STDOUT=sys.stdout
    f=open('results.txt','w');
    sys.stdout=f
    
    SigmFactory=SignSigm()
    print('Funkcja AND:')
    inputData=(
            ((0,0),0),
            ((0,1),0),
            ((1,0),0),
            ((1,1),1)
            )
    for x in inputData:
        print("data: {0}, expected: {1}".format(*x))
    
    listPerc=[]
    RES_NUMBER=100
    while(len(listPerc)<RES_NUMBER):
        w=[np.random.ranf()*np.random.choice([-1,1]) for _ in range(2)]
        p=Perceptron(w,hardOne,one,learnRate=np.random.ranf()*np.random.ranf()*np.random.ranf(),bias=np.random.ranf()*-1.0)
        i=0
        run=True
        print(p)
        print('iteration;bad')
        while(run):
            samples=list(inputData)
            while(run and samples):
                i+=1
                inp=random.sample(samples,1).pop(0)
                expected=inp[1]
                result=p.process(inp[0])
                p.propagateError([1],[expected-result])
                bad=0
                for data,expected in inputData:
                    r=p.process(data)
                    if r!=expected:
                        bad+=1
                if bad==0:
                    run=False
                print(i,bad,sep=';')
        print(p)

        #w=('initialWeights:['+','.join('{:8.5f}'.format(x) for x in w)+']')
        
        listPerc.append((w,p[:],p['learnRate'],i))
    for x in sorted(listPerc,key=itemgetter(2)):
        print(*x,sep=';')
    sys.stdout=STDOUT
    f.close()
    """
    #Kod testujący funkcję XOR na sieci neuronowej:
        
    
    print('\n\nFunkcja XOR:')
    xorInputData=(
            ((0,0),[0]),
            ((0,1),[1]),
            ((1,0),[1]),
            ((1,1),[0])
            )
    for x in xorInputData:
        print('data: ',x[0],', expected: ',*x[1])

    listPercMin=[]
    listPercAver=[]
    listPercMax=[]
    RES_NUMBER=100
    while(len(listPercMin)+len(listPercAver)+len(listPercMax)<RES_NUMBER):
        mult=Multilayer([2,2,1],[hardSign,SignSigm()(1.0),hardOne],[zero,SignSigm().derivative(1.0),one],weights=[[1.0]],learnRates=[0.0,3.0,.01],biases=[-0.5])
        i=0
        cont=True
        while(cont):
            cont=False
            i+=1
            inp=random.choice(xorInputData)
            mult.learn(*inp)
            for data,expected in xorInputData:
                r=mult.process(data)
                if r[0]!=expected[0]:
                    cont=True
                    break
            if i>5000:
                cont=False
                break
        if i<10 and len(listPercMin)<RES_NUMBER:
            listPercMin.append((mult,"iterNumber: %d;"%i))
            print(i)
        elif i>=10 and i<100 and len(listPercAver)<RES_NUMBER:
            listPercAver.append((mult,"iterNumber: %d;"%i))
            print(i)
        elif i>=100 and len(listPercMax)<RES_NUMBER:
            listPercMax.append((mult,"iterNumber: %d;"%i))
            print(i)
    print('\n------------ percentage ------------\n')
    print('min iter: ',len(listPercMin),'; aver iter: ',len(listPercAver),'; max iter: ',len(listPercMax))
    print('\n\n------------ min iter number ------------')
    for x in listPercMin:
        print(x[1],end=' ')
    print('\n\n------------ average iter number ------------')
    for x in listPercAver:
        print(x[1],end=' ')
    print('\n\n------------ max iter number ------------')
    for x in listPercMax:
        print(x[1],end=' ')
    print('\n')
    print(mult)
    """