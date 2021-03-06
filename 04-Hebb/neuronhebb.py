# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:01:25 2017

@author: PiotrTutak
"""

import neuron
import numpy as np
from itertools import zip_longest

class NeuronHebb(neuron.Neuron):
    def __init__(self, weights, activFunc, learnRate=0.1,forgetRate=0.1, bias=-0.5):
        self.__dict__['_weights']=np.array(weights)
        self.__dict__['_learnRate']=learnRate
        self.__dict__['_activFunc']=activFunc
        self.__dict__['_bias']=bias
        self.__dict__['_error']=None
        self.__dict__['_inputValues']=None
        self.__dict__['_val']=None
        self.__dict__['_output']=None
        self.__dict__['_forgetRate']=forgetRate
    def learnHebb(self):
        if (self._learnRate):
            output=self._output
            for i in range(len(self._weights)):
                self._weights[i]*=1.0-self._forgetRate
                self._weights[i]+=self._learnRate*output*self._inputValues[i]
            self.__dict__['_bias']*=1.0-self._forgetRate
            self.__dict__['_bias']+=self._learnRate*output
    def __getitem__(self,index):
        if index=='error':
            return self._error
        elif index=='input':
            return self._inputValues
        elif index=='value':
            return self._val
        elif index=='learnRate':
            return self._learnRate
        elif index=='forgetRate':
            return self._forgetRate
        return self._weights[index]
    def __setitem__(self,index,value):
        if index=='learnRate':
            self.__dict__['_learnRate']=value
        elif index=='activFunc':
            self.__dict__['_activFunc']=value
        elif index=='forgetRate':
            self.__dict__['_forgetRate']=value
    def __repr__(self):
        w='['+','.join('{:8.5f}'.format(x) for x in self._weights)+']'
        return 'NeuronHebb(weights:{0},bias:{1:8.5f},learnRate:{2:.5f},forgetRate:{4:.5f},activFunc:{3!s})'.format(w,self._bias,self._learnRate,self._activFunc.__name__,self._forgetRate)

class LayerHebb(neuron.Layer):
    """
    Klasa wartstwy używana w wielowarstwowej sieci neuronowej.
    """
    def __init__(self,inputNumber,neuronNumber,activFunc,weights=None,learnRate=None, forgetRate=None,bias=None):
        self.__dict__['_inputNumber']=inputNumber
        self.__dict__['_neuronNumber']=neuronNumber
        self.__dict__['_activFunc']=activFunc
        
        if weights!=None:
            _weights=list(weights)
            if inputNumber>len(_weights):
                _weights.extend([(np.random.ranf()*np.random.ranf())*np.random.choice([-1.0,1.0]) for _ in range(inputNumber-len(_weights))])
        else:
            _weights=None
        
        if learnRate!=None:
            self.__dict__['_learnRate']=learnRate
        else:
            self.__dict__['_learnRate']=0.1
        if forgetRate!=None:
            self.__dict__['_forgetRate']=forgetRate
        else:
            self.__dict__['_forgetRate']=0.1
            
        _bias=bias
        if _weights:
            if _bias!=None:
                self.__dict__['_neurons']=[NeuronHebb(_weights[:inputNumber],activFunc,learnRate=self._learnRate,forgetRate=self._forgetRate,bias=_bias) for _ in range(neuronNumber)]
            else:
                self.__dict__['_neurons']=[NeuronHebb(_weights[:inputNumber],activFunc,learnRate=self._learnRate,forgetRate=self._forgetRate,bias=np.random.ranf()*np.random.ranf()*2.0-1.0) for _ in range(neuronNumber)]
        else:
            if _bias!=None:
                self.__dict__['_neurons']=[NeuronHebb([(np.random.ranf()*np.random.ranf())*np.random.choice([-1.0,1.0]) for _ in range(inputNumber)],activFunc,learnRate=self._learnRate,forgetRate=self._forgetRate,bias=_bias) for _ in range(neuronNumber)]
            else:
                self.__dict__['_neurons']=[NeuronHebb([(np.random.ranf()*np.random.ranf())*np.random.choice([-1.0,1.0]) for _ in range(inputNumber)],activFunc,learnRate=self._learnRate,forgetRate=self._forgetRate,bias=np.random.ranf()*np.random.ranf()*2.0-1.0) for _ in range(neuronNumber)]
    def __repr__(self):
        result='Layer(inputNumber:{0}, neuronNumber:{1}, activFunc:{2!s},learnRate:{3:.5f},forgetRate:{3:.5f})'\
              .format(self._inputNumber,self._neuronNumber,self._activFunc.__name__,self._learnRate,self._forgetRate)
        return result
    def __setitem__(self,index,value):
        if index=='learnRate':
            self.__dict__['_learnRate']=value
            for x in self._neurons:
                x['learnRate']=value
        elif index=='forgetRate':
            self.__dict__['_forgetRate']=value
            for x in self._neurons:
                x['forgetRate']=value
        elif index=='activFunc':
            self.__dict__['_activFunc']=value
            for x in self._neurons:
                x['activFunc']=value

class MultilayerHebb(neuron.Multilayer):
    """
    Wielowarstwa z możliwoscią zaprogramwania indywidualnie każdej wartstwy
    """
    def __init__(self,layers,activFuncs=None,weights=[], learnRates=[],forgetRates=[], biases=[], batchSize=1):
        self._batchSize=batchSize
        if isinstance(layers[0],LayerHebb):
            self._layers=layers
        elif isinstance(layers[0],int):
            if not all([activFuncs]):
                raise TypeError('Missing activation functions')
            neuronNumbers=layers
            l=zip_longest(neuronNumbers,activFuncs,weights,learnRates,forgetRates,biases,fillvalue=None)
            prev=next(l)
            layerList=[LayerHebb(1,*prev)]
            for x in l:
                layerList.append(LayerHebb(prev[0],*x))
                prev=x
            self._layers=layerList
    def learnHebb(self,inputValues):
        results=self.process(inputValues)
        for layer in self._layers[1:]:
            for p in layer:
                p.learnHebb()
        return results
    def setForgetRates(self,value):
        for l in self._layers:
            l['forgetRate']=value