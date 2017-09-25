#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 19:16:01 2017

@author: Zhan
"""
import numpy as np
import copy

class StreamingProximalSGD:

    
    def __init__(self,stepsize,batchsize, x0,lb=None,ub=None,simplexSet=None):
        
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.x = np.array(x0)

        if lb is not None:
            assert np.shape(x0) == np.shape(lb), "dimension of x does not match dimension of LB constraints"
            self.lb = np.copy(lb)
        else:
            self.lb = -np.inf * np.ones(np.shape(x0))            

        if ub is not None:
            assert np.shape(x0) == np.shape(ub), "dimension of x does not match dimension of UB constraints"
            self.ub = np.copy(ub)
        else:
            self.ub = np.inf * np.ones(np.shape(x0))           
            
        if True in (self.lb > self.ub) :
            raise ValueError("lower bound cannot greater than upper bound")

        self.simplexSet = copy.deepcopy(simplexSet)
        if self.simplexSet is not None:
            for i in self.simplexSet:
                if self.lb[i] != -np.inf :
                    raise ValueError("lower bound must be -inf for components projected onto simplex")
            
                if self.ub[i] != np.inf :
                    raise ValueError("upper bound must be inf for components projected onto simplex")

        self.batchgradient = np.zeros(np.shape(x0))
        self.numgradient = 0
    
    def psgdUpdate(self,grad):
        
        if (self.numgradient<self.batchsize):
            self.batchgradient += grad
            self.numgradient += 1
        
        if (self.numgradient == self.batchsize):    
            xnew = self.x - self.stepsize * (self.batchgradient/self.batchsize)
            xnew = self.boxProjection(xnew)
            xnew = self.simplexProjection(xnew)
            self.batchgradient = np.zeros(np.shape(self.x))
            self.numgradient = 0
            self.x = np.copy(xnew)
            
        
            
    def boxProjection(self,xbefore):
        
        for i in range(xbefore.shape[0]):
            if xbefore[i] < self.lb[i] :
                xbefore[i] = self.lb[i]
            elif xbefore[i] > self.ub[i] :
                xbefore[i] = self.ub[i]
                
        return xbefore
    
    
    def simplexProjection(self,xbefore):
        
        xafter = np.copy(xbefore)
        xInSimplex = xbefore[self.simplexSet]
        xsorted = np.copy(xInSimplex).reshape(xInSimplex.shape[0])
        xsorted[::-1].sort()
        
        pho = xsorted.shape[0]
        for i in range(xsorted.shape[0]):
            val = xsorted[i] + 1/(i+1) * (1-np.sum(xsorted[0:i+1]))
            if val <= 0:
                pho = i 
                break 
     
        shift = 1/pho * (1 - np.sum(xsorted[0:pho]))
        xprojected = np.maximum(xInSimplex+shift*np.ones(xInSimplex.shape),0)

        
        xafter[self.simplexSet] = xprojected
            
        return xafter
    
        
        
        
        
        
        
        
