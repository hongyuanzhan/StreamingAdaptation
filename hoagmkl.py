#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:35:40 2017

@author: Zhan
"""
#from bkt import RollingBackTest
import copy
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as scipylbfgs

class HOAG:
    
    def __init__(self,backtester,initialparam,
                 xTr,tTr,yTr,xVal,tVal,yVal):
        
        self.xTr = np.copy(xTr)
        self.tTr = np.copy(tTr)
        self.yTr = np.copy(yTr)
        self.xVal = np.copy(xVal)
        self.tVal = np.copy(tVal)
        self.yVal = np.copy(yVal)
        self.yhat = np.zeros(np.shape(self.yVal))
        self.bktester = copy.deepcopy(backtester)  
        self.bktester.mklregressor.training_X = np.copy(xTr)
        self.bktester.mklregressor.training_T = np.copy(tTr)
        self.bktester.mklregressor.training_Y = np.copy(yTr)

        self.lb = np.copy(backtester.lb)
        self.ub = np.copy(backtester.ub)
        self.simplexSet = np.copy(backtester.simplexSet)
        self.paramvector = np.copy(initialparam)
        
    
    def validationObjective(self,paramvector):
        
        self.bktester.updateHyperParam(paramvector)
        self.bktester.mklregressor.fit(self.xTr,self.tTr,self.yTr)
        self.bktester.mklregressor.dualWeightJacobian()

        [nval,pval] = np.shape(self.xVal)
        l2error = 0
        for i in range(nval):
            self.yhat[i] = self.bktester.mklregressor.predict(self.xVal[i,:],self.tVal[i])
            l2error += (self.yVal[i] - self.yhat[i]) ** 2
        
        return l2error/nval
    
    def validationGradient(self,paramvector):
        
        l2error = self.validationObjective(paramvector)
        #print(l2error)
        [nval,pval] = np.shape(self.xVal)
        hypergrad = np.zeros(np.shape(paramvector))
        for i in range(nval):
            hypergradDict = self.bktester.mklregressor.computeHyperGradient(self.xVal[i,:],
                                self.tVal[i],self.yhat[i],self.yVal[i])
            hypergrad += self.bktester.hypergradDict2Array(hypergradDict)

        result = np.array([hypergrad/nval,l2error])
        return result
    
    
    def hoag(self,maxit,tol,stepsize):
        
        paramvector = np.copy(self.paramvector)
        for it in range(maxit):
            
            grad = self.validationGradient(paramvector)[0]
            paramOld = np.copy(paramvector)
            paramvector -= stepsize * grad
            paramvector = self.boxProjection(paramvector)
            paramvector = self.simplexProjection(paramvector)
            print("change in iteration " + str(it) + " " + str(np.linalg.norm(paramOld - paramvector)) )
            print("validation error " + str(self.validationGradient(paramvector)[1]) )
            if np.linalg.norm(paramOld - paramvector) < tol: 
                break 
          
        self.paramvector = np.copy(paramvector)
        
        return paramvector
                      
    def boxProjection(self,paramOld):
        
        for i in range(paramOld.shape[0]):
            if paramOld[i] < self.lb[i] :
                paramOld[i] = self.lb[i]

            elif paramOld[i] > self.ub[i] :
                paramOld[i] = self.ub[i]

        return paramOld
    
    
    def simplexProjection(self,paramOld):
        
        paramAfter = np.copy(paramOld)
        paramInSimplex = paramOld[self.simplexSet]
        paramSorted = np.copy(paramInSimplex).reshape(paramInSimplex.shape[0])
        paramSorted[::-1].sort()
        pho = paramSorted.shape[0]
        for i in range(paramSorted.shape[0]):
            val = paramSorted[i] - 1/(i+1) * np.sum(paramSorted[0:i+1]) + 1/(i+1)
            
            if val <= 0:
                pho = i 
                break 
     
        shift = 1/pho * (1 - np.sum(paramSorted[0:pho]))
        paramProjected = np.maximum(paramInSimplex+shift*np.ones(paramInSimplex.shape),0)

        
        paramAfter[self.simplexSet] = paramProjected
            
        return paramAfter
    
 
    
        
        
        
        