#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:50:59 2017

@author: Zhan
"""

import numpy as np
from mklr import MKLregression
from mklr import SqExpKernel
from mklr import PeriodicKernel
from mklr import OUprocessKernel
from mklr import LinearKernel
from mklr import ARDKernel
from stmpsgd import OPGD
from hoagmkl import HOAG
import copy
from timeit import default_timer as timer

class RollingBackTest:
    
    
    def __init__(self,timeSeriesData, order, trainingWindow, horizon,
                 retrainPeriod, hyperParameterUpdateOption, selectionPeriod, 
                 lambda_reg, regLB = None, regUB = None,
                 squareExponentialKernel=None, sqExpScaleLB = None, sqExpScaleUB=None,
                 automaticrelavanceKernel=None, autoRelevanceLB = None, autoRelevanceUB=None,
                 periodicityKernel=None, quasiperiodic=False, prdScaleLB=None, prdScaleUB=None, prdPeriodLB=None, prdPeriodUB=None,
                 ouKernel=None,  ouScaleLB=None, ouScaleUB=None,
                 linearKernel = None, 
                 ensembleWeight=None):
                 #ensembleWeight=None, ensembleWeightLB=None, ensembleWeightUB=None):

        
        self.timeSeriesData = np.copy(timeSeriesData)
        self.timeSeriesLength = self.timeSeriesData.shape[0]
        self.order = np.copy(order)
        # number of trainign data
        self.trainingWindow = np.copy(trainingWindow)
        # number of steps ahead, must be at least 1 
        self.horizon = np.copy(horizon)
        # frequency of training
        self.retrainPeriod = retrainPeriod
        # hyperparameter update option
        hpOptionSet = ['sgd','fixed','hoag','randomsearch']
        if not(hyperParameterUpdateOption in hpOptionSet):
            print("hyperparameter selection options are: 'fixed' or 'sgd', using 'fixed' by default")
            self.hyperParameterUpdateOption = 'sgd'
        
        self.hyperParameterUpdateOption = hyperParameterUpdateOption
        # frequency of hyperparameter reselection
        self.selectionPeriod = selectionPeriod
        # input ensemble weight
        self.ensembleWeight = np.copy(ensembleWeight)
        # construct MKL regressor
        self.mklregressor = MKLregression(order, lambda_reg, 
                                     squareExponentialKernel=squareExponentialKernel, 
                                     automaticrelavanceKernel=automaticrelavanceKernel, 
                                     periodicityKernel=periodicityKernel,quasiperiodic=quasiperiodic, 
                                     ouKernel=ouKernel, 
                                     linearKernel = linearKernel, 
                                     ensembleWeight=ensembleWeight) 

        
        # LB and UB used for gradient descend
        initialHyperParameters = np.array([])
        self.parameterIndex = {}
        simplexSet = np.array([],dtype = np.intp)
        lb = np.array([])
        ub = np.array([])
        numberParameters = 0
        # initialization , LB, UB for regularizer
        initialHyperParameters = np.append(initialHyperParameters,lambda_reg)
        self.parameterIndex.update({'Regularizer':numberParameters})
        numberParameters += 1
        # lower bound for regularization hyperparameter
        if regLB is not None:
            lb = np.append(lb, regLB)
        else:
            lb = np.append(lb, -np.inf)
        # upper bound for regularization hyperparameter
        if regUB is not None:
            ub = np.append(ub,regUB)
        else:
            ub = np.append(ub, np.inf)
        
        
        if squareExponentialKernel is not None:
            initialHyperParameters = np.append(initialHyperParameters,squareExponentialKernel.kernelscale)
            self.parameterIndex.update({'SQEXP_KernelScale':numberParameters})
            numberParameters += 1
            initialHyperParameters = np.append(initialHyperParameters,self.mklregressor.weight_squareExponentialKernel)
            self.parameterIndex.update({'SQEXP_EnsembleWeight':numberParameters})
            numberParameters += 1
            simplexSet = np.append(simplexSet,self.parameterIndex.get('SQEXP_EnsembleWeight'))
            print("construct square exponential kernel with scale: " + str(squareExponentialKernel.kernelscale) )
            
            if sqExpScaleLB is not None:
                lb = np.append(lb,sqExpScaleLB)
            else:
                lb = np.append(lb,-np.inf)

            if sqExpScaleUB is not None:
                ub = np.append(ub,sqExpScaleUB)
            else:
                ub = np.append(ub,np.inf)
            
            # bounds for ensemble weights, implicit simplex constraints applied
            lb = np.append(lb,-np.inf)
            ub = np.append(ub,np.inf)                    

            
        if periodicityKernel is not None:
            initialHyperParameters = np.append(initialHyperParameters,periodicityKernel.kernelscale)
            self.parameterIndex.update({'PRD_KernelScale':numberParameters})
            numberParameters += 1
            initialHyperParameters = np.append(initialHyperParameters,periodicityKernel.period)
            self.parameterIndex.update({'PRD_Period':numberParameters})
            numberParameters += 1          
            initialHyperParameters = np.append(initialHyperParameters,self.mklregressor.weight_periodicityKernel)
            self.parameterIndex.update({'PRD_EnsembleWeight':numberParameters})
            numberParameters += 1  
            simplexSet = np.append(simplexSet,self.parameterIndex.get('PRD_EnsembleWeight'))
            print("construct periodicity kernel with scale: " + str(periodicityKernel.kernelscale) + ", period: " + str(periodicityKernel.period) )
            
            if prdScaleLB is not None:
                lb = np.append(lb,prdScaleLB)
            else:
                lb = np.append(lb,-np.inf)

            if prdScaleUB is not None:
                ub = np.append(ub,prdScaleUB)
            else:
                ub = np.append(ub,np.inf)
 
            if prdPeriodLB is not None:
                lb = np.append(lb,prdPeriodLB)
            else:
                lb = np.append(lb,-np.inf)

            if prdPeriodUB is not None:
                ub = np.append(ub,prdPeriodUB)
            else:
                ub = np.append(ub,np.inf)  
                
            # bounds for ensemble weights, implicit simplex constraints applied                
            lb = np.append(lb,-np.inf)
            ub = np.append(ub,np.inf)                

        if ouKernel is not None:
            
            initialHyperParameters = np.append(initialHyperParameters,ouKernel.kernelscale)
            self.parameterIndex.update({'OU_KernelScale':numberParameters})
            numberParameters += 1              
            initialHyperParameters = np.append(initialHyperParameters,self.mklregressor.weight_ouKernel) 
            self.parameterIndex.update({'OU_EnsembleWeight':numberParameters})
            numberParameters += 1               
            simplexSet = np.append(simplexSet,self.parameterIndex.get('OU_EnsembleWeight'))
            print("construct OU process kernel wit scale: " + str(ouKernel.kernelscale))
            
            if ouScaleLB is not None:
                lb = np.append(lb,ouScaleLB)
            else: 
                lb = np.append(lb,-np.inf)

            if ouScaleUB is not None:
                ub = np.append(ub,ouScaleUB)
            else: 
                ub = np.append(ub,np.inf)

            # bounds for ensemble weights, implicit simplex constraints applied   
            lb = np.append(lb,-np.inf)
            ub = np.append(ub,np.inf)                    
     
        if linearKernel is not None:
            
            initialHyperParameters = np.append(initialHyperParameters,self.mklregressor.weight_linearKernel)
            self.parameterIndex.update({'Linear_EnsembleWeight':numberParameters})
            numberParameters += 1
            simplexSet = np.append(simplexSet,self.parameterIndex.get('Linear_EnsembleWeight'))
            print("construct linear kernel")
            # bounds for ensemble weights, implicit simplex constraints applied
            lb = np.append(lb,-np.inf)
            ub = np.append(ub,np.inf)                    
 
        if automaticrelavanceKernel is not None:
            initialHyperParameters = np.append(initialHyperParameters,automaticrelavanceKernel.componentwiseScale)  
            ardparamIndexFront = numberParameters
            ardparamIndexEnd = numberParameters + automaticrelavanceKernel.componentwiseScale.shape[0] - 1 
            self.parameterIndex.update({'ARD_KernelScale':np.array([ardparamIndexFront,ardparamIndexEnd])})
            numberParameters += automaticrelavanceKernel.componentwiseScale.shape[0]
            initialHyperParameters = np.append(initialHyperParameters,self.mklregressor.weight_automaticrelavanceKernel)
            self.parameterIndex.update({'ARD_EnsembleWeight':numberParameters})
            numberParameters += 1
            simplexSet = np.append(simplexSet,self.parameterIndex.get('ARD_EnsembleWeight'))
            print("construct automatic relevance kernel")
            
            if autoRelevanceLB is not None:
                lb = np.append(lb,autoRelevanceLB * np.ones(automaticrelavanceKernel.componentwiseScale.shape[0]))
            else:
                lb = np.append(lb,-np.inf * np.ones(automaticrelavanceKernel.componentwiseScale.shape[0]))
                
            if autoRelevanceUB is not None:
                ub = np.append(ub,autoRelevanceUB * np.ones(automaticrelavanceKernel.componentwiseScale.shape[0]))
            else:
                ub = np.append(ub,np.inf * np.ones(automaticrelavanceKernel.componentwiseScale.shape[0]))
         
            # bounds for ensemble weights, implicit simplex constraints applied
            lb = np.append(lb,-np.inf)
            ub = np.append(ub,np.inf)                    
           
            
        self.lb = copy.deepcopy(lb)
        self.ub = copy.deepcopy(ub)
        self.initialHyperParam = np.copy(initialHyperParameters)
        self.currentHyperParam = np.copy(initialHyperParameters)
        self.simplexSet = simplexSet
        
                
        
    def backTesting(self,stepsize=None,maxit=None,numberDraws=None,centerdata=False):
               

        if self.hyperParameterUpdateOption == 'sgd':
            if stepsize is None:
                raise ValueError("stepsize cannot be None for sgd")
             
        elif self.hyperParameterUpdateOption == 'hoag':
            if stepsize is None:
                raise ValueError("stepsize cannot be None for hoag")

            elif maxit is None:
                raise ValueError("maxit cannot be None for hoag")
        
        elif self.hyperParameterUpdateOption == 'randomsearch' :
            if numberDraws is None:
                raise ValueError("number of draws cannot be None for randomsearch")
        
        # need cold start period order + trainingWindow*2  + (horizon-1)
        # number of validation data equals number of training data
        teststart = self.order+self.trainingWindow*2+(self.horizon-1)-1
        testend = self.timeSeriesLength-self.horizon           
        y_true = np.empty((len(range(teststart,self.timeSeriesLength)),1))
        y_true[:] = np.nan
        #y_true[:] = np.copy(self.timeSeriesData[teststart:self.timeSeriesLength].reshape((len(range(teststart,self.timeSeriesLength)),1)) )
        y_pred = np.empty((len(range(teststart,self.timeSeriesLength)),1))
        y_pred[:] = np.nan          
        paramHistory = np.empty((self.initialHyperParam.shape[0],len(range(teststart,self.timeSeriesLength))))
        paramHistory[:] = np.nan
                    
        if self.hyperParameterUpdateOption == 'sgd':
            sgdLearner = OPGD(stepsize,self.retrainPeriod,self.initialHyperParam,self.lb,self.ub,self.simplexSet)

        # t is the index for now
        bktstart = timer() 
        hypersearchtime = 0
        for t in range(teststart,testend):
            
            # periodic hyperparameter selection
            if( (t-teststart) % self.selectionPeriod ==0 ):
                if self.hyperParameterUpdateOption == 'hoag' :
                    trainingSet = self.buildTrainingSet(currentTimeStamp=t, centering = centerdata)
                    validationSet = self.buildTrainingSet(currentTimeStamp=t-self.trainingWindow, centering = centerdata)
                    xTr = trainingSet.get('X')
                    tTr = trainingSet.get('T')
                    yTr = trainingSet.get('Y')
                    xVal = validationSet.get('X')
                    tVal = validationSet.get('T')
                    yVal = validationSet.get('Y')
                    hoagOptimizer = HOAG(self,self.currentHyperParam,
                                        xTr,tTr,yTr,xVal,tVal,yVal)   
                    tol = 0.00001
                    searchstart = timer()
                    hoagopt = hoagOptimizer.hoag(maxit,tol,stepsize)
                    searchend = timer()
                    hypersearchtime = hypersearchtime + (searchend - searchstart)    
                    self.currentHyperParam = np.copy(hoagopt)

                if (self.hyperParameterUpdateOption == 'randomsearch')  :
                    if self.mklregressor.automaticrelavanceKernel is not None:
                        ardscaleLB = np.copy(self.lb[self.parameterIndex.get('ARD_KernelScale')[0]:self.parameterIndex.get('ARD_KernelScale')[1]+1])
                        ardscaleUB = np.copy(self.ub[self.parameterIndex.get('ARD_KernelScale')[0]:self.parameterIndex.get('ARD_KernelScale')[1]+1])
                    else:
                        ardscaleLB = None
                        ardscaleUB = None
                    rds = HyperParameterSearch(self.timeSeriesData, self.order, self.trainingWindow, self.horizon,
                                    self.retrainPeriod, self.lb[self.parameterIndex.get('Regularizer')], self.ub[self.parameterIndex.get('Regularizer')],
                                    squareExponentialKernel=self.mklregressor.squareExponentialKernel, sqExpScaleLB = self.lb[self.parameterIndex.get('SQEXP_KernelScale')], sqExpScaleUB = self.ub[self.parameterIndex.get('SQEXP_KernelScale')],
                                    automaticrelavanceKernel=self.mklregressor.automaticrelavanceKernel, 
                                    autoRelevanceLB = ardscaleLB, autoRelevanceUB = ardscaleUB,
                                    periodicityKernel = self.mklregressor.periodicityKernel, quasiperiodic=self.mklregressor.quasiperiodic, 
                                    prdScaleLB = self.lb[self.parameterIndex.get('PRD_KernelScale')], prdScaleUB=self.ub[self.parameterIndex.get('PRD_KernelScale')],
                                    prdPeriodLB = self.lb[self.parameterIndex.get('PRD_Period')], prdPeriodUB=self.ub[self.parameterIndex.get('PRD_Period')],
                                    ouKernel=self.mklregressor.ouKernel,  ouScaleLB=self.lb[self.parameterIndex.get('OU_KernelScale')], ouScaleUB=self.ub[self.parameterIndex.get('OU_KernelScale')],
                                    linearKernel = self.mklregressor.linearKernel, 
                                    ensembleWeight=self.ensembleWeight) 
                    searchstart = timer()                       
                    searchresult = rds.randomSearch(numberDraws)
                    searchend = timer()
                    hypersearchtime = hypersearchtime + (searchend - searchstart)    
                    self.currentHyperParam = np.copy(searchresult.get('hyperparameter'))
                    
                                             
            # periodic model update
            paramHistory[:,t-teststart] = np.copy(self.currentHyperParam)
            if( (t-teststart) % self.retrainPeriod ==0 ): 
                
                lastretrain = t
                trainingSet = self.buildTrainingSet(currentTimeStamp=t, centering = centerdata)
                X_train = np.copy(trainingSet.get('X'))
                T_train = np.copy(trainingSet.get('T'))
                y_train = np.copy(trainingSet.get('Y'))
                if centerdata is True:
                    X_mean = np.copy(trainingSet.get('Xmean'))
                    y_mean = np.copy(trainingSet.get('Ymean'))
                else:
                    X_mean = np.zeros((1,self.order))
                    y_mean = np.zeros((1,1))
                
                # model fitting          
                if (self.hyperParameterUpdateOption == 'sgd'):
                    # adaptive hyperparameters
                    oldmklregressor = copy.deepcopy(self.mklregressor)
                    self.updateHyperParam(self.currentHyperParam)                    
                    self.mklregressor.fit(X_train,T_train,y_train)
                    searchstart = timer()
                    self.mklregressor.dualWeightJacobian()
                    searchend = timer()
                    hypersearchtime = hypersearchtime + (searchend - searchstart)
                    
                else :
                    self.updateHyperParam(self.currentHyperParam)                    
                    self.mklregressor.fit(X_train,T_train,y_train)

            # prediction
            # t is now, want to predict t + horizon
            feature = np.copy(self.timeSeriesData[t+1-self.order:t+1])
            # feature imputation        
            predictionMadeStep = t
            imputationResult = self.featureImputation(feature,predictionMadeStep)
            imputedFeature = imputationResult.get('imputedFeature')
            countMissing = imputationResult.get('imputeCount')
            failImpute= imputationResult.get('failImputeFlag')

            if (countMissing <= 0.1 * self.order and failImpute == 0):
                print("prediction for step ", t-teststart)
                X_test = np.zeros((1,self.order))
                X_test[0,:] = np.copy(imputedFeature)
                X_test = X_test - X_mean 
                T_test = t+self.horizon
                # predict
                y_hat = self.mklregressor.predict(X_test,T_test)
                # adding sample mean
                y_hat = y_hat + y_mean
                # store prediction at now + horizon
                y_pred[t+self.horizon-teststart] = y_hat[0,0] 
                y_true[t+self.horizon-teststart] = self.timeSeriesData[t+self.horizon]
            
            #hyper-parameter gradient with OGD
            if ( ~np.isnan(self.timeSeriesData[t]) and ~np.isnan(y_pred[t-teststart]) ):
                if (self.hyperParameterUpdateOption == 'sgd' ): 
                    predictionMadeStep = t - self.horizon
                    feature = np.copy(self.timeSeriesData[predictionMadeStep+1-self.order:predictionMadeStep+1])
                    imputationResult = self.featureImputation(feature,predictionMadeStep)
                    X_test = np.zeros((1,self.order))
                    X_test[0,:] = np.copy(imputationResult.get('imputedFeature')) - X_mean
                    searchstart = timer()                    
                    if t < lastretrain + self.horizon :
                        hypergradDict = oldmklregressor.computeHyperGradient(X_test,t,y_pred[t-teststart],self.timeSeriesData[t])
                    else:
                        hypergradDict = self.mklregressor.computeHyperGradient(X_test,t,y_pred[t-teststart],self.timeSeriesData[t])
                    searchend = timer()
                    hypersearchtime = hypersearchtime + (searchend - searchstart)
                    hypergrad = self.hypergradDict2Array(hypergradDict)
                    searchstart = timer()                    
                    sgdLearner.psgdUpdate(np.copy(hypergrad))
                    searchend = timer()
                    hypersearchtime = hypersearchtime + (searchend - searchstart)  
                    self.currentHyperParam = np.copy(sgdLearner.x)
      
        bktend = timer()
        totaltime = bktend - bktstart                        
        resultDict  = {}
        mae = self.evaluateMAE(y_true,y_pred)
        resultDict.update({'mae':mae})
        mpe = self.evaluateMPE(y_true,y_pred)
        resultDict.update({'mpe':mpe})
        mse = self.evaluateMSE(y_true,y_pred)
        resultDict.update({'mse':mse})
        resultDict.update({'true':y_true})
        resultDict.update({'predict':y_pred})
        resultDict.update({'paramHistory':paramHistory})
        resultDict.update({'totaltime':totaltime})
        resultDict.update({'searchtime':hypersearchtime})
        return resultDict
         
    def updateHyperParam(self,hpVector):
        
        # update regularizer
        self.mklregressor.lambda_reg = np.copy(hpVector[self.parameterIndex.get('Regularizer')])
        
        # update square exponential kernel
        if self.mklregressor.squareExponentialKernel is not None:
            kernelscale = hpVector[self.parameterIndex.get('SQEXP_KernelScale')]
            kernelweight = hpVector[self.parameterIndex.get('SQEXP_EnsembleWeight')]
            self.mklregressor.weight_squareExponentialKernel = np.copy(kernelweight)
            sqExp = SqExpKernel(kernelscale)
            self.mklregressor.updateKernel(squareExpKernel=sqExp)

        if self.mklregressor.automaticrelavanceKernel is not None:
            kernelscale = hpVector[self.parameterIndex.get('ARD_KernelScale')[0]:self.parameterIndex.get('ARD_KernelScale')[1]+1]
            kernelweight = hpVector[self.parameterIndex.get('ARD_EnsembleWeight')]
            self.mklregressor.weight_automaticrelavanceKernel = np.copy(kernelweight)
            ard = ARDKernel(componentwiseScale=kernelscale)  
            self.mklregressor.updateKernel(ardKernel=ard) 
            
        if self.mklregressor.periodicityKernel is not None:
            kernelscale = hpVector[self.parameterIndex.get('PRD_KernelScale')]
            period = hpVector[self.parameterIndex.get('PRD_Period')]
            kernelweight = hpVector[self.parameterIndex.get('PRD_EnsembleWeight')]
            self.mklregressor.weight_periodicityKernel = np.copy(kernelweight)
            prdKernel = PeriodicKernel(kernelscale,period)
            self.mklregressor.updateKernel(periodKernel=prdKernel)
            
        if self.mklregressor.ouKernel is not None:
            kernelscale =hpVector[self.parameterIndex.get('OU_KernelScale')]
            kernelweight = hpVector[self.parameterIndex.get('OU_EnsembleWeight')]
            self.mklregressor.weight_ouKernel = np.copy(kernelweight)
            ouKernel = OUprocessKernel(kernelscale)
            self.mklregressor.updateKernel(ouKernel=ouKernel)
            
        if self.mklregressor.linearKernel is not None:
            kernelweight = hpVector[self.parameterIndex.get('Linear_EnsembleWeight')]
            self.mklregressor.weight_linearKernel = np.copy(kernelweight)
        
        
    def hypergradDict2Array(self,gradDict):
        
        gradArray = np.zeros(self.initialHyperParam.shape)     
        # regularizaer
        gradArray[self.parameterIndex.get('Regularizer')] = gradDict.get('Regularizer')
        # square exponential kernel
        if self.mklregressor.squareExponentialKernel is not None:
            gradArray[self.parameterIndex.get('SQEXP_KernelScale')] = gradDict.get('SQEXP_KernelScale')
            gradArray[self.parameterIndex.get('SQEXP_EnsembleWeight')] = gradDict.get('SQEXP_EnsembleWeight')
        # periodicity kernel 
        if self.mklregressor.periodicityKernel is not None:
            gradArray[self.parameterIndex.get('PRD_KernelScale')] = gradDict.get('PRD_KernelScale')
            gradArray[self.parameterIndex.get('PRD_Period')] = gradDict.get('PRD_Period')
            gradArray[self.parameterIndex.get('PRD_EnsembleWeight')] = gradDict.get('PRD_EnsembleWeight')
        # ou process kernel
        if self.mklregressor.ouKernel is not None:
            gradArray[self.parameterIndex.get('OU_KernelScale')] = gradDict.get('OU_KernelScale')
            gradArray[self.parameterIndex.get('OU_EnsembleWeight')] = gradDict.get('OU_EnsembleWeight')
        # linear kernel
        if self.mklregressor.linearKernel is not None:
            gradArray[self.parameterIndex.get('Linear_EnsembleWeight')] = gradDict.get('Linear_EnsembleWeight')
        # automatic relevance kernel
        if self.mklregressor.automaticrelavanceKernel is not None:
            gradArray[self.parameterIndex.get('ARD_KernelScale')[0]:self.parameterIndex.get('ARD_KernelScale')[1]+1] = gradDict.get('ARD_KernelScale').reshape(self.order)
            gradArray[self.parameterIndex.get('ARD_EnsembleWeight')] = gradDict.get('ARD_EnsembleWeight')
       
        return gradArray
        
    
    def buildTrainingSet(self,currentTimeStamp, centering = False):
        X_train = np.zeros((self.trainingWindow,self.order))
        T_train = np.zeros(self.trainingWindow)
        y_train = np.zeros((self.trainingWindow,1))
        validSamples = list()
        # i starts from 0, trainning data includes now    
        for i in range(0,self.trainingWindow):
            # ignore missing target data 
            if ~np.isnan(self.timeSeriesData[currentTimeStamp-i]):
                
                # raw feature from past observations
                featureX = np.copy(self.timeSeriesData[currentTimeStamp-i-self.horizon+1-self.order:currentTimeStamp-i-self.horizon+1])
                # feature imputation        
                predictionMadeStep = currentTimeStamp-i-self.horizon 
                imputationResult = self.featureImputation(featureX,predictionMadeStep)
                imputedFeature = imputationResult.get('imputedFeature')
                countMissing = imputationResult.get('imputeCount')
                failImpute= imputationResult.get('failImputeFlag')

                # ignore training sample if imputation fails
                if (countMissing <= 0.1 * self.order and failImpute == 0):
                    validSamples.append(i)
                    
                # training pair
                y_train[i] = self.timeSeriesData[currentTimeStamp-i]
                T_train[i] = currentTimeStamp-i
                X_train[i,:] = np.copy(imputedFeature)

        # remove empty training pairs
        resultDict = {}
        validSamples = np.asarray(validSamples)
        X_train = X_train[validSamples,:]
        T_train = T_train[validSamples]
        y_train = y_train[validSamples]

        # centering
        if centering is True:
            X_mean = (1/X_train.shape[0])  * np.dot(np.ones((1,X_train.shape[0])),X_train)
            X_train = X_train - np.dot( np.ones((X_train.shape[0],1)), X_mean)
            y_mean = (1/y_train.shape[0]) * np.dot(np.ones((1,y_train.shape[0])), y_train)
            y_train = y_train - y_mean * np.ones((y_train.shape[0],1)) 
    
            resultDict.update({'Xmean':X_mean})
            resultDict.update({'Ymean':y_mean})
        
        resultDict.update({'X':X_train})
        resultDict.update({'T':T_train})        
        resultDict.update({'Y':y_train})        
        
        return resultDict
       
    def featureImputation(self,feature,queryEndStep):
        countMissing = 0
        failImpute = 0
        for j in range(0,self.order):
            if (np.isnan(feature[j])):
                countMissing = countMissing+1
                if( j==0 and ~np.isnan(self.timeSeriesData[queryEndStep-self.order]) and ~np.isnan(feature[j+1]) ):
                    feature[j] = 0.5 * (self.timeSeriesData[queryEndStep-self.order] + feature[j+1])
                elif(j == self.order-1 and ~np.isnan(feature[j-1]) ):
                    feature[j] = feature[j-1]
                elif(0<j<self.order-1 and ~np.isnan(feature[j-1]) and ~np.isnan(feature[j+1]) ):
                    feature[j] = 0.5 * (feature[j-1] + feature[j+1])
                else:
                    failImpute = 1
                    
        returnitems  = {}
        returnitems.update({'imputedFeature':feature})
        returnitems.update({'imputeCount':countMissing})
        returnitems.update({'failImputeFlag':failImpute})                    
        return returnitems                
     
    def evaluateMAE(self,y_true,y_pred):
        
        mae = 0
        numsamples = y_true.shape[0]
        numeffective = 0
        for t in range(numsamples):
            if (~np.isnan(y_true[t]) and (~np.isnan(y_pred[t]))):
               numeffective = numeffective + 1
               mae = mae + np.abs(y_true[t]-y_pred[t]) 
    
    
        mae = mae/numeffective
        return mae
    
    def evaluateMPE(self,y_true,y_pred):
        
        mpe = 0
        numsamples = y_true.shape[0]
        numeffective = 0
        for t in range(numsamples):
            if (~np.isnan(y_true[t]) and (~np.isnan(y_pred[t])) and y_true[t]!=0 ):
               numeffective += 1
               mpe += np.abs(y_true[t]-y_pred[t]) / np.abs(y_true[t])
    
    
        mpe = mpe/numeffective
        return mpe
    
    def evaluateMSE(self,y_true,y_pred):
        
        mse = 0
        numsamples = y_true.shape[0]
        numeffective = 0
        for t in range(numsamples):
            if (~np.isnan(y_true[t]) and (~np.isnan(y_pred[t]))):
               numeffective += 1
               mse += np.power(y_true[t]-y_pred[t],2) 
    
    
        mse = (mse/numeffective)
        return mse
        
        

class HyperParameterSearch:
    
    def __init__(self,timeSeriesData, order, trainingWindow, horizon,
                 retrainPeriod, regLB = None, regUB = None,
                 squareExponentialKernel=None, sqExpScaleLB = None, sqExpScaleUB=None,
                 automaticrelavanceKernel=None, autoRelevanceLB = None, autoRelevanceUB=None,
                 periodicityKernel=None, quasiperiodic=False, prdScaleLB=None, prdScaleUB=None, prdPeriodLB=None, prdPeriodUB=None,
                 ouKernel=None,  ouScaleLB=None, ouScaleUB=None,
                 linearKernel = None, 
                 ensembleWeight=None):
        
        self.timeSeriesData = np.copy(timeSeriesData)
        self.order = copy.deepcopy(order)
        self.trainingWindow = copy.deepcopy(trainingWindow)
        self.horizon = copy.deepcopy(horizon)
        self.retrainPeriod = copy.deepcopy(retrainPeriod)
        self.regLB = copy.deepcopy(regLB)
        self.regUB = copy.deepcopy(regUB)
        # SE
        self.squareExponentialKernel = copy.deepcopy(squareExponentialKernel)
        self.sqExpScaleLB = copy.deepcopy(sqExpScaleLB)
        self.sqExpScaleUB = copy.deepcopy(sqExpScaleUB)
        # ARD
        self.automaticrelavanceKernel = copy.deepcopy(automaticrelavanceKernel)
        self.autoRelevanceLB = copy.deepcopy(autoRelevanceLB)
        self.autoRelevanceUB = copy.deepcopy(autoRelevanceUB)
         # periodicity
        self.periodicityKernel = copy.deepcopy(periodicityKernel)
        self.quasiperiodic = copy.deepcopy(quasiperiodic)
        self.prdScaleLB = copy.deepcopy(prdScaleLB) 
        self.prdScaleUB = copy.deepcopy(prdScaleUB)
        self.prdPeriodLB = copy.deepcopy(prdPeriodLB)
        self.prdPeriodUB = copy.deepcopy(prdPeriodUB)
        # OU
        self.ouKernel = copy.deepcopy(ouKernel)
        self.ouScaleLB = copy.deepcopy(ouScaleLB)
        self.ouScaleUB = copy.deepcopy(ouScaleUB)
        # linear
        self.linearKernel = copy.deepcopy(linearKernel)
        # weight
        self.ensembleWeight = copy.deepcopy(ensembleWeight)
        
        
    def randomSearch(self,numberDraws):
            
        bestMSE = np.inf
        
        for trial in range(numberDraws):
            if  self.automaticrelavanceKernel is not None:
                componentscale = np.random.uniform(self.autoRelevanceLB,self.autoRelevanceUB,size=self.order)
                self.automaticrelavanceKernel = ARDKernel(componentscale)
                
            if self.squareExponentialKernel is not None:
                sescale = np.random.uniform(self.sqExpScaleLB,self.sqExpScaleUB,size=1)[0]
                self.squareExponentialKernel = SqExpKernel(sescale)

            if self.periodicityKernel is not None:
                prdscale = np.random.uniform(self.prdScaleLB,self.prdScaleUB,size=1)[0]
                period = np.random.uniform(self.prdPeriodLB,self.prdPeriodUB,size=1)[0]
                self.periodicityKernel = PeriodicKernel(prdscale,period)
                
            lambda_reg = np.random.uniform(self.regLB,self.regUB,size=1)[0]
            
            # construct new MKL backtest
            backtester = RollingBackTest(self.timeSeriesData, self.order, trainingWindow= self.trainingWindow, horizon=self.horizon,
                 retrainPeriod=self.retrainPeriod,selectionPeriod=9999999999,hyperParameterUpdateOption='fixed', 
                 lambda_reg=lambda_reg, regLB=self.regLB, regUB=self.regUB,
                 squareExponentialKernel=self.squareExponentialKernel,sqExpScaleLB=self.sqExpScaleLB,sqExpScaleUB=self.sqExpScaleUB,
                 periodicityKernel=self.periodicityKernel,prdPeriodLB=self.prdPeriodLB, prdPeriodUB=self.prdPeriodUB,
                 prdScaleLB=self.prdScaleLB,prdScaleUB=self.prdScaleUB,
                 automaticrelavanceKernel=self.automaticrelavanceKernel, autoRelevanceLB=self.autoRelevanceLB, autoRelevanceUB=self.autoRelevanceUB,
                 ouKernel=self.ouKernel,ouScaleLB=self.ouScaleLB,ouScaleUB=self.ouScaleUB,
                 linearKernel=self.linearKernel,
                 ensembleWeight= self.ensembleWeight)            
            
            
            validationResult = backtester.backTesting(centerdata=True)
            if validationResult.get('mse')[0] < bestMSE :
                bestMSE = validationResult.get('mse')[0]
                selecthyperparam = np.copy(backtester.currentHyperParam)
                
            
        returnDict= {}
        returnDict.update({'bestmse':bestMSE})
        returnDict.update({'hyperparameter':selecthyperparam})
        return returnDict    

                

