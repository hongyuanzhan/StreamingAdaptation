#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 14:14:17 2017

@author: Zhan
"""
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from KernelRidgeARD import KernelRidgeARD

class RollingBackTestKRR:
        

    def __init__(self,timeSeriesData, order, trainingWindow, horizon):
        
        self.timeSeriesData = timeSeriesData
        self.timeSeriesLength = self.timeSeriesData.shape[0]
        self.order = order
        # number of trainign data
        self.trainingWindow = trainingWindow
        # number of steps ahead, must be at least 1 
        self.horizon = horizon
        
    def backTesting(self,retrainPeriod,hyperParameterUpdateOption,componentKernelWidth = None, kernelWidthLearningRate=None,kernelWidthLB=None,kernelWidthUB=None):
               
            
        # how to set hyperparameters
        hpOptionSet = ['sgd','sgdnormalized','fixed']
        if not(hyperParameterUpdateOption in hpOptionSet):
            print("hyperparameter selection options are: 'fixed' or 'sgd', using 'fixed' by default")
            hyperParameterUpdateOption = 'fixed'
        
        # need cold start period order + trainingWindow + (horizon-1)
        teststart = self.order+self.trainingWindow+(self.horizon-1)-1
        testend = self.timeSeriesLength-self.horizon           
        y_true = np.empty((len(range(teststart,self.timeSeriesLength)),1))
        y_true[:] = np.copy(self.timeSeriesData[teststart:self.timeSeriesLength].reshape((len(range(teststart,self.timeSeriesLength)),1)) )
        y_pred = np.empty((len(range(teststart,self.timeSeriesLength)),1))
        y_pred[:] = np.nan          
    
            
        # using ARD kennel ridge regression
        componentKernelWidth.reshape(self.order)
        ardKRR = KernelRidgeARD(componentKernelWidth,0.5,self.order)

        # t is the index for now
        for t in range(teststart,testend):
            # periodic model update
            if( (t-teststart) % retrainPeriod ==0 ): 
                X_train = np.zeros((self.trainingWindow,self.order))
                y_train = np.zeros((self.trainingWindow,1))
                validSamples = list()
                # i starts from 0, trainning data includes now    
                for i in range(0,self.trainingWindow):
                    # ignore missing target data 
                    if ~np.isnan(self.timeSeriesData[t-i]):
                        # raw feature
                        feature = np.copy(self.timeSeriesData[t-i-self.horizon+1-self.order:t-i-self.horizon+1])
                        # feature imputation        
                        predictionMadeStep = t-i-self.horizon 
                        imputationResult = self.featureImputation(feature,predictionMadeStep)
                        imputedFeature = imputationResult.get('imputedFeature')
                        countMissing = imputationResult.get('imputeCount')
                        failImpute= imputationResult.get('failImputeFlag')

                        # ignore training sample if imputation fails
                        if (countMissing <= 0.1 * self.order and failImpute == 0):
                            validSamples.append(i)
                            
                        # training pair
                        y_train[i] = self.timeSeriesData[t-i]
                        X_train[i,:] = np.copy(imputedFeature)


                # remove empty training pairs
                validSamples = np.asarray(validSamples)
                X_train = X_train[validSamples,:]
                y_train = y_train[validSamples]
                
                # centering
                X_mean = (1/X_train.shape[0])  * np.dot(np.ones((1,X_train.shape[0])),X_train)
                X_train = X_train - np.dot( np.ones((X_train.shape[0],1)), X_mean)
                y_mean = (1/y_train.shape[0]) * np.dot(np.ones((1,y_train.shape[0])), y_train)
                y_train = y_train - y_mean * np.ones((y_train.shape[0],1)) 
                # model fitting
                if (hyperParameterUpdateOption == 'fixed'):
                    # set fixed hyperparameter
                    ardKRR.setARDkernel(componentKernelWidth);
                    ardKRR.setRegularization(1)
                    ardKRR.fit(X_train,y_train)

                elif (hyperParameterUpdateOption == 'sgd'):
                    # adaptive hyperparameters
                    ardKRR.kernelSGDupdate(kernelWidthLearningRate,kernelWidthLB,kernelWidthUB,'sgd')
                    ardKRR.fit(X_train,y_train)
                    ardKRR.computeDesignMatrixDerivatives()
                    
                elif (hyperParameterUpdateOption == 'sgdnormalized'):
                    # adaptive hyperparameters
                    ardKRR.kernelSGDupdate(kernelWidthLearningRate,kernelWidthLB,kernelWidthUB,'sgdnormalized')
                    ardKRR.fit(X_train,y_train)
                    ardKRR.computeDesignMatrixDerivatives()
                    print("retrained model")
        
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
                # predict
                #y_hat = fittedKrr.predict(X_test)
                y_hatARD = ardKRR.predict(X_test)
                #print("difference between skilearn and ARD before re-normalizing: ", y_hat - y_hatARD)

                # adding sample mean
                #y_hat = y_hat + y_mean
                y_hatARD = y_hatARD + y_mean
                #print("difference between skilearn and ARD: ", y_hat - y_hatARD)
                # store prediction at now + horizon
                y_pred[t+self.horizon-teststart] = y_hatARD[0,0]  
                
            else:
                print("fail to pred for step ",t-teststart)         
            
            #hyper-parameter gradient
            if ( ~np.isnan(self.timeSeriesData[t]) and ~np.isnan(y_pred[t-teststart]) ):
                print("prediction squared error: ", (y_pred[t-teststart] - self.timeSeriesData[t])**2)
                print("prediction relative error: ", np.abs(y_pred[t-teststart] - self.timeSeriesData[t])/np.abs(self.timeSeriesData[t]) )
                if (hyperParameterUpdateOption == 'sgd' or hyperParameterUpdateOption == 'sgdnormalized'): 
                    
                    predictionMadeStep = t - self.horizon
                    feature = np.copy(self.timeSeriesData[predictionMadeStep+1-self.order:predictionMadeStep+1])
                    imputationResult = self.featureImputation(feature,predictionMadeStep)
                    X_test = np.zeros((1,self.order))
                    X_test[0,:] = np.copy(imputationResult.get('imputedFeature')) - X_mean
                    analyticG = ardKRR.computeKernelHyperGradient(X_test,y_pred[t-teststart],self.timeSeriesData[t])
                          
                    #analyticG = ardKRR.computeKernelHyperGradient(X_test,y_hatARD[0,0],self.timeSeriesData[t])
                    
                    print("compute hyperparameter analytical gradient: ",analyticG)
                    #finiteDiffG = ardKRR.kernelGradientFintieDifference(X_test,y_hatARD[0,0]-y_mean,self.timeSeriesData[t]-y_mean,0.00000000000001)
                    #print("compute hyperparameter finite difference gradient: ",finiteDiffG)
                    #print("relative error between finite difference and analytical gradient",np.linalg.norm(analyticG - finiteDiffG)/np.linalg.norm(analyticG) )
                
                
        resultList  = {}
        mae = self.evaluateMAE(y_true,y_pred)
        resultList.update({'mae':mae})
        mpe = self.evaluateMPE(y_true,y_pred)
        resultList.update({'mpe':mpe})
        mse = self.evaluateMSE(y_true,y_pred)
        resultList.update({'mse':mse})
        resultList.update({'true':y_true})
        resultList.update({'predict':y_pred})
        return resultList
        
    
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
