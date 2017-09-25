#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:19:21 2017

@author: Hongyuan Zhan
"""

import numpy as np 
from numpy.linalg import solve
from numpy.linalg import inv
import time


class KernelRidgeARD:
    
    def __init__(self,lambda_ard,lambda_reg,p):
        
        # number of parameters
        self.p = np.copy(p)
        # kernel parameter
        self.lambda_ard = np.copy(lambda_ard.reshape((p,1)))
        # regularization parameter
        self.lambda_reg = np.copy(lambda_reg)
        # kernel matrix of training data
        self.kernelmatrix = np.nan
        # kernel matrix + ridge factor
        self.kernelridgeDesign = np.nan 
        # fitted weight 
        self.fittedweight = np.nan
        self.training_X = np.nan
        self.training_Y = np.nan
        self.ardNorm = np.nan
        # derivative of kernel design matrix (K + lambda I)^{-1}
        self.kernelDerivative = np.nan
        # average hypergradient of kernel hyperparameter since the last training
        self.aRDHyperGradient = np.zeros((p,1))
        # number of steps predicted since the last training
        self.numStepsPredicted = 0
        print("construct Kernel Ridge predictor with ARD kernel")
        
    def fit(self,training_X,training_Y):
        # assign training data
        self.training_X = np.copy(training_X)
        self.training_Y = np.copy(training_Y)
        # number of training samples
        N = training_X.shape[0]
        # feature dimension
        p = training_X.shape[1]
        self.kernelmatrix = np.zeros((N,N))
        # compute ARD kernel matrix
        grammatrix = np.dot( np.dot(training_X,np.diag(self.lambda_ard.reshape(p))) , np.transpose(training_X))
        self.ardNorm = np.diag(grammatrix).reshape((N,1))
        self.kernelmatrix = ( -2*grammatrix 
                              + np.dot(self.ardNorm , np.ones((1,N))) 
                              + np.dot(self.ardNorm , np.ones((1,N))).transpose() )
        self.kernelmatrix = np.exp(-self.kernelmatrix)
        self.kernelridgeDesign = self.kernelmatrix+ self.lambda_reg*np.eye(N)
        # dual weights 
        self.fittedweight = solve(self.kernelridgeDesign,training_Y)
 
        returnlist = {}
        returnlist.update({'KernelMatrix':self.kernelmatrix})
        returnlist.update({'KernelRidgeDesignMatrix':self.kernelridgeDesign})
        returnlist.update({'FittedWeights':self.fittedweight})
        return returnlist    
    
    
    def computeDesignMatrixDerivatives(self):
        # number of samples
        N = self.training_X.shape[0]
        # derivative of kernel design matrix using for hyperparameter updates
        self.kernelDerivative = np.zeros((self.p,N))
        # inverse of design matrix
        invKernelRidgeDesign = inv(self.kernelridgeDesign)  
        for l in range(self.p):
            #print("hyperparameter ",l)
            featureDiff = (self.training_X[:,l][:,np.newaxis] - self.training_X[:,l])**2
            partialDerivativeTrainingKernel = -self.kernelmatrix * featureDiff
            b = partialDerivativeTrainingKernel.dot(self.fittedweight) 
            self.kernelDerivative[l,:] = - np.transpose(invKernelRidgeDesign.dot(b))

        # ard kernel hyper gradient --- accumluated in the prediction steps
        self.aRDHyperGradient = np.zeros((self.p,1))   
        # number of steps predicted since the latest training
        self.numStepsPredicted = 0 
    
    
    def predict(self,x_new):
        # number of training samples
        N = self.training_X.shape[0]
        # feature dimension
        p = self.training_X.shape[1]
        kernelDistance = np.zeros((1,N))
        x_new.reshape((1,p))
        grammatrix = np.dot(x_new , np.dot(np.diag(self.lambda_ard.reshape(p)) , np.transpose(self.training_X)) )
        kernelDistance = (-2 * grammatrix 
                          + x_new.dot(np.diag(self.lambda_ard.reshape(p))).dot(np.transpose(x_new)).dot(np.ones((1,N)))
                          + self.ardNorm.reshape((1,N)) )
        kernelDistance = np.exp(-kernelDistance)
        y_predict = kernelDistance.dot(self.fittedweight)
        return y_predict
    
    def setARDkernel(self,lambda_ard):
        self.lambda_ard = np.copy(lambda_ard).reshape((self.p,1))
        
    def setRegularization(self,lambda_reg):
        self.lambda_reg = np.copy(lambda_reg)
    
    def computeKernelHyperGradient(self,x_new,y_predict,y_true):
        
        start = time.time()
        # number of training samples
        N = self.training_X.shape[0]
        # feature dimension
        p = self.training_X.shape[1]
        assert p == self.p, "dimension of feature does not match number of columns in X matrix"
        kernelDistance = np.zeros((1,N))
        x_new.reshape((1,p))
        grammatrix = np.dot(x_new , np.dot(np.diag(self.lambda_ard.reshape(p)) , np.transpose(self.training_X)) )
        kernelDistance = (-2 * grammatrix 
                          + x_new.dot(np.diag(self.lambda_ard.reshape(p))).dot(np.transpose(x_new)).dot(np.ones((1,N)))
                          + self.ardNorm.reshape((1,N)) )
     
        kernelDistance = np.exp(-kernelDistance)
        # p-by-N matrix of square difference of features, compared to the N training samples
        newSampleFeatureSquareDifference = np.transpose((self.training_X - np.ones((N,1)).dot(x_new) )**2)
        # store hyperparameter gradient w.r.t ARD kernel
        hyperGradient = np.zeros((p,1))
        testingKernelPartialDerivative = - np.outer(np.ones((p,1)),kernelDistance) * newSampleFeatureSquareDifference
        A = testingKernelPartialDerivative.dot(self.fittedweight)
        B = self.kernelDerivative.dot(kernelDistance.transpose())
        hyperGradient = -2 * (y_true - y_predict) * (A + B) 
        # accumulated average hypergradient since the last training
        self.aRDHyperGradient = (self.aRDHyperGradient * self.numStepsPredicted + hyperGradient) / (self.numStepsPredicted+1)
        # accumulated number of steps predicted sicne the last training
        self.numStepsPredicted+= 1                                         
        # calculate elapsed time        
        end = time.time()
        #print("elapsed time: ", end - start)
        return hyperGradient
                
    
    def kernelGradientFintieDifference(self,x_new,y_predict,y_true,epsilon):
        
        # error without hyperparameer perturbation
        pred_error = (y_true - y_predict)**2
        # number of training samples
        N = self.training_X.shape[0]
        # feature dimension
        p = self.training_X.shape[1]

        finiteDiffKernelGrad = np.zeros((p,1))
        for l in range(p):
            #print("computing ARD kernel hyperparameter finite difference gradient for coodinate",l)
            perturbKernelParameter = np.copy(self.lambda_ard) 
            perturbKernelParameter[l] = perturbKernelParameter[l] + epsilon
            # compute ARD kernel matrix
            perturbGramMatrix = self.training_X.dot(np.diag(perturbKernelParameter.reshape(p))).dot(np.transpose(self.training_X))
            perturbARDnorm = np.diag(perturbGramMatrix).reshape((N,1))
            perturbKernelMatrix = ( -2*perturbGramMatrix
                              + np.dot(perturbARDnorm , np.ones((1,N))) 
                              + np.dot(perturbARDnorm , np.ones((1,N))).transpose() )
            perturbKernelMatrix = np.exp(-perturbKernelMatrix)
            # perturb fitted weight
            perturbWeight = solve(perturbKernelMatrix+ self.lambda_reg*np.eye(N),self.training_Y)
            # perturb kernel distance of testing sample
            x_new.reshape((1,p))
            perturbTestGram = np.dot(x_new , np.dot(np.diag(perturbKernelParameter.reshape(p)) , np.transpose(self.training_X)) )
            perturbTestKernel = (-2 * perturbTestGram
                          + x_new.dot(np.diag(perturbKernelParameter.reshape(p))).dot(np.transpose(x_new)).dot(np.ones((1,N)))
                          + perturbARDnorm.reshape((1,N)) )
            perturbTestKernel = np.exp(-perturbTestKernel)
            y_perturbed_predict = perturbTestKernel.dot(perturbWeight)
            perturb_error = (y_perturbed_predict  - y_true)**2
            finiteDiffKernelGrad[l,0] = (perturb_error - pred_error) / epsilon
                                
        return finiteDiffKernelGrad
    
    
    def kernelSGDupdate(self,stepsize,lb,ub, option):

        if (option == 'sgdnormalized'):
            
            if (np.linalg.norm(self.aRDHyperGradient) != 0):
                self.lambda_ard = self.lambda_ard - stepsize * self.aRDHyperGradient / np.linalg.norm(self.aRDHyperGradient)
                self.lambda_ard[self.lambda_ard > ub] = ub
                self.lambda_ard[self.lambda_ard < lb] = lb
                               
            print("kernel parameters are: ",self.lambda_ard.reshape(self.p))
               
        elif (option == 'sgd'):
            self.lambda_ard = self.lambda_ard - stepsize * self.aRDHyperGradient
            self.lambda_ard[self.lambda_ard > ub] = ub
            self.lambda_ard[self.lambda_ard < lb] = lb
            print("kernel parameters are: ",self.lambda_ard.reshape(self.p))
             
        else:
            print("invalid hyperparameter update method")
                           
            
    
                
    