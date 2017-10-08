#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:28:46 2017

@author: Zhan
"""

import numpy as np 
import copy


class SqExpKernel:
    
    def __init__(self,kernelscale,kernelmatrix=None,kernelmatrixDerivative = None):
        self.kernelscale = np.copy(kernelscale)
        self.kernelmatrix =  copy.deepcopy(kernelmatrix) 
        self.kernelmatrixDerivative = copy.deepcopy(kernelmatrixDerivative)

    def computeKernelMatrix(self,X):     
        N = X.shape[0]
        grammatrix = np.dot(X,np.transpose(X))
        rownorm = np.diag(grammatrix).reshape((N,1))
        pairwiseEuclidDistance =  ( -2*grammatrix 
                         + np.dot(rownorm , np.ones((1,N))) 
                         + np.dot(rownorm , np.ones((1,N))).transpose() )
        kernelmatrix = np.exp(- self.kernelscale * pairwiseEuclidDistance)
        self.kernelmatrix = np.copy(kernelmatrix)
        kernelmatrixDeriv = - kernelmatrix * pairwiseEuclidDistance
        self.kernelmatrixDerivative = np.copy(kernelmatrixDeriv)
        returnitems = {}
        returnitems.update({'KernelMatrix':kernelmatrix})
        returnitems.update({'KernelMatrixDerivative':kernelmatrixDeriv})
        return returnitems
    
    def computeKernel(self,X,xnew,requireDerivative = False):
        grammatrix = np.dot(xnew,np.transpose(X))
        xRowNormSq = np.linalg.norm(X,axis=1)**2
        xRowNormSq = np.reshape(xRowNormSq,grammatrix.shape)
        EuclidDistance = (-2*grammatrix + xRowNormSq
                          + np.linalg.norm(xnew)**2 * np.ones(grammatrix.shape))
        kernel = np.exp(-self.kernelscale * EuclidDistance)
        returnitems = {}
        returnitems.update({'Kernel':kernel})
        if (requireDerivative):
            kernelDerivative = - kernel * EuclidDistance
            returnitems.update({'KernelDerivative':kernelDerivative})
        
        return returnitems
        
          
class ARDKernel:

    def __init__(self,componentwiseScale,kernelmatrix=None,kernelmatrixDerivative = None):
        self.componentwiseScale = np.copy(componentwiseScale)  
        self.kernelmatrix =  copy.deepcopy(kernelmatrix) 
        self.normInARD = None
        
    def computeKernelMatrix(self,X):
        #compute ARD kernel matrix
        [N,p] = X.shape
        grammatrix = np.dot(np.dot(X,np.diag(self.componentwiseScale.reshape(p))),np.transpose(X))
        normInARD = np.diag(grammatrix).reshape((N,1))
        self.normInARD = np.copy(normInARD)
        kernelmatrix = ( -2*grammatrix 
                              + np.dot(normInARD , np.ones((1,N))) 
                              + np.dot(normInARD , np.ones((1,N))).transpose())
        kernelmatrix = np.exp(- kernelmatrix)
        self.kernelmatrix = np.copy(kernelmatrix)

        returnitems = {}
        returnitems.update({'KernelMatrix':kernelmatrix})
        return returnitems  

    
    def computeKernel(self,X,xnew,requireDerivative = False):

        [N,p] = X.shape
        grammatrix = np.dot(xnew , np.dot(np.diag(self.componentwiseScale.reshape(p)) , np.transpose(X)) )
        kernel = (-2 * grammatrix 
                  + xnew.dot(np.diag(self.componentwiseScale.reshape(p))).dot(np.transpose(xnew)) * np.ones((1,N))
                  + self.normInARD.reshape((1,N)) )
     
        kernel = np.exp(-kernel)
        returnitems = {}
        returnitems.update({'Kernel':kernel}) 
        if (requireDerivative):       
            # p-by-N matrix of square difference of features, compared to the N training samples
            featureDiff = np.transpose((X - np.ones((N,1)).dot(xnew.reshape(1,p)) )**2)
            # kernel derivative in ARD is p-by-N 
            kernelDerivative = - np.outer(np.ones((p,1)),kernel) * featureDiff
            returnitems.update({'KernelDerivative':kernelDerivative})

        return returnitems                                     

   
class PeriodicKernel:
    
    def __init__(self,kernelscale,period,kernelmatrix = None,kernelmatrixDerivative_scale = None,kernelmatrixDerivative_period = None):
        self.kernelscale = np.copy(kernelscale);
        self.period = np.copy(period)
        self.kernelmatrix =  copy.deepcopy(kernelmatrix)
        self.kernelmatrixDerivative_scale =  copy.deepcopy(kernelmatrixDerivative_scale)
        self.kernelmatrixDerivative_period = copy.deepcopy(kernelmatrixDerivative_period)
        
    def computeKernelMatrix(self,T, quasi = True):
        """
        T is a (N,1) or (N,) ndarray  
        timediff is (N,N) matrix of abs of difference in timestamp
        """
        # move computation outside
        N = T.shape[0]
        T = np.copy(T.reshape(N))
        timediff = np.abs(T[:,np.newaxis] - T)
        sinsq = np.sin(np.pi/self.period * timediff)**2 
                              
        if quasi is True:
            kernelmatrix = np.exp(- self.kernelscale * (sinsq + timediff**2))
            self.kernelmatrix = np.copy(kernelmatrix)
            kernelmatrixDeriv_scale = -kernelmatrix * (sinsq + timediff**2)
            self.kernelmatrixDerivative_scale = np.copy(kernelmatrixDeriv_scale)
            kernelmatrixDeriv_period = (2*kernelmatrix * self.kernelscale 
                                        * np.sin(np.pi/self.period * timediff) 
                                        * np.cos(np.pi/self.period * timediff)
                                        * np.pi * timediff / self.period**(2))
            self.kernelmatrixDerivative_period = np.copy(kernelmatrixDeriv_period)
                     
        else:
            kernelmatrix = np.exp( - self.kernelscale * sinsq)
            self.kernelmatrix = np.copy(kernelmatrix)
            kernelmatrixDeriv_scale = -kernelmatrix * sinsq 
            self.kernelmatrixDerivative_scale = np.copy(kernelmatrixDeriv_scale)
            kernelmatrixDeriv_period = (2*kernelmatrix * self.kernelscale 
                                        * np.sin(np.pi/self.period * timediff) 
                                        * np.cos(np.pi/self.period * timediff)
                                        * np.pi * timediff / self.period**(2))
            self.kernelmatrixDerivative_period = np.copy(kernelmatrixDeriv_period)
        
        returnitems = {}
        returnitems.update({'KernelMatrix':kernelmatrix})
        returnitems.update({'KernelMatrixDerivativeOfScale':kernelmatrixDeriv_scale})
        returnitems.update({'KernelMatrixDerivativeOfPeriod':kernelmatrixDeriv_period})
        
        return returnitems

    def computeKernel(self,T,tnew,requireDerivative = False,quasi = True):     
        timediff = np.abs(T - np.ones(np.shape(T))*tnew)
        sinsq = np.sin(np.pi/self.period * timediff)**2
        
        if quasi is True:
            kernel = np.exp(-self.kernelscale * (sinsq + timediff**2))   
        else:
            kernel = np.exp(-self.kernelscale * sinsq)
        
        returnitems = {}
        returnitems.update({'Kernel':kernel})        
        if(requireDerivative):    
            if quasi is True:
                kernelDerivative_scale = -kernel * (sinsq+timediff**2) 

            else:
                kernelDerivative_scale = -kernel * sinsq 

            kernelDerivative_period = (2*kernel * self.kernelscale 
                                        * np.sin(np.pi/self.period * timediff) 
                                        * np.cos(np.pi/self.period * timediff)
                                        * np.pi * timediff / self.period**(2))
            returnitems.update({'KernelDerivativeOfScale':kernelDerivative_scale})
            returnitems.update({'KernelDerivativeOfPeriod':kernelDerivative_period})

        return returnitems
        
        
class OUprocessKernel:
    
    def __init__(self,kernelscale,kernelmatrix=None,kernelmatrixDerivative=None):
        self.kernelscale = np.copy(kernelscale)
        self.kernelmatrix = copy.deepcopy(kernelmatrix) 
        self.kernelmatrixDerivative =  copy.deepcopy(kernelmatrixDerivative)
        
    def computeKernelMatrix(self,T):
        """
        T is a (N,1) or (N,) ndarray
        timediff is (N,N) matrix of abs of difference in timestamp
        """
        # move computation outside
        N = T.shape[0]
        T = np.copy(T.reshape(N))
        timediff = np.abs(T[:,np.newaxis] - T)
        kernelmatrix = np.exp(-self.kernelscale * timediff)
        self.kernelmatrix = np.copy(kernelmatrix)
        kernelmatrixDeriv = - kernelmatrix * timediff
        self.kernelmatrixDerivative = np.copy(kernelmatrixDeriv)
        returnitems = {}
        returnitems.update({'KernelMatrix':kernelmatrix})
        returnitems.update({'KernelMatrixDerivative':kernelmatrixDeriv})

        return returnitems
 
    def computeKernel(self,T,tnew,requireDerivative = False):
        timediff = np.abs(T - np.ones(np.shape(T))*tnew)
        kernel = np.exp(-self.kernelscale * timediff)
        returnitems = {}
        returnitems.update({'Kernel':kernel})
        if(requireDerivative):
            kernelDerivative = - kernel * timediff
            returnitems.update({'KernelDerivative':kernelDerivative})
        
        return returnitems
       


class LinearKernel:
    
    def __init__(self,kernelmatrix=None):
        self.kernelmatrix =  copy.deepcopy(kernelmatrix)

    def computeKernelMatrix(self,X):
        kernelmatrix = np.dot(X,np.transpose(X))
        self.kernelmatrix = np.copy(kernelmatrix)
        returnitems = {}
        returnitems.update({'KernelMatrix':kernelmatrix}) 
        return returnitems       

    def computeKernel(self,X,xnew):
        kernel = np.dot(xnew,np.transpose(X))
        returnitems = {}
        returnitems.update({'Kernel':kernel})
        return returnitems
    

class MKLregression:
    
    def __init__(self,p, lambda_reg, 
                 squareExponentialKernel=None, automaticrelavanceKernel=None, periodicityKernel=None, quasiperiodic=False, ouKernel=None, linearKernel = None, ensembleWeight=None, 
                 training_T = None,training_X = None, training_Y = None,
                 ensembleKernelMatrix=None, kernelridgeDesign=None, inverseKernelRidgeDesign = None, fittedweight = None, jacobian_regularization=None,
                 jacobian_SqExp_EnsWeight=None,jacobian_SqExp_kernelscale=None,
                 jacobian_PRD_EnsWeight=None,jacobian_PRD_kernelscale=None,jacobian_PRD_period=None,
                 jacobian_OU_EnsWeight=None,jacobian_OU_kernelscale=None,
                 jacobian_Linear_EnsWeight=None,
                 jacobian_ARD_EnsWeight=None, jacobian_ARD_kernelscale=None):

        # number of parameters
        self.p = np.copy(p)
        # regularization parameter
        self.lambda_reg = np.copy(lambda_reg)
        # number of kernels used
        self.numKernels = 0
        # squared exponential kernel parameter
        self.squareExponentialKernel = copy.deepcopy(squareExponentialKernel)        
        if squareExponentialKernel is not None:
            print("square exponential kernel used")
            self.numKernels += 1

        # ard SE kernel
        self.automaticrelavanceKernel = copy.deepcopy(automaticrelavanceKernel)
        if automaticrelavanceKernel is not None:
            print("ARD kernel used")
            self.numKernels += 1

        # periodic kernel parameter
        self.periodicityKernel = copy.deepcopy(periodicityKernel)
        self.quasiperiodic = quasiperiodic
        if periodicityKernel is not None:
            print("periodicity kernel used")
            self.numKernels += 1
              
        # OU process kernel parameter
        self.ouKernel = copy.deepcopy(ouKernel)
        if ouKernel is not None:
            print("OU process kernel used")
            self.numKernels += 1
            
        # linear kernel 
        self.linearKernel = copy.deepcopy(linearKernel)
        if linearKernel is not None:
            print("linear kernel used")
            self.numKernels += 1

        """
        to do
        # default kernel
        # if (seKernel is None) and (ardKernel is None) and (ouKernel is None) and (prdKernel is None):
        """    
        # kernel ensemble weight
        if ensembleWeight is not None:
            totalweightInput = 0
            if squareExponentialKernel is not None:
                totalweightInput += ensembleWeight[0]
                self.weight_squareExponentialKernel = ensembleWeight[0]    
                
            if periodicityKernel is not None:
                totalweightInput += ensembleWeight[1]
                self.weight_periodicityKernel = ensembleWeight[1]

            if ouKernel is not None:
                totalweightInput += ensembleWeight[2]
                self.weight_ouKernel = ensembleWeight[2]
 
            if linearKernel is not None:
                totalweightInput += ensembleWeight[3]
                self.weight_linearKernel = ensembleWeight[3]
    
            if automaticrelavanceKernel is not None:
                totalweightInput += ensembleWeight[4]
                self.weight_automaticrelavanceKernel = ensembleWeight[4]
            
            
            if (totalweightInput != 1):
                raise ValueError('total ensemble weight should be one')
                 
            # contain negative values    
            if (True in (ensembleWeight<0)):
                raise ValueError('weight should be positive')
                
        else:
            
            if squareExponentialKernel is not None:
                self.weight_squareExponentialKernel = 1/self.numKernels 
                
            if periodicityKernel is not None:
                self.weight_periodicityKernel =  1/self.numKernels 

            if ouKernel is not None:
                self.weight_ouKernel =  1/self.numKernels 
 
            if linearKernel is not None:
                self.weight_linearKernel = 1/self.numKernels
    
            if automaticrelavanceKernel is not None:
                self.weight_automaticrelavanceKernel = 1/self.numKernels 
             
        # ensemble multiple kernel matrix 
        self.ensembleKernelMatrix = copy.deepcopy(ensembleKernelMatrix)                                   
        # ensemble kernel matrix + ridge factor
        self.kernelridgeDesign = copy.deepcopy(kernelridgeDesign) 
        # inverse of self.kernelridgeDesign
        self.inverseKernelRidgeDesign = copy.deepcopy(inverseKernelRidgeDesign)
        # fitted weight 
        self.fittedweight = copy.deepcopy(fittedweight)
        # training data
        self.training_T = copy.deepcopy(training_T)
        self.training_X = copy.deepcopy(training_X)
        self.training_Y = copy.deepcopy(training_Y)
        # jacobian of dual solution w.r.t hyperparameters
        # regularizater
        self.jacobian_regularization = copy.deepcopy(jacobian_regularization)
        # Square Exponential Kernel
        self.jacobian_SqExp_EnsWeight = copy.deepcopy(jacobian_SqExp_EnsWeight)
        self.jacobian_SqExp_kernelscale = copy.deepcopy(jacobian_SqExp_kernelscale)
        # Periodicity kernel 
        self.jacobian_PRD_EnsWeight = copy.deepcopy(jacobian_PRD_EnsWeight)
        self.jacobian_PRD_kernelscale = copy.deepcopy(jacobian_PRD_kernelscale)
        self.jacobian_PRD_period = copy.deepcopy(jacobian_PRD_period)
        # OU process kernel
        self.jacobian_OU_EnsWeight = copy.deepcopy(jacobian_OU_EnsWeight)
        self.jacobian_OU_kernelscale = copy.deepcopy(jacobian_OU_kernelscale)
        # Linear kernel
        self.jacobian_Linear_EnsWeight = copy.deepcopy(jacobian_Linear_EnsWeight)
        # ARD kernel
        self.jacobian_ARD_EnsWeight = copy.deepcopy(jacobian_ARD_EnsWeight)
        self.jacobian_ARD_kernelscale = copy.deepcopy(jacobian_ARD_kernelscale)
        
        print("construct Multiple Kernel Learning Regression")
        
    
    def updateKernel(self,squareExpKernel=None,periodKernel=None,ouKernel=None,linearKernel=None,ardKernel=None):
        
        if (squareExpKernel is not None) and (self.squareExponentialKernel is not None):
            self.squareExponentialKernel = copy.deepcopy(squareExpKernel)
        
        if (periodKernel is not  None) and (self.periodicityKernel is not None):
            self.periodicityKernel = copy.deepcopy(periodKernel)

        if (ouKernel is not None) and (self.ouKernel is not None):
            self.ouKernel = copy.deepcopy(ouKernel)

        if (linearKernel is not None) and (self.linearKernel is not None):
            self.linearKernel = copy.deepcopy(linearKernel)

        if (ardKernel is not None) and (self.automaticrelavanceKernel is not None):
            self.automaticrelavanceKernel = copy.deepcopy(ardKernel)              
        
    
    def fit(self,X,T,Y):
        
        # number of training samples
        N = X.shape[0]
        self.training_T = np.copy(T)
        self.training_X = np.copy(X)
        self.training_Y = np.copy(Y)
        self.ensembleKernelMatrix = np.zeros((N,N))
        if self.squareExponentialKernel is not None:
            seTraining = self.squareExponentialKernel.computeKernelMatrix(X)
            self.ensembleKernelMatrix += self.weight_squareExponentialKernel * seTraining.get('KernelMatrix')
            
        if self.periodicityKernel is not None:
            prdTraining = self.periodicityKernel.computeKernelMatrix(T,quasi=self.quasiperiodic)
            self.ensembleKernelMatrix += self.weight_periodicityKernel * prdTraining.get('KernelMatrix')

        if self.ouKernel is not None:
            ouTraining = self.ouKernel.computeKernelMatrix(T)
            self.ensembleKernelMatrix += self.weight_ouKernel * ouTraining.get('KernelMatrix')

        if self.linearKernel is not None:
            linearTraining = self.linearKernel.computeKernelMatrix(X)
            self.ensembleKernelMatrix += self.weight_linearKernel * linearTraining.get('KernelMatrix')

        if self.automaticrelavanceKernel is not None:
            ardTraining = self.automaticrelavanceKernel.computeKernelMatrix(X)
            self.ensembleKernelMatrix += self.weight_automaticrelavanceKernel * ardTraining.get('KernelMatrix')
            

        
        # kernel ridge design matrix
        self.kernelridgeDesign = (self.ensembleKernelMatrix + self.lambda_reg * np.eye(N))
        # inverse of kernel ridge design matrix
        self.inverseKernelRidgeDesign = np.linalg.inv(self.kernelridgeDesign)
        # dual weight
        self.fittedweight = self.inverseKernelRidgeDesign.dot(Y)

    def predict(self,xnew,tnew):

        if self.fittedweight is None:
            raise ValueError('must fit the model before predict')

        N = self.training_X.shape[0]
        ensembleKernel = np.zeros(N)            
        if self.squareExponentialKernel is not None:
            sekernel = self.squareExponentialKernel.computeKernel(self.training_X,xnew)
            ensembleKernel += self.weight_squareExponentialKernel * sekernel.get('Kernel').reshape(N)
            
        if self.periodicityKernel is not None:
            prdkernel = self.periodicityKernel.computeKernel(self.training_T,tnew,quasi=self.quasiperiodic)
            ensembleKernel += self.weight_periodicityKernel * prdkernel.get('Kernel').reshape(N)

        if self.ouKernel is not None:
            oukernel = self.ouKernel.computeKernel(self.training_T,tnew)
            ensembleKernel += self.weight_ouKernel * oukernel.get('Kernel').reshape(N)
        
        if self.linearKernel is not None:
            linearkernel = self.linearKernel.computeKernel(self.training_X,xnew)
            ensembleKernel += self.weight_linearKernel * linearkernel.get('Kernel').reshape(N)
        
        if self.automaticrelavanceKernel is not None:               
            ardkernel = self.automaticrelavanceKernel.computeKernel(self.training_X,xnew)
            ensembleKernel += self.weight_automaticrelavanceKernel * ardkernel.get('Kernel').reshape(N)
            
        ynew = ensembleKernel.dot(self.fittedweight)
        return ynew
    
    
    def dualWeightJacobian(self):
        
        if(self.fittedweight is None) or (self.inverseKernelRidgeDesign is None):
            raise ValueError('dual weight and inverse of kernel ridge design not available')

        resultDict = {}
        # jacobian wrt to regularization parameter
        jacobian_regularization = -self.inverseKernelRidgeDesign.dot(self.fittedweight) 
        self.jacobian_regularization = np.copy(jacobian_regularization)
        resultDict.update({'Jacobian_Regularization':jacobian_regularization})
 
        if self.squareExponentialKernel is not None:
            
            jacobian_SqExp_EnsWeight = -self.inverseKernelRidgeDesign.dot(self.squareExponentialKernel.kernelmatrix).dot(self.fittedweight)             
            self.jacobian_SqExp_EnsWeight = np.copy(jacobian_SqExp_EnsWeight)
            resultDict.update({'Jacobian_SQEXP_ensweight':jacobian_SqExp_EnsWeight})
            
            jacobian_SqExp_kernelscale = -self.inverseKernelRidgeDesign.dot(self.weight_squareExponentialKernel * self.squareExponentialKernel.kernelmatrixDerivative).dot(self.fittedweight)
            self.jacobian_SqExp_kernelscale = np.copy(jacobian_SqExp_kernelscale)
            resultDict.update({'Jacobian_SQEXP_kernelscale':jacobian_SqExp_kernelscale})
            
        if self.periodicityKernel is not None:
            
            jacobian_PRD_EnsWeight = -self.inverseKernelRidgeDesign.dot(self.periodicityKernel.kernelmatrix).dot(self.fittedweight)
            self.jacobian_PRD_EnsWeight = np.copy(jacobian_PRD_EnsWeight)
            resultDict.update({'Jacobian_PRD_ensweight':jacobian_PRD_EnsWeight})
            
            jacobian_PRD_kernelscale = -self.inverseKernelRidgeDesign.dot(self.weight_periodicityKernel*self.periodicityKernel.kernelmatrixDerivative_scale).dot(self.fittedweight)
            self.jacobian_PRD_kernelscale = np.copy(jacobian_PRD_kernelscale)
            resultDict.update({'Jacobian_PRD_kernelscale':jacobian_PRD_kernelscale})

            jacobian_PRD_period = -self.inverseKernelRidgeDesign.dot(self.weight_periodicityKernel*self.periodicityKernel.kernelmatrixDerivative_period).dot(self.fittedweight)
            self.jacobian_PRD_period = np.copy(jacobian_PRD_period)
            resultDict.update({'Jacobian_PRD_period':jacobian_PRD_period})
            
        if self.ouKernel is not None:
            
            jacobian_OU_EnsWeight = -self.inverseKernelRidgeDesign.dot(self.ouKernel.kernelmatrix).dot(self.fittedweight)
            self.jacobian_OU_EnsWeight = np.copy(jacobian_OU_EnsWeight)
            resultDict.update({'Jacobian_OU_ensweight':jacobian_OU_EnsWeight})
 
            jacobian_OU_kernelscale = -self.inverseKernelRidgeDesign.dot(self.weight_ouKernel*self.ouKernel.kernelmatrixDerivative).dot(self.fittedweight)
            self.jacobian_OU_kernelscale = np.copy(jacobian_OU_kernelscale)
            resultDict.update({'Jacobian_OU_kernelscale':jacobian_OU_kernelscale})

        if self.linearKernel is not None:
            
            jacobian_Linear_EnsWeight  = -self.inverseKernelRidgeDesign.dot(self.linearKernel.kernelmatrix).dot(self.fittedweight)
            self.jacobian_Linear_EnsWeight  = np.copy(jacobian_Linear_EnsWeight)
            resultDict.update({'Jacobian_Linear_ensweight':jacobian_Linear_EnsWeight})
            
        
        if self.automaticrelavanceKernel is not None:
            jacobian_ARD_EnsWeight = -self.inverseKernelRidgeDesign.dot(self.automaticrelavanceKernel.kernelmatrix).dot(self.fittedweight)
            self.jacobian_ARD_EnsWeight = np.copy(jacobian_ARD_EnsWeight)
            resultDict.update({'Jacobian_ARD_ensweight':jacobian_ARD_EnsWeight})
            # compute jacobian w.r.t componentwise scales
            [N,p] = np.shape(self.training_X)
            # jacobian of theta w.r.t componentwise scale is N-by-p
            jacobian_ARD_kernelscale = np.zeros((N,p))
            for l in range(p):
                featureDiff = (self.training_X[:,l][:,np.newaxis] - self.training_X[:,l])**2
                # N-by-N
                kernelmatrixDeriv = -self.automaticrelavanceKernel.kernelmatrix * featureDiff
                jacobian_ARD_kernelscale[:,l] = - self.inverseKernelRidgeDesign.dot(self.weight_automaticrelavanceKernel*kernelmatrixDeriv).dot(self.fittedweight).reshape(N)
              
            self.jacobian_ARD_kernelscale = np.copy(jacobian_ARD_kernelscale)
            resultDict.update({'Jacobian_ARD_kernelscale':jacobian_ARD_kernelscale})
            
        return resultDict
    
    
    def computeHyperGradient(self,xnew,tnew,ynew,ytrue):
        
        if self.fittedweight is None:
            raise ValueError('must fit the model before predict')

        [N,p] = self.training_X.shape
        ensembleKernel = np.zeros(N)            
        if self.squareExponentialKernel is not None:
            sqExp = self.squareExponentialKernel.computeKernel(self.training_X,xnew,requireDerivative=True)
            sqExpKernel = sqExp.get('Kernel').reshape(N)
            sqExpKernelDerivt = sqExp.get('KernelDerivative').reshape(N)
            ensembleKernel += self.weight_squareExponentialKernel * sqExpKernel
            
            
        if self.periodicityKernel is not None:
            prd = self.periodicityKernel.computeKernel(self.training_T,tnew,requireDerivative=True,quasi=self.quasiperiodic)
            prdKernel = prd.get('Kernel').reshape(N)
            prdKernelDerivt_kernelscale = prd.get('KernelDerivativeOfScale').reshape(N)
            prdKernelDerivt_period = prd.get('KernelDerivativeOfPeriod').reshape(N)
            ensembleKernel += self.weight_periodicityKernel * prdKernel

        if self.ouKernel is not None:
            ou = self.ouKernel.computeKernel(self.training_T,tnew,requireDerivative=True)
            ouKernel = ou.get('Kernel').reshape(N)
            ouKernelDerivt = ou.get('KernelDerivative').reshape(N)
            ensembleKernel += self.weight_ouKernel * ouKernel
            
        if self.linearKernel is not None:
            linear = self.linearKernel.computeKernel(self.training_X,xnew)
            linearKernel = linear.get('Kernel').reshape(N)
            ensembleKernel += self.weight_linearKernel * linearKernel
            
        if self.automaticrelavanceKernel is not None:
            ard = self.automaticrelavanceKernel.computeKernel(self.training_X,xnew,requireDerivative=True)
            ardKernel = ard.get('Kernel').reshape(N)
            # kernel derivative w.r.t componentwise scale is p-by-N
            ardKernelDerivt = ard.get('KernelDerivative')
            ensembleKernel += self.weight_automaticrelavanceKernel * ardKernel
            
            
        resultDict = {}
        hpg_regularizer = -2*(ytrue-ynew) * ensembleKernel.dot(self.jacobian_regularization);
        resultDict.update({'Regularizer':hpg_regularizer})            
        
        if self.squareExponentialKernel is not None:
            hpg_SQEXP_EnsWeight = -2*(ytrue-ynew)*(sqExpKernel.dot(self.fittedweight)
                                                    + ensembleKernel.dot(self.jacobian_SqExp_EnsWeight))
            hpg_SQEXP_kernelscale = -2*(ytrue-ynew)*(self.weight_squareExponentialKernel * sqExpKernelDerivt.dot(self.fittedweight)
                                                    + ensembleKernel.dot(self.jacobian_SqExp_kernelscale)) 
            resultDict.update({'SQEXP_EnsembleWeight':hpg_SQEXP_EnsWeight})
            resultDict.update({'SQEXP_KernelScale':hpg_SQEXP_kernelscale})


        if self.periodicityKernel is not None:
            hpg_PRD_EnsWeight = -2*(ytrue-ynew)*(prdKernel.dot(self.fittedweight)
                                                    + ensembleKernel.dot(self.jacobian_PRD_EnsWeight))
            hpg_PRD_kernelscale = -2*(ytrue-ynew)*(self.weight_periodicityKernel * prdKernelDerivt_kernelscale.dot(self.fittedweight)
                                                    + ensembleKernel.dot(self.jacobian_PRD_kernelscale)) 
            hpg_PRD_period = -2*(ytrue-ynew) * (self.weight_periodicityKernel * prdKernelDerivt_period.dot(self.fittedweight) 
                                                + ensembleKernel.dot(self.jacobian_PRD_period))
            resultDict.update({'PRD_EnsembleWeight':hpg_PRD_EnsWeight})
            resultDict.update({'PRD_KernelScale':hpg_PRD_kernelscale})
            resultDict.update({'PRD_Period':hpg_PRD_period})
 
        if self.ouKernel is not None:
            hpg_OU_EnsWeight = -2*(ytrue-ynew)*(ouKernel.dot(self.fittedweight)
                                                + ensembleKernel.dot(self.jacobian_OU_EnsWeight))
            hpg_OU_kernelscale = -2*(ytrue-ynew)*(self.weight_ouKernel * ouKernelDerivt.dot(self.fittedweight)
                                                + ensembleKernel.dot(self.jacobian_OU_kernelscale))
            resultDict.update({'OU_EnsembleWeight':hpg_OU_EnsWeight})
            resultDict.update({'OU_KernelScale':hpg_OU_kernelscale})
            
        if self.linearKernel is not None:
            hpg_Linear_EnsWeight = -2*(ytrue-ynew)*(linearKernel.dot(self.fittedweight)
                                                + ensembleKernel.dot(self.jacobian_Linear_EnsWeight))
            resultDict.update({'Linear_EnsembleWeight':hpg_Linear_EnsWeight})
            
            
        if self.automaticrelavanceKernel is not None:
            hpg_ARD_EnsWeight = -2*(ytrue-ynew)*(ardKernel.dot(self.fittedweight)
                                                 + ensembleKernel.dot(self.jacobian_ARD_EnsWeight))
            hpg_ARD_kernelscale = -2*(ytrue-ynew)*(self.weight_automaticrelavanceKernel * ardKernelDerivt.dot(self.fittedweight)
                                                 + ensembleKernel.dot(self.jacobian_ARD_kernelscale).reshape(p,1) )
            
            resultDict.update({'ARD_EnsembleWeight':hpg_ARD_EnsWeight})
            resultDict.update({'ARD_KernelScale':hpg_ARD_kernelscale})
            
        return resultDict
    
    
    
    def approximateHyperGradient(self,xnew,tnew,ytrue,ensembleWeight):
        
        """
        deprecated, debuging and verification purpose only

        """
        if self.fittedweight is None:
            raise ValueError('must fit the model before predict')
         
        epsilon = 0.0000001
        if self.squareExponentialKernel is not None:           
            kernelscalePerturbed = self.squareExponentialKernel.kernelscale 
            sqExpPerturbed = SqExpKernel(kernelscalePerturbed)        
        else:
            sqExpPerturbed = None

        if self.periodicityKernel is not None:            
            kernelscalePerturbed = self.periodicityKernel.kernelscale 
            periodPerturbed = self.periodicityKernel.period  
            prdPerturbed = PeriodicKernel(kernelscalePerturbed,periodPerturbed)
        else:
            prdPerturbed = None
            
        if self.ouKernel is not None:            
            kernelscalePerturbed = self.ouKernel.kernelscale 
            ouPerturbed = OUprocessKernel(kernelscalePerturbed)
        else:
            ouPerturbed = None
 
        ensembleWeight = ensembleWeight 
        mklRregressorPerturbed = MKLregression(p=self.p, lambda_reg=self.lambda_reg+epsilon, 
                                               squareExponentialKernel=sqExpPerturbed,periodicityKernel=prdPerturbed,ouKernel=ouPerturbed,linearKernel=self.linearKernel,
                                               training_X=self.training_X,training_T=self.training_T,training_Y=self.training_Y,
                                               ensembleWeight = ensembleWeight )
           
        
        ynew = self.predict(xnew,tnew)
        print('old pred:')
        print(ynew)
        error = (ytrue - ynew)**2

        mklRregressorPerturbed.fit(self.training_X,self.training_T,self.training_Y)                
        ynewPertrubed  = mklRregressorPerturbed.predict(xnew,tnew)
        print('perturbed scale pred')
        print(ynewPertrubed)
        errorPertubed = (ytrue - ynewPertrubed)**2
        
        approximatehpg = (errorPertubed - error)/epsilon
        print('apprximate FD hyper gradient')
        print(approximatehpg)
        
        
        
        
        """
        perturbedKernelMatrix = mklRregressorPerturbed.squareExponentialKernel.kernelmatrix
        approx_kernelmatrixDeriv = (perturbedKernelMatrix - self.squareExponentialKernel.kernelmatrix)/epsilon 
        error_kernelmatrixDeriv = np.linalg.norm(approx_kernelmatrixDeriv-self.squareExponentialKernel.kernelmatrixDerivative,'fro')
        print("error in kernel matrix derivative")
        print(error_kernelmatrixDeriv)
        
        approx_jacobian = (mklRregressorPerturbed.fittedweight-self.fittedweight)/epsilon
        print("error in jacobian")
        #print(self.jacobian_SqExp_kernelscale)
        #print(approx_jacobian)
        print(np.linalg.norm(approx_jacobian-self.jacobian_SqExp_kernelscale))
        
        sqExpPerturbed = mklRregressorPerturbed.squareExponentialKernel.computeKernel(self.training_X,xnew,requireDerivative = True)
        sqExpKernelPerturbed = sqExpPerturbed.get('Kernel')
        
        sqExp = self.squareExponentialKernel.computeKernel(self.training_X,xnew,requireDerivative = True)
        sqExpDeriv = sqExp.get('KernelDerivative')
        sqExpKernel = sqExp.get('Kernel')
        approx_kernelDerivative = (sqExpKernelPerturbed-sqExpKernel)/epsilon
        print('error in kernel derivative')
        #print(approx_kernelDerivative)
        #print(sqExpDeriv)
        print(np.linalg.norm(approx_kernelDerivative-sqExpDeriv))
        """
        

                                 