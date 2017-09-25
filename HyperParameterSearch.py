#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:18:24 2017

@author: Zhan
"""
from RollingBackTestKRR import RollingBackTestKRR
import numpy as np

class HyperParameterSearch:
    
    def __init__(self,timeSeriesData, order, trainingWindow, horizon, retrainPeriod \
                 ,kernelWidthLowerBound, kernelWidthUpperBound, reglrConstLowerBound, reglrConstUpperBound, numberRandomDraws, numberGridPoints):
        
        self.timeSeriesData = timeSeriesData
        self.order = order
        self.trainingWindow = trainingWindow
        self.horizon = horizon
        self.retrainPeriod = retrainPeriod
        self.kernelWidthLowerBound = kernelWidthLowerBound
        self.kernelWidthUpperBound = kernelWidthUpperBound
        self.reglrConstLowerBound = reglrConstLowerBound
        self.reglrConstUpperBound= reglrConstUpperBound
        self.numberRandomDraws = numberRandomDraws
        self.numberGridPoints = numberGridPoints
        self.bestMSE = np.inf
        self.bestKernelWidth = np.zeros(self.horizon)
        
    def RandomSearch(self,outputPath):
        
        
        backtester = RollingBackTestKRR(self.timeSeriesData, self.order, self.trainingWindow, self.horizon)
        
        for trial in range(self.numberRandomDraws):
            kernelWidth = np.random.uniform(self.kernelWidthLowerBound,self.kernelWidthUpperBound,size=self.order)
            print("draws for componentwise kernel width: ",kernelWidth)
            testScores = backtester.backTesting(self.order,'fixed',componentKernelWidth=kernelWidth)
            if testScores.get('mse')[0] < self.bestMSE :
                self.bestMSE = testScores.get('mse')[0]
                self.bestKernelWidth = kernelWidth
            
            
            with open(outputPath,'a') as resultfile:
                resultfile.write(str(testScores.get('mse')[0]))
                resultfile.write(',')
                resultfile.write(str(testScores.get('mae')[0]))
                resultfile.write(',')
                resultfile.write(str(testScores.get('mpe')[0]))
                resultfile.write(',')
                for p in range(self.order):
                    resultfile.write(str(kernelWidth[p]))
                    resultfile.write(',')
                resultfile.write('\n')
           
        returnitems = {}
        returnitems.update({'bestmse':self.bestMSE})
        returnitems.update({'bestKernelWidth':self.bestKernelWidth})
        return returnitems    

                
    def GridSearch(self,outputPath):
        
        backtester = RollingBackTestKRR(self.timeSeriesData, self.order, self.trainingWindow, self.horizon)
        grids = np.linspace(self.kernelWidthLowerBound,self.kernelWidthUpperBound,self.numberGridPoints)
        for g in grids:
            
            kernelWidth = g * np.ones(self.order)
            testScores = backtester.backTesting(self.order,'fixed',componentKernelWidth=kernelWidth)
            if testScores.get('mse')[0] < self.bestMSE :
                self.bestMSE = testScores.get('mse')[0]
                self.bestKernelWidth = kernelWidth
                
            with open(outputPath,'a') as resultfile:
                resultfile.write(str(testScores.get('mse')[0]))
                resultfile.write(',')
                resultfile.write(str(testScores.get('mae')[0]))
                resultfile.write(',')
                resultfile.write(str(testScores.get('mpe')[0]))
                resultfile.write(',')
                for p in range(self.order):
                    resultfile.write(str(kernelWidth[p]))
                    resultfile.write(',')
                resultfile.write('\n')           
           
        returnitems = {}
        returnitems.update({'bestmse':self.bestMSE})
        returnitems.update({'bestKernelWidth':self.bestKernelWidth})
        return returnitems             

        
                
        