#!/usr/bin/env python3componentKernelWidth
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:20:51 2017

@author: Zhan
"""
import numpy as np
import pandas as pd 
from RollingBackTestKRR import RollingBackTestKRR

df = pd.read_csv('data/716078_20170101_20170531_15mins.csv',header=None)
df.columns = ['flow']
scaling = 1/100

flowTimeSeries = scaling * df['flow'].as_matrix()
flowTimeSeries[flowTimeSeries<0] = np.nan
testTimeSeries = flowTimeSeries[96*60+1:]


timeSeriesLength = testTimeSeries.shape[0]
order = 96
trainingWindow =96*30
retrainPeriod = 96
horizon = 1
# hyperparameter update
# hyperparameter bounds and learning rate
kernelLB = 0.00005
kernelUB = 0.001
regConstLB = 0.1
regConstUB = 2
sgdRate = 0.00005
kernelWidth = 0.00015 * np.ones(order)
algo = 'sgd'



backtester = RollingBackTestKRR(testTimeSeries, order, trainingWindow, horizon)
testScores_fixedHP = backtester.backTesting(retrainPeriod,'fixed',componentKernelWidth=kernelWidth)
testScores_SGDHP = backtester.backTesting(retrainPeriod,algo,componentKernelWidth=kernelWidth,kernelWidthLearningRate=sgdRate,kernelWidthLB=kernelLB,kernelWidthUB=kernelUB)
SGDimprovement = (testScores_fixedHP.get('mse') - testScores_SGDHP.get('mse')) / testScores_fixedHP.get('mse')
print("SGD improvments: " + str(SGDimprovement) + '\n')