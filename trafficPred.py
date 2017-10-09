#!/usr/bin/env python3componentKernelWidth
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:20:51 2017

@author: Zhan
"""
import numpy as np
import pandas as pd 
from mklr import PeriodicKernel
from mklr import ARDKernel
from mklr import MKLregression
from sklearn.kernel_ridge import KernelRidge
from stmpsgd import OPGD
from bkt import RollingBackTest

df = pd.read_csv('data/716078_20170101_20170531_15mins.csv',header=None)
df.columns = ['flow']
scaling = 1/100

flowTimeSeries = scaling * df['flow'].as_matrix()
flowTimeSeries[flowTimeSeries<0] = np.nan
testTimeSeries = flowTimeSeries[96*60+1:]


timeSeriesLength = testTimeSeries.shape[0]
order = 20
trainingWindow =96*30
retrainPeriod = 96
selectionPeriod = 96*30
horizon = 1
# hyperparameter update
# construct MKL
scale = 0.00015
period = 96*2
prdKernel = PeriodicKernel(scale,period)
kernelWidth = 0.00015 * np.ones(order)
autoRelevKernel = ARDKernel(componentwiseScale=kernelWidth)
lambda_reg = 0.3
# hyperparameter bounds and learning rate
regLB = lambda_reg /10
regUB = lambda_reg * 10
prdLB = period / 10
prdUB = period * 10
scaleLB = scale / 100
scaleUB = scale * 100
ensemblweight = np.array([0, 1/2, 0, 0, 1/2],dtype=np.float)
"""
stepsize = 0.00005
backtester = RollingBackTest(testTimeSeries, order, trainingWindow= trainingWindow, horizon=horizon,
                 retrainPeriod=retrainPeriod,selectionPeriod=selectionPeriod,hyperParameterUpdateOption='sgd',
                 lambda_reg=lambda_reg, regLB=regLB, regUB=regUB,
                 periodicityKernel=prdKernel,prdPeriodLB=prdLB, prdPeriodUB=prdUB,prdScaleLB=scaleLB,prdScaleUB=scaleUB,
                 automaticrelavanceKernel=autoRelevKernel, autoRelevanceLB=scaleLB, autoRelevanceUB=scaleUB,
                 ensembleWeight= ensemblweight)
result_sgd0 = backtester.backTesting(centerdata=True,stepsize=stepsize) 

stepsize = 0.00005
maxit = 30
backtester = RollingBackTest(testTimeSeries, order, trainingWindow= trainingWindow, horizon=horizon,
                 retrainPeriod=retrainPeriod,selectionPeriod=selectionPeriod,hyperParameterUpdateOption='hoag',
                 lambda_reg=lambda_reg, regLB=regLB, regUB=regUB,
                 periodicityKernel=prdKernel,prdPeriodLB=prdLB, prdPeriodUB=prdUB,prdScaleLB=scaleLB,prdScaleUB=scaleUB,
                 automaticrelavanceKernel=autoRelevKernel, autoRelevanceLB=scaleLB, autoRelevanceUB=scaleUB,
                 ensembleWeight= ensemblweight)
result_hoag = backtester.backTesting(centerdata=True,stepsize=stepsize,maxit=maxit) 
"""
backtester = RollingBackTest(testTimeSeries, order, trainingWindow= trainingWindow, horizon=horizon,
                 retrainPeriod=retrainPeriod,selectionPeriod=selectionPeriod,hyperParameterUpdateOption='randomsearch',
                 lambda_reg=lambda_reg, regLB=regLB, regUB=regUB,
                 periodicityKernel=prdKernel,prdPeriodLB=prdLB, prdPeriodUB=prdUB,prdScaleLB=scaleLB,prdScaleUB=scaleUB,
                 automaticrelavanceKernel=autoRelevKernel, autoRelevanceLB=scaleLB, autoRelevanceUB=scaleUB,
                 ensembleWeight= ensemblweight)
numberDraws = 10
result_random = backtester.backTesting(centerdata=True,numberDraws=10 )
