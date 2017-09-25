#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:49:02 2017

@author: Zhan
"""

import argparse
import os
import numpy as np
import pandas as pd 
from HyperParameterSearch import HyperParameterSearch
from RollingBackTestKRR import RollingBackTestKRR

parser = argparse.ArgumentParser("back testing for searching hyper-parameters")
parser.add_argument("timeSeriesCSV",type=str,help="input time series CSV file")
parser.add_argument("beginStep",type=int,help="the begining time steps for backtesting")
parser.add_argument("endStep",type=int,help="the ending time steps for backtesting")
parser.add_argument("numberDraws",type=int,help="number of randomized draws for random search")
parser.add_argument("numberGridPoints",type=int,help="number of enumerating grid points for grid search")
#parser.add_argument("hyperParamOutputFile",type=str,help="file to store the back testing results of different hyper-parameters")
#parser.add_argument("configurationFile",type=str,help="file to store the configuration of the experiment")

args = parser.parse_args()

df = pd.read_csv(args.timeSeriesCSV,header=None)
df.columns = ['flow']
scaling = 1/100

flowTimeSeries = scaling * df['flow'].as_matrix()
flowTimeSeries[flowTimeSeries<0] = np.nan
hyperParamValidationData = flowTimeSeries[args.beginStep:args.endStep+1]
   
timeSeriesLength = hyperParamValidationData.shape[0]
order = 96
trainingWindow =96*30
retrainPeriod = 96
horizon = 1
# hyperparameter bounds and learning rate
kernelLB = 0.00005
kernelUB = 0.001
regConstLB = 0.1
regConstUB = 2
sgdRate = 0.00001


# 
resultDir = 'krr_' + str(horizon) + 'step_' + str(sgdRate)
if not os.path.exists(resultDir):
    os.makedirs(resultDir)
    
    


# searching best hyper-parameter in validation data
hpsearch = HyperParameterSearch(hyperParamValidationData, order, trainingWindow, horizon, retrainPeriod \
             , kernelWidthLowerBound = kernelLB , kernelWidthUpperBound = kernelUB, reglrConstLowerBound = regConstLB, reglrConstUpperBound = regConstUB, \
                 numberRandomDraws = args.numberDraws, numberGridPoints = args.numberGridPoints)

#outputPath = args.hyperParamOutputFile
outputPath = os.getcwd() + '/' + resultDir + '/' + args.timeSeriesCSV.replace('.csv','').replace('data/','') + '_HPsearch.o'
randomSearchResult = hpsearch.RandomSearch(outputPath)
gridSearchResult = hpsearch.GridSearch(outputPath)

# out-of-sample backtesting after selecting a fixed hyperparameters
testingData = flowTimeSeries[args.endStep+1:]
backtester = RollingBackTestKRR(testingData, order, trainingWindow, horizon)
testScores_fixedHP = backtester.backTesting(retrainPeriod,'fixed',componentKernelWidth=hpsearch.bestKernelWidth)

# out-of-sample backtesting with adaptive hyperparameter learning
algo = 'sgd'
testScores_SGDHP = backtester.backTesting(retrainPeriod,algo,componentKernelWidth=hpsearch.bestKernelWidth,kernelWidthLearningRate=sgdRate,kernelWidthLB=kernelLB,kernelWidthUB=kernelUB)



# configuration file
#confiFilePath = args.configurationFile
confiFilePath = os.getcwd() + '/' + resultDir  + '/' + args.timeSeriesCSV.replace('.csv','').replace('data/','') + '_testing.o'

with open(confiFilePath,'a') as confiFile:
    confiFile.write('input time series CSV:' + args.timeSeriesCSV + '\n')
    #confiFile.write('hyper-parameter search result file:' + args.hyperParamOutputFile + '\n')
    confiFile.write('time series order:' + str(order) + '\n')
    confiFile.write('training window:' + str(trainingWindow) + '\n' )
    confiFile.write('retrain period:' + str(retrainPeriod) + '\n')
    confiFile.write('horizon:' + str(horizon) + '\n')
    confiFile.write('hyper-parameter search backtesting starting step:' + str(args.beginStep) + '\n')
    confiFile.write('hyper-parameter search backtesting ending step:' + str(args.endStep) + '\n')
    confiFile.write('kernel width LB:' + str(kernelLB) + '\n')
    confiFile.write('kernel width UB:' + str(kernelUB) + '\n')
    confiFile.write('number of random search draws:' + str(args.numberDraws) + '\n')
    confiFile.write('number of grid points in grid search:' + str(args.numberGridPoints) + '\n')
    confiFile.write('adaptive algorithm:' + algo + '\n')
    confiFile.write('kernel width learning rate:' + str(sgdRate) + '\n')
    confiFile.write('regularization constant LB:' + str(regConstLB) + '\n')
    confiFile.write('regularization constant UB:' + str(regConstUB) + '\n')
    confiFile.write('best validation mse:' + str(hpsearch.bestMSE) + '\n')
    confiFile.write('best validation kernel width:')   
    for i in range(order):
        confiFile.write(str(hpsearch.bestKernelWidth[i]))
        confiFile.write(',')
    confiFile.write('\n')    
    confiFile.write('fixed HP testing mse:' + str(testScores_fixedHP.get('mse')[0]) + '\n')
    confiFile.write('fixed HP testing mae:' + str(testScores_fixedHP.get('mae')[0]) + '\n')
    confiFile.write('fixed HP testing mpe:' + str(testScores_fixedHP.get('mpe')[0]) + '\n')
    confiFile.write('SGD HP testing mse:' + str(testScores_SGDHP.get('mse')[0]) + '\n')
    confiFile.write('SGD HP testing mae:' + str(testScores_SGDHP.get('mae')[0]) + '\n')
    confiFile.write('SGD HP testing mpe:' + str(testScores_SGDHP.get('mpe')[0]) + '\n')
    confiFile.write('SGD improvements in mse:' + str((testScores_fixedHP.get('mse')[0]-testScores_SGDHP.get('mse')[0])/testScores_fixedHP.get('mse')[0] ) + '\n')
    confiFile.write('SGD improvements in mae:' + str((testScores_fixedHP.get('mae')[0]-testScores_SGDHP.get('mae')[0])/testScores_fixedHP.get('mae')[0] ) + '\n')
    confiFile.write('SGD improvements in mpe:' + str((testScores_fixedHP.get('mpe')[0]-testScores_SGDHP.get('mpe')[0])/testScores_fixedHP.get('mpe')[0] ) + '\n')
    
        



