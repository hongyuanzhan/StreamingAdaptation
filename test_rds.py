#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:45:22 2017

@author: Zhan
"""


import argparse
import os
import numpy as np
import pandas as pd 
from mklr import PeriodicKernel
from mklr import ARDKernel
from bkt import RollingBackTest

parser = argparse.ArgumentParser("back testing for with periodic random search")
parser.add_argument("timeSeriesCSV",type=str,help="input time series CSV file")
parser.add_argument("reselectionperiod",type=int,help="hyperparameter reselection period")
parser.add_argument("numberDraws",type=int,help="number of randomized draws for random search")


args = parser.parse_args()
numberDraws = args.numberDraws
selectionPeriod = args.reselectionperiod
df = pd.read_csv(args.timeSeriesCSV,header=None)
df.columns = ['flow']
scaling = 1/100
flowTimeSeries = scaling * df['flow'].as_matrix()
flowTimeSeries[flowTimeSeries<0] = np.nan
# learning setup         
timeSeriesLength = flowTimeSeries.shape[0]
order = 20
trainingWindow =96*30
retrainPeriod = 96  
horizon = 1
# hyperparameter update
# construct MKL
scale = 0.00015
period = 96*2
prdKernel = PeriodicKernel(scale,period)
kernelWidth = scale * np.ones(order)
autoRelevKernel = ARDKernel(componentwiseScale=kernelWidth)
lambda_reg = 0.3
# hyperparameter bounds and learning rate
regLB = lambda_reg /10
regUB = lambda_reg * 10
prdLB = period / 2
prdUB = period * 7
scaleLB = scale / 100
scaleUB = scale * 100
ensemblweight = np.array([0, 1/2, 0, 0, 1/2],dtype=np.float)

"""
start test
"""
backtester = RollingBackTest(flowTimeSeries, order, trainingWindow= trainingWindow, horizon=horizon,
                 retrainPeriod=retrainPeriod,selectionPeriod=selectionPeriod,hyperParameterUpdateOption='randomsearch',
                 lambda_reg=lambda_reg, regLB=regLB, regUB=regUB,
                 periodicityKernel=prdKernel,prdPeriodLB=prdLB, prdPeriodUB=prdUB,prdScaleLB=scaleLB,prdScaleUB=scaleUB,
                 automaticrelavanceKernel=autoRelevKernel, autoRelevanceLB=scaleLB, autoRelevanceUB=scaleUB,
                 ensembleWeight= ensemblweight)
result = backtester.backTesting(centerdata=True,numberDraws=numberDraws )

resultarray = np.array([result.get('mse'), result.get('mae'),result.get('mpe')])
predictionarray = np.array([result.get('true'),result.get('predict')])
paramarray = np.array(result.get('paramHistory'))

# 
nppath = os.getcwd() + '/' + args.timeSeriesCSV.replace('.csv','').replace('data/','') + '/' + str(horizon) + 'step/randomsearch/np/'  
if not os.path.exists(nppath):
    os.makedirs(nppath)

nppath = nppath + str(numberDraws)
np.save(nppath + '_error',resultarray)
np.save(nppath + '_prediction',predictionarray)
np.save(nppath + '_param',paramarray)

confipath = os.getcwd() + '/' + args.timeSeriesCSV.replace('.csv','').replace('data/','') + '/' + str(horizon) +'step/randomsearch/config/'  
if not os.path.exists(confipath):
    os.makedirs(confipath)

confipath = confipath + str(numberDraws) + '_testconfig.o'
with open(confipath,'a') as confiFile:
    confiFile.write('input time series CSV:' + args.timeSeriesCSV + '\n')
    confiFile.write('number of random search draws:' + str(args.numberDraws) + '\n')
    confiFile.write('time series order:' + str(order) + '\n')
    confiFile.write('training window:' + str(trainingWindow) + '\n' )
    confiFile.write('retrain period:' + str(retrainPeriod) + '\n')
    confiFile.write('reselection period:' + str(selectionPeriod) + '\n')

    confiFile.write('horizon:' + str(horizon) + '\n')
    
    confiFile.write('kernel width default:' + str(scale) + '\n' )
    confiFile.write('kernel width LB:' + str(scaleLB) + '\n')
    confiFile.write('kernel width UB:' + str(scaleUB) + '\n')

    confiFile.write('regularization constant default' + str(lambda_reg) + '\n')
    confiFile.write('regularization constant LB:' + str(regLB) + '\n')
    confiFile.write('regularization constant UB:' + str(regUB) + '\n')

    confiFile.write('kernel period default' + str(period) + '\n')
    confiFile.write('kernel period LB' + str(prdLB) + '\n')
    confiFile.write('kernel period UB' + str(prdUB) + '\n')

    confiFile.write('total runing time:' + str(result.get('totaltime')) + '\n')
    confiFile.write('hyperparameter search time:' + str(result.get('searchtime')) + '\n')     
    confiFile.write('mse:' + str(result.get('mse')[0]) + '\n')
    confiFile.write('mae:' + str(result.get('mae')[0]) + '\n')
    confiFile.write('mpe:' + str(result.get('mpe')[0]) + '\n')
                                                     

    

