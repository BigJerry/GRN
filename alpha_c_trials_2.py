# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:19:44 2019

@author: Jerry
"""

import tensorflow as tf
from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 1
avgDegree = 3
gamma = 0.7
limitedRegimePatNum = 6
denseRegimeExpRate = 0.3
temperature = 0.00
totalSteps = 50
binDist = 100
numSim = 100
regimes_P = ['saturated','sub_extensive','limited']
regimes_C = ['sparse','extreme_dilution','dense']
figFile = r'E:\KCL\FinalProject\figures/'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95

dm = DynamicModel(temperature,
                  N,
                  avgDegree,
                  alpha,
                  ifDeco=False,
                  tfConfig=config,
                  gamma=gamma,
                  withExt=True)
dm1 = DynamicModel(temperature,
                  N,
                  avgDegree,
                  alpha,
                  ifDeco=False,
                  tfConfig=config,
                  gamma=gamma,
                  withExt=False)
re = Recoder(dm,totalSteps,binDist,ifSync=False)
multi_overlap_observation = Recoder.multi_observation(55)(Recoder.hamming_overlap)

for reg_p in regimes_P:
    for reg_c in regimes_C:
        if reg_p is not 'limited':
            dm.regime_P(reg_p)
        else:
            dm.regime_P(reg_p,limitedRegimePatNum)
            
        if reg_c is not 'dense':
            dm.regime_C(reg_c)
        else:
            dm.regime_C(reg_c,denseRegimeExpRate)

        dm._init_memMat()
        print("now is in "+reg_p+" and "+reg_c+" regime...")

        multi_overlap_observation(re)

        re.plot()
        figName = 'distanceMatrix_'+reg_p+'_'+reg_c+'.png'
        re.fig.savefig(figFile+figName)
        print("Exiting current regime...")