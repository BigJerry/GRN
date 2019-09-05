# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:57:18 2019

@author: Jerry
"""

import tensorflow as tf
from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 0.1
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
figFile = r'E:\KCL\FinalProject\figures\withExt\trials_1\alpha_1_C_3/'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95

dm = DynamicModel(temperature,
                  N,
                  avgDegree,
                  alpha,
                  tfConfig=config,
                  gamma=gamma,
                  withExt=False)
re = Recoder(dm,totalSteps,binDist,False)
simulator = MonteCarloSimulator(dm,totalSteps,numSim)

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
        
        matShape = (totalSteps, dm.P)
        compute_EA_overlap_matrix = Recoder.observable_matrix(re,
                                                     simulator,
                                                     matShape,
                                                     needPatternInfo=True)(Recoder.hamming_overlap)
        simulator.simRes = []
        for k in range(totalSteps):
            simulator.simulation(runByStep=True)
            simulator.simRes.append(simulator.dynModel.sysArr)
       
        compute_EA_overlap_matrix()
        re.disMat = re.disMat[-10:,:].reshape((10,dm.P))
        re.draw_distance_matrix()
        figName = 'distanceMatrix_'+reg_p+'_'+reg_c+'.png'
#        re.fig.savefig(figFile+figName)

        print("Exiting current regime...")
