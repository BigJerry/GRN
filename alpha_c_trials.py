# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:11:06 2019

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
figFile = r'E:\KCL\FinalProject\figures\withoutExt\trials_0\alpha_1_C_3/'

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
                  withExt=True)
re = Recoder(dm,totalSteps,binDist,False)
simulator = MonteCarloSimulator(dm,totalSteps,numSim)
matShape = (simulator.numSim,simulator.numSim)
compute_q_matrix = Recoder.observable_matrix(re,simulator,matShape)(Recoder.EA_overlap)

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

        simulator.simulation()
        compute_q_matrix()

        re.draw_distance_matrix()
        figName = 'distanceMatrix_'+reg_p+'_'+reg_c+'.png'
        re.fig.savefig(figFile+figName)

        re.distribution_q_matrix(numSim)
        figName = 'distribution_'+reg_p+'_'+reg_c+'.png'
        re.fig.savefig(figFile+figName)
        print("Exiting current regime...")
