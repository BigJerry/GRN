# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:32:48 2019

@author: Jerry
"""

import tensorflow as tf
import numpy as np
from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 1
avgDegree = 3
gamma = 0.7
limitedRegimePatNum = 6
denseRegimeExpRate = 0.6
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
                  withExt=True)
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
                                                     needPatternInfo=True)(Recoder.mattis_overlap_)
#        re.binDistTimes = 100
#        re.distribution_interaction()
#        figName = 'distribution_interaction_'+reg_p+'_'+reg_c+'.png'
#        re.ax.figure.savefig(figFile+figName)
        
        re.binDistTimes = 100
        simulator.simRes = []
        for k in range(totalSteps):
            simulator.simulation(runByStep=True)
            simulator.simRes.append(simulator.dynModel.sysArr)
       
        compute_EA_overlap_matrix()
        re.fx = re.disMat[-1:,:].flatten()
        varLastSteps = np.var(re.disMat[-10,0].flatten())
        mean = np.mean(re.fx)
        var = np.var(re.fx)
        textstr = '\n'.join((
            r'$\mu=%.9f$' % (mean, ),
            r'$\sigma=%.9f$' % (var, ),
            r'variance over last ten steps=%.9f' % (varLastSteps, )))
        re.bar()
        re.ax.text(0.05,0.95,textstr,transform=re.ax.transAxes,fontsize=14,verticalalignment='top')
        figName = 'distanceMatrix_'+reg_p+'_'+reg_c+'.png'
        re.ax.figure.savefig(figFile+figName)
        

        
        print("Exiting current regime...")