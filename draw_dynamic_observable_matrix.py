# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:44:08 2019

@author: Jerry
"""
import tensorflow as tf
from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 0.25
avgDegree = 3
temperature = 0.00
totalSteps = 120
binDist = 20
numSim = 70

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
                  withExt=False)
recorder = Recoder(dm,
                   totalSteps,
                   binDist)
simulator = MonteCarloSimulator(dm,
                                totalSteps,
                                numSim)
matShape = (totalSteps, dm.P)
compute_observable_matrix = Recoder.observable_matrix(recorder,
                                                      simulator,
                                                      matShape,
                                                      True)(Recoder.hamming_overlap)

for n in range(totalSteps):
    simulator.simulation(runByStep=True)

compute_observable_matrix()
resMat = recorder.disMat
recorder.disMat = resMat[:1,:]
recorder.draw_distance_matrix()
recorder.disMat = resMat
recorder.draw_distance_matrix()
recorder.disMat = resMat[-10:,:]
recorder.draw_distance_matrix()