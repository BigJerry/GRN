# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:14:31 2019

@author: Jerry
"""
import tensorflow as tf
from GRN import DynamicModel,Recoder

N = 2000
alpha = 0.09
avgDegree = 3
temperature = 0.0
totalSteps = 50
binDist = 20
runTimes = 1

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
                  withExt=True)
re = Recoder(dm,totalSteps,binDist,False)
multi_overlap_observation = Recoder.multi_observation(55)(Recoder.hamming_overlap)

for n in range(runTimes):
    multi_overlap_observation(re)
    re.plot()
    print("activity level: ",re.activity)