# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 18:50:44 2019

@author: Jerry
"""

from GRN import PatternGenerator, DynamicModel
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95

if __name__ == '__main__':
    fname = r'E:\temptemptemp\npy\hehe.npy'
    #model-related parameters
    delta = 1/3
    gamma = 0.7
    alpha = 1
    
    patNumFiniteReg = 5
    
    T = 0.14
    dims = 8000
    avgDegree = 240
    #simulation-related parameters
    numSim = 300
    itNum = 30                                                                  #total time steps to run
    #deciding which regime we're now in
    reg_p = 'limited'
    reg_c = 'sparse'
    #recording configuration we got 
    cfg = {'delta':delta,
           'gamma':gamma,
           'alpha':alpha,
           'T':T,
           'dims':dims,
           'avgDegree':avgDegree,
           'numSim':numSim,
           'itNum':itNum,
           'reg_p':reg_p,
           'reg_c':reg_c}
    
    dm = DynamicModel(T,
                      dims,
                      avgDegree,
                      alpha,
                      gamma=gamma,
                      delta=delta,
                      tfConfig=config)
    dm.regime_P(reg_p,patNumFiniteReg)
    dm.regime_C(reg_c)
    dm.save(fname)
