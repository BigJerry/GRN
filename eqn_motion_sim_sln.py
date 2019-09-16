# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:23:26 2019

@author: Jerry
"""

from GRN import DynamicModel
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95

class Observable(object):
    def __init__(self,model):
        self.model = model
        
    class Quant(object):
        def __init__(self):
            pass
    
    @property
    def overlap(self):
        assert(self.model.pat['ita'].shape[1]==self.model.sysArr.shape[0])
        ret = np.matmul(self.model.pat['ita'], self.model.sysArr) / self.model.C_mu
        assert(ret.shape[1]==1)
        
        return ret
    
    @property
    def activity(self):
        return np.sum(self.model.sysArr) / self.model.N
    
    def hamming_dist(self,arr1,arr2):
        ret = np.abs(arr1 - arr2)
        _ = np.ones((1,self.model.N),dtype=np.float32)
        ret = np.inner(ret.T, _ )[0,0]
        return ret/self.model.N

if __name__ == '__main__':
    fname = r'E:\temptemptemp\npy\hehe.npy'
    #model-related parameters
    delta = 1/3
    gamma = 0.7
    alpha = 1
    
    patNumFiniteReg = 5
    
    T = 1.8
    dims = 5000
    avgDegree = 12
    #simulation-related parameters
    numSim = 50
    itNum = 20                                                                  #total time steps to run
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
    dm.load(fname)
    
    initState = dm.pat['ita'][0,:].T.reshape((dm.N,1))
#    initState = np.zeros((dm.N,1),dtype=np.float32)
#    for i in range(8):
#        initState[i,0] = 1.0

    obs = Observable(dm)
    obs.Quant.Traj_m = np.zeros((dm.P,itNum),dtype=np.float32)
    obs.Quant.Traj_a = np.zeros((1,itNum),dtype=np.float32)
    
    for n in range(numSim):
        t_s = time.time()
        ################################
#        dm.generate()
        
        dm.init_system_with_value(initState)
        
        traj_m = obs.overlap
        traj_a = np.array([obs.activity],dtype=np.float32)
        for i in range(itNum-1):
            dm.update()
            m = obs.overlap
            a = obs.activity
            
            traj_m = np.hstack((traj_m, m))
            traj_a = np.append(traj_a, a)
        
        obs.Quant.Traj_m += traj_m
        obs.Quant.Traj_a += traj_a
        #################################
        t_e = time.time()
        tt = t_e-t_s
        numRem = numSim-n-1
        if n%10 == 0:
            print("Now in No.%d simulation, %d simulations remain. %fs is estimated to be needed"%(n+1,numRem,numRem*tt))
    
    obs.Quant.Traj_m /= numSim
    obs.Quant.Traj_a /= numSim
    print(cfg)
    for n in range(5):
        plt.plot(np.arange(itNum),obs.Quant.Traj_m[n,:].reshape((itNum,)))
#        print(obs.Quant.Traj_m[n,:])
    f = plt.plot(np.arange(itNum),obs.Quant.Traj_a.reshape((itNum,)))
#    print(obs.Quant.Traj_a)



