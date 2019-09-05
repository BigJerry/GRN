# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:19:17 2019

@author: Jerry
"""

import numpy as np
from matplotlib import pyplot as plt
import functools
import tensorflow as tf
#np.seterr(divide='ignore', invalid='ignore')

class PatternGenerator(object):
    def __init__(self,dims,avgDegree,alpha,gamma=1,delta=1):
        self.N = dims
        self.C_mu = avgDegree
        self.alpha = alpha
        self.P = int(alpha*dims)
        self.pat = {'ita':None,
                    'kesi':None}
        self.a_mu = self.C_mu / self.N
        self.gamma = gamma
        self.delta = delta
        
    def regime_P(self,reg,arg=None):
        if reg is 'limited':
            assert(arg is not None)
            self.P = int(arg)
        else:
            self.P = int(eval("self."+reg))
    
    def regime_C(self,reg,arg=None):
        if reg is 'dense':
            assert(arg is not None)
            self.a_mu = arg
        else:
            self.a_mu = eval("self."+reg)

    @property
    def saturated(self):
        return self.alpha * self.N
    @property
    def sub_extensive(self):
        return self.alpha * self.N ** self.delta
    
    @property
    def sparse(self):
        return self.C_mu / self.N
    @property
    def extreme_dilution(self):
        return self.C_mu / (self.N ** self.gamma)
    
    def generate(self):
        self.pat = self.pat.fromkeys(self.pat.keys()) #clear patterns generated last time
        self.pat['ita'] = np.random.binomial(1,self.a_mu,[self.P,self.N]).astype(np.float32)  #generate ita
        self.pat['kesi'] = np.copy(self.pat['ita'])  #generate kesi
        idx = np.where(self.pat['kesi']!=1)
        self.pat['kesi'][idx] = np.float32(-self.a_mu / (1-self.a_mu))
        
    def save(self,path):
        np.save(path,self.pat)
        
    def load(self,path):
        self.pat = np.load(path, allow_pickle=True).item()

class DynamicModel(PatternGenerator):
    def __init__(self,T,*args,ifDeco=False,tfConfig=None,withExt=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.T = T
        self.tfConfig = tfConfig
        self.withExt = withExt
        self.ifDeco = ifDeco
        self.sysArr = self._init_system()
        
        self.generate()
        
    def _generate(self):
        self.pat = self.pat.fromkeys(self.pat.keys()) #clear patterns generated last time
        self.pat['ita'] = np.random.binomial(1,self.a_mu,[self.P,self.N]).astype(np.float32)  #generate ita
        self.pat['kesi'] = np.copy(self.pat['ita'])  #generate kesi
        idx = np.where(self.pat['kesi']!=1)
        self.pat['kesi'][idx] = np.float32(-self.a_mu / (1-self.a_mu))        
        
    #overriding function 'generate()'
    def generate(self):
        self._generate()
        self._init_memMat()

    def _init_memMat(self):
        if not self.ifDeco:
            self.memMat = self._compInteractionMat()
        else:
            self.__compCorrelationMat()
            self.memMat = self.__compInteractionMat()
        print("Done.")
        
    def _init_system(self):
        tf.reset_default_graph()
        return np.mat(np.random.binomial(1,0.5,[self.N,1]).astype(np.float32))
    
    def init_system(self):
        self.sysArr = self._init_system()
        
    def init_system_with_value(self,initValue):
        tf.reset_default_graph()
        self.sysArr = initValue
        
    def __compCorrelationMat(self):
        with tf.name_scope("compute_correlation_matrix"):
            ita = tf.placeholder(shape=[self.P,self.N],dtype=tf.float32)
            itaT = tf.placeholder(shape=[self.N,self.P],dtype=tf.float32)
            _corrMat = tf.matmul(ita,itaT) / self.N
            corrMat = tf.matrix_inverse(_corrMat)
        with tf.Session(config=self.tfConfig) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {ita:self.pat['ita'],
                         itaT:self.pat['ita'].T}
            self.corrMat = np.mat(sess.run(corrMat,feed_dict))
    
    def __compInteractionMat(self):
        with tf.name_scope("compute_interaction_matrix"):
            kesi = tf.placeholder(shape=[self.N,self.P],dtype=tf.float32)
            ita = tf.placeholder(shape=[self.P,self.N],dtype=tf.float32)
            corrMat = tf.placeholder(shape=[self.P,self.P],dtype=tf.float32)
            _memMat = tf.matmul(tf.matmul(kesi,corrMat),ita)
            memMat = _memMat / self.N
        with tf.Session(config=self.tfConfig) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {kesi:self.pat['kesi'].T,
                         ita:self.pat['ita'],
                         corrMat:self.corrMat}
            ret = sess.run(memMat, feed_dict)
        return np.mat(ret)
        
    def _compInteractionMat(self):
        print("calculating memMat...")
        with tf.name_scope("compute_interaction_matrix"):
            kesi = tf.placeholder(shape=[self.N,self.P],dtype=tf.float32)
            ita = tf.placeholder(shape=[self.P,self.N],dtype=tf.float32)
            _kesi = kesi / self.C_mu
            _ita = ita - self.C_mu / self.N
            memMat_with_ext = tf.matmul(_kesi,_ita)
            memMat_without_ext = tf.matmul(_kesi,ita)
        with tf.Session(config=self.tfConfig) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {kesi:self.pat['kesi'].T,
                         ita:self.pat['ita']}
            if self.withExt:
                return np.mat(sess.run(memMat_with_ext,feed_dict))
            else:
                return np.mat(sess.run(memMat_without_ext,feed_dict))
    
    def update(self):
        tf.reset_default_graph()
        with tf.name_scope("update_state"):
            memMat = tf.placeholder(shape=[self.N,self.N],dtype=tf.float32)
            sysArr = tf.placeholder(shape=[self.N,1],dtype=tf.float32)
            res = tf.matmul(memMat,sysArr)
        with tf.Session(config=self.tfConfig) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {memMat:self.memMat,
                         sysArr:self.sysArr}
            ret = np.mat(sess.run(res,feed_dict))
        noise = self.T * np.random.normal(0,1,(self.N,1))
        self.sysArr = np.heaviside(ret + noise,0)

class SequentialModel(DynamicModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def _compInteractionMat(self):
        print("calculating memMat...")
        self.generate()
        memMat = np.mat(np.zeros((self.N,self.N),dtype=np.float32))
        for p in range(self.P):
            if p == self.P-1:
                j = 0
            else:
                j = p+1
            a = np.mat(self.pat['ita'][p,:]).T - self.a_mu
            b = np.mat(self.pat['ita'][j,:]) - self.a_mu
            memMat += np.matmul(a,b) / self.a_mu * (1 - self.a_mu)
        print("Done.")
        return memMat / self.N

class MonteCarloSimulator(object):
    def __init__(self,dynModel,timesteps,numSim):
        self.dynModel = dynModel
        self.totalTimeStep = timesteps
        self.numSim = numSim
        self.simRes = None
    
    def simulation(self,runByStep=False):
        if not runByStep:
            self.simRes = []
            for n in range(self.numSim):
#                print("simulation step: %s ..."%n)
                self.dynModel.init_system()
                for i in range(self.totalTimeStep):
                    self.dynModel.update()
                self.simRes.append(self.dynModel.sysArr)
        else:
            self.dynModel.update()

class Recoder(object):
    def __init__(self,dynModel,timesteps,binDist,ifSync=True):
        self.dynModel = dynModel
        self.totalTimeStep = timesteps
        self.binDistTimes = binDist
        
        self.sync = ifSync
        if(self.sync): self.initSysArr = dynModel.sysArr
   
    def observation(observable):
        @functools.wraps(observable)
        def wrapped_func(*args):
            if not args[0].sync: args[0].dynModel.init_system()
            args[0].fx = []
            for i in range(args[0].totalTimeStep):
                m = observable(args[0],args[1])
                args[0].fx.append(m)
                args[0].dynModel.update()
        return wrapped_func
    
    def multi_observation(numOb):
        def _multi_observation(observable):
            @functools.wraps(observable)
            def wrapped_func(*args):
                toOb = list(range(args[0].dynModel.P))
                np.random.shuffle(toOb)
                toOb = toOb[:numOb]
                if not args[0].sync: 
                    args[0].dynModel.init_system()
                else:
                    args[0].dynModel.init_system_with_value(args[0].initSysArr)
                args[0].fx = dict()
                for p in toOb:
                    args[0].fx[p] = []
                for i in range(args[0].totalTimeStep):
                    for p in toOb:
                        args[0].fx[p].append(observable(args[0],p))
                    args[0].dynModel.update()
            return wrapped_func    
        return _multi_observation
    
    def observation_mean(observable):
        @functools.wraps(observable)
        def wrapped_func(*args):
            if not args[0].sync:
                args[0].dynModel.init_system()
            else:
                args[0].dynModel.init_system_with_value(args[0].initSysArr)
            args[0].fx = []
            for t in range(args[0].totalTimeStep):
                m = 0
                for p in range(args[0].dynModel.P):
                    m += observable(args[0],p)
                args[0].fx.append(m/args[0].dynModel.P)
                args[0].dynModel.update()
        return wrapped_func
    
    def distribution(observable):
        @functools.wraps(observable)
        def wrapped_func(*args):
            if not args[0].sync:
                args[0].dynModel.init_system()
            else:
                args[0].dynModel.init_system_with_value(args[0].initSysArr)
            for i in range(args[0].totalTimeStep):
                args[0].dynModel.update()
            args[0].fx = []
            args[0].binDist = np.float16(args[0].binDistTimes * 1/(args[0].dynModel.N-args[0].dynModel.C_mu))
            for p in range(args[0].dynModel.P):
                args[0].fx.append(observable(args[0],p))
            begin, end = min(args[0].fx), max(args[0].fx)
            args[0].stat = {'retrieved patterns':np.argmax(args[0].fx)}
            args[0].binEdges = [begin]
            while(True):
                args[0].binEdges.append(begin + args[0].binDist)
                begin += args[0].binDist
                if begin >= end:
                    args[0].binEdges.append(begin)
                    break
        return wrapped_func
    
    def distribution_q_matrix(self,simNum):
        listMat = self.disMat.flatten()
        begin, end = min(listMat), max(listMat)
        self.binEdges = [begin]
        self.binDist = self.binDistTimes * 1 /(simNum*(simNum-1))
        while(True):
            self.binEdges.append(begin + self.binDist)
            begin += self.binDist
            if begin >= end:
                self.binEdges.append(begin)
                break
        
        self.fig, ax = plt.subplots(figsize=(8,5),dpi=90)
        ax.hist(self.disMat.flatten(),self.binEdges,histtype='bar')
        ax.grid(True)
        plt.show()
        
    def distribution_interaction(self):
        listMat = self.dynModel.memMat.flatten()
        begin, end = np.min(listMat), np.max(listMat)
        self.binEdges = [begin]
        a = self.dynModel.a_mu
        N = self.dynModel.N
        self.binDist = self.binDistTimes * 1/(N*(1-a))
        while(True):
            self.binEdges.append(begin + self.binDist)
            begin += self.binDist
            if begin >= end:
                self.binEdges.append(begin)
                break
        print(len(self.binEdges))
        self.fig, self.ax = plt.subplots(figsize=(8,5),dpi=90)
        print("drawing...")
        _listMat = listMat[0,:]
        np.random.shuffle(_listMat)
        toHist = _listMat[:4000]
        print(toHist.shape)
        self.ax.hist(toHist,self.binEdges,histtype='bar')
        print("Done.")
        self.ax.grid(True)
        plt.show()
    
    def observable_matrix(recoder,simulator,shape,needPatternInfo=False):
        def _observable_matrix(observable):
            @functools.wraps(observable)
            def wrapped_func(*args,**kwargs):
                rows, cols = shape[0], shape[1]
                recoder.disMat = np.zeros((rows,cols))
                if not needPatternInfo:
                    for i in range(rows):
                        for j in range(cols):
                            recoder.disMat[i,j] = observable(simulator.simRes[i],
                                          simulator.simRes[j])
                else:
                    for i in range(rows):
                        for j in range(cols):
                            recoder.disMat[i,j] = observable(recoder,j,simulator.simRes[i])
            return wrapped_func
        return _observable_matrix
    
    def plot(self):
        self.fig, ax = plt.subplots(figsize=(8,5),dpi=80)
        if isinstance(self.fx,list):
            ax.plot(list(range(self.totalTimeStep)),self.fx,label=f"mattis overlap")
            
        if isinstance(self.fx,dict):
            for k in self.fx:
                ax.plot(list(range(self.totalTimeStep)),self.fx[k],label=r"$m_{%s}$"%k)
#        ax.legend(loc='upper right', shadow=True)
        plt.show()
        
    def draw_distance_matrix(self):
        self.fig, ax = plt.subplots(figsize=(8,5),dpi=80)
        plt.imshow(self.disMat,cmap=plt.cm.BuPu_r)
        cax = plt.axes([0.85, 0.1, 0.055, 0.8])
        plt.colorbar(cax=cax)
        plt.show()
        
    def draw_observable_matrix(self):
        self.fig, ax = plt.subplots(figsize=(8,5),dpi=80)
        plt.imshow(self.obMat,cmap=plt.cm.BuPu_r)
        cax = plt.axes([0.85, 0.1, 0.055, 0.8])
        plt.colorbar(cax=cax)
        plt.show()
        
    def hist(self):
        self.fig, ax = plt.subplots(figsize=(8,5),dpi=90)
        ax.hist(self.fx,self.binEdges,histtype='bar')
        ax.grid(True)
        plt.show()
        
    def bar(self):
        self.fig, self.ax = plt.subplots(figsize=(8,5),dpi=80)
        self.ax.bar(np.arange(self.dynModel.P),self.fx)
        self.ax.grid(True)
        plt.show()
    
    def mattis_overlap(self,idx):
        kesi = np.mat(self.dynModel.pat['kesi'][idx,:],dtype=np.float32)
        ret = np.inner(kesi,self.dynModel.sysArr.T)[0,0]
        return np.float32(ret / self.dynModel.C_mu)
    
    def mattis_overlap_(self,idx,arr=None):
        if arr is None:
            _arr = self.dynModel.sysArr.T
        else:
            _arr = arr.T
        _ita = np.mat(self.dynModel.pat['ita'][idx,:],dtype=np.float32)
        ita = (_ita-self.dynModel.a_mu) / (self.dynModel.a_mu*(1-self.dynModel.a_mu))
        ret = np.inner(ita.astype(np.float32),_arr)[0,0]
        return np.float32(ret / self.dynModel.N)
    
    def hamming_overlap(self,idx,arr=None):
        if arr is None:
            _arr = self.dynModel.sysArr.T
        else:
            _arr = arr.T
        ita = np.mat(self.dynModel.pat['ita'][idx,:],dtype=np.float32)
        assert(_arr.shape==ita.shape)
        ret = 2*(0.5 - np.abs(_arr - ita))
        _ = np.ones((1,self.dynModel.N),dtype=np.float32)
        ret = np.inner(ret.astype(np.float32), _ )[0,0]
        return np.float32(ret / self.dynModel.N)

    @staticmethod
    def EA_overlap(arr1,arr2):
        N = len(arr1)
        ret = 2*(0.5 - np.abs(arr1 - arr2))
        _ = np.ones((1,N), dtype=np.float32)
        ret = np.inner(ret.T, _)[0,0]
        return np.float32(ret / N)
    
    @property
    def activity(self):
        return np.sum(self.dynModel.sysArr)/self.dynModel.N
