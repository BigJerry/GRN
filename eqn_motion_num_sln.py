# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:23:50 2019

@author: Jerry
"""

from GRN import PatternGenerator
import numpy as np
import functools
from matplotlib import pyplot as plt

def braket(avgNum,pg):
    def _braket(_func):
        @functools.wraps(_func)
        def wrapped_func(*args):
            if _func.__name__ is 'eqn_m_mu' or 'eqn_m_mu_1':
                summation = np.zeros((pg.P,1),dtype=np.float32)
            if _func.__name__ is 'eqn_a':
                summation = 0
                
            if _func.__name__ is 'm_mu':
                summ_m, summ_a = np.zeros((pg.P,1),dtype=np.float32),0
                for n in range(avgNum):
                    pg.generate()
                    argFromHere = pg.pat['ita']
                    s_m, s_a = _func(*args, argFromHere)
                    summ_m += s_m
                    summ_a += s_a
                return summ_m / avgNum, summ_a / avgNum
            
            pg.generate()
            for i in range(avgNum):
                
                argFromHere = [pg.pat['ita'][:,i].reshape((pg.P,1)), \
                               pg.pat['kesi'][:,i].reshape((pg.P,1))]

                summation += _func(*args, argFromHere)
            if _func.__name__ is 'eqn_m_mu':    
                return summation
            if _func.__name__ is 'eqn_a' or 'eqn_m_mu_1':
                return summation /avgNum
        return wrapped_func
    return _braket

class NumericalSolver(object):
    def __init__(self,args):
        self.Beta = args['beta']
        self.avgNum = args['avgNum']
        self.itNum = args['itNum']
        self.pg = args['patGen']
        self.gamma = args['gamma']
        
    @property
    def N(self):
        return self.pg.N
    @property
    def P(self):
        return self.pg.P
    @property
    def C_mu(self):
        return self.pg.C_mu        
        
    def m_mu(self,state,fromBraket=None):
        if fromBraket is not None:
            state = fromBraket[0,:].T.reshape((self.N,1))
            a = np.sum(state) / self.N
            m = np.matmul(fromBraket,state) / self.C_mu
            return m,a
        else:
            assert(state.T.shape[1]==1)
            return np.matmul(self.ita, state.T) / self.C_mu                     #notice here we didn't differentiate varied C_mu
                                                                                #even if it can be applicable

    def eqn_m_mu(self, mPrimeVec, aPrime, fromBraket):                          #arguments of equation of motion come from two
                                                                                #sources, one is function 'solve', another is 
                                                                                #wrapper 'braket'.
        assert(mPrimeVec.shape[1]==1)
        field = np.matmul(fromBraket[1].T, mPrimeVec-aPrime)
        ret = 1 / ( 2 * self.C_mu) * fromBraket[0] * (1 + np.tanh((self.Beta / 2) * field))
        assert(ret.shape[1]==1)
        return ret
    
    def eqn_m_mu_1(self, mPrimeVec, aPrime, fromBraket):
        assert(mPrimeVec.shape[1]==1)
        #prepare for 'field' matrix
        k = fromBraket[1].T
        f = np.array(k,dtype=np.float32)
        for n in range(self.pg.P-1):
            f = np.append(f, np.array(k,dtype=np.float32), 0)
        for n in range(self.pg.P):
            f[n,n] = 1.0
        field = np.matmul(f, mPrimeVec - aPrime)
        ret = 1/2* (1 + np.tanh(self.Beta/2 * field))
        assert(ret.shape[1]==1)
        return ret
    
    def eqn_a(self,mPrimeVec,aPrime,fromBraket):
        
        field = np.matmul(fromBraket[1].T,mPrimeVec-aPrime)
        ret = 1/2 * (1 + np.tanh(self.Beta/2*field[0,0]))

        return ret       
    
    def solve(self):
        #initializing initial patterns and states        
        self.ita = self.pg.pat['ita']                                           #save initial pattern 'ita' as member variable
        self.kesi = self.pg.pat['kesi']                                         #save initial pattern 'kesi' as member variable
        
        initState = np.zeros((1,self.N),dtype=np.float32)                       #initialize neuron states.
                                                                                #together with above initial patterns,
                                                                                #the initial overlap order parameters
                                                                                #will be determined.
        gen_mPrimeVec = braket(self.avgNum,self.pg)(self.m_mu)
        mPrimeVec,aPrime = gen_mPrimeVec(initState)
        
#        mPrimeVec = np.array([[1.0],[0.0],[0.0],[0.0],[0.0]],dtype=np.float32)
#        aPrime = 0.0015
        print("initial overlaps vector configuration: \n",mPrimeVec)
        print("initial activity level is: ",aPrime)
        
        self.records = {'activity':[],'overlap':None}
        self.records['overlap'] = mPrimeVec
        self.records['activity'].append(aPrime)
        #wrapping equation of motion using 'braket'
        m_mu_ = braket(self.avgNum,self.pg)(self.eqn_m_mu)
        a_ = braket(self.avgNum,self.pg)(self.eqn_a)
        #iterating equation of motion
        
        for n in range(self.itNum-1):
            mPrimeVec_, aPrime_ = m_mu_(mPrimeVec,aPrime), a_(mPrimeVec,aPrime)
            assert(mPrimeVec_.shape[1]==mPrimeVec.shape[1]==1)
            self.records['overlap'] = np.hstack((self.records['overlap'],mPrimeVec_))
            self.records['activity'].append(aPrime)
            mPrimeVec , aPrime = mPrimeVec_ , aPrime_
            
            print("iteration No.%d finished, next run..."%n)

if __name__ == '__main__':
    fname = r'E:\temptemptemp\npy\hehe.npy'
    #model-related parameters
    delta = 1/3
    gamma = 0.7
    alpha = 1
    
    beta = 300
    dims = 25000
    avgDegree = 12
    #simulation-related parameters
    avgNum = dims
    itNum = 20                                                                 #total time steps to run
    #deciding which regime we're now in
    reg_p = 'limited'
    reg_c = 'sparse'
    #recording configuration we got 
    cfg = {'delta':delta,
           'gamma':gamma,
           'alpha':alpha,
           'beta':beta,
           'dims':dims,
           'avgDegree':avgDegree,
           'avgNum':avgNum,
           'itNum':itNum,
           'reg_p':reg_p,
           'reg_c':reg_c}
    
    pg = PatternGenerator(dims,avgDegree,alpha,gamma=gamma,delta=delta)
    pg.regime_P(reg_p,5)
    pg.regime_C(reg_c)
    pg.generate()
    
    solverArgs = {'beta':beta,
                  'patGen':pg,
                  'avgNum':avgNum,
                  'itNum':itNum,
                  'gamma':gamma}
    
    ns = NumericalSolver(solverArgs)
    ns.solve()
    print(cfg)
    x = np.arange(itNum)
    for n in range(pg.P):
        y = ns.records['overlap'][n,:]
        plt.plot(x, y)
    y = ns.records['activity']
    plt.plot(x, y)
    