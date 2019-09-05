# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:14:53 2019

@author: Jerry
"""

from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 0.1
avgDegree = 3
temperature = 0.00
totalSteps = 50
binDist = 100
numSim = 100
dm = DynamicModel(temperature,N,avgDegree,alpha)
re = Recoder(dm,totalSteps,binDist,False)
simulator = MonteCarloSimulator(dm,totalSteps,numSim)
matShape = (simulator.numSim,simulator.numSim)

simulator.simulation()

compute_q_matrix = Recoder.observable_matrix(re,simulator,matShape)(Recoder.EA_overlap)
print("computing q matrix...")
compute_q_matrix()
print("Done. Now drawing...")
re.draw_distance_matrix()
re.distribution_q_matrix(numSim)
