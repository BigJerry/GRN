# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:26:13 2019

@author: Jerry
"""

from GRN import DynamicModel,Recoder,MonteCarloSimulator

N = 2000
alpha = 0.003
avgDegree = 3
temperature = 0.00
totalSteps = 50
binDist = 20
numSim = 20
dm = DynamicModel(temperature,N,avgDegree,alpha)
re = Recoder(dm,totalSteps,binDist,False)
simulator = MonteCarloSimulator(dm,totalSteps,numSim)
matShape = (simulator.numSim, re.dynModel.P)

simulator.simulation()

compute_observable_matrix = Recoder.observable_matrix(re,simulator,matShape,True)(Recoder.hamming_overlap)
print("computing observable matrix...")
compute_observable_matrix()
print("Done. Now drawing...")
re.draw_distance_matrix()