# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:01:49 2019

@author: Jerry
"""

from GRN import DynamicModel,Recoder

N = 2000
alpha = 0.03
avgDegree = 1000
temperature = 0.00
totalSteps = 90
binDist = 20
dm = DynamicModel(temperature,N,avgDegree,alpha)
re = Recoder(dm,totalSteps,binDist)

multi_overlap_observation = Recoder.multi_observation(6)(Recoder.hamming_overlap)
multi_overlap_observation(re)
re.plot()

mean_overlap_observation = Recoder.observation_mean(Recoder.hamming_overlap)
mean_overlap_observation(re)
re.plot()

dist = Recoder.distribution(Recoder.hamming_overlap)
dist(re)
re.hist()