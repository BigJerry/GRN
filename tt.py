# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:16:07 2019

@author: Jerry
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8,5),dpi=80)
ax.imshow(np.random.random((3, 7)), cmap=plt.cm.BuPu_r)

#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.055, 0.8])
fig.colorbar(cax=cax)
#plt.show()