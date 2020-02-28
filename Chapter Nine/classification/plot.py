#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-28 11:55:42
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-28 11:55:52
'''


import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1, 1, 50)
y = 1.0/(1.0+np.exp(-5*x))
plt.figure()
plt.plot(x, y)
plt.show()