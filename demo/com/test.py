"""
Similar to test_mult_learn, but this time as inverse regression.

Training z = x * y

x and y are noisy inputs.
"""

import numpy as np
import tenbilac
import os

import logging
logging.basicConfig(level=logging.INFO)


# We prepare some data to play with:

nrea = 5
ncas = 500
xs = np.random.uniform(-3, 6, ncas)
ys = np.random.uniform(-10, -5, ncas)
zs = xs * ys 
inputs = np.array([xs, ys])
inputs = np.tile(inputs, (nrea, 1, 1))
inputs += 0.01*np.random.randn(inputs.size).reshape(inputs.shape)
# This is 3D (rea, features=2, case)
targets = np.array([zs])
# This is 2D (feature=1, case)


# And play:

ten = tenbilac.com.Tenbilac("tenbilac.cfg")

ten.setup()

ten.train()




