"""
Create some data to play with
"""

import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


nrea = 5
ncas = 500
xs = np.random.uniform(-3, 6, ncas)
ys = np.random.uniform(-10, -5, ncas)
zs = xs * ys 
inputs = np.array([xs, ys])
inputs = np.tile(inputs, (nrea, 1, 1))
inputs += 0.01*np.random.randn(inputs.size).reshape(inputs.shape)
targets = np.array([zs])


logger.info("Inputs have shape {}, targets have shape {}".format(inputs.shape, targets.shape))
tenbilac.utils.writepickle((inputs, targets), "data.pkl")




