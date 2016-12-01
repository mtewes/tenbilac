"""
Regularisation functions
These work on the weights.

The typical call is fct(weights)
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


def l1(weights):
	"""
	The L1 norm is simply the sum of the absolute values. 
	"""
	return np.sum(np.abs(weights))

def l2(weights):
	"""
	The L2 norm is the sqrt of the sum of the squares. 
	"""
	return np.sqrt(np.sum(weights * weights))