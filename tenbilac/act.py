"""
Activation functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


class Sig:
	"""
	Sigmoid	
	"""

	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	
	
class Id:
	"""
	Plain identity
	"""
	
	def __call__(self, x):
		return x
