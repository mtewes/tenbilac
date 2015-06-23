"""
Error functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


class MSE:
	"""
	Standard MSE, simply treats realizations as if they were independent galaxies	
	"""



class MSB:
	"""
	Mean square bias
	
	:param inputs: 3D array (realization, feature, galaxy)
	:param targets: 2D array (feature, galaxy)
	
	"""

	def __call__(self, net, inputs, targets):
		return np.mean(np.square(np.mean(net.run(inputs), axis=0) - targets))
		
	


