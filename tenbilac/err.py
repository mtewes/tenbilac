"""
Error functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)



	
def msb(predictions, targets):
	"""
	Mean square bias
	
	:param predictions: 3D array (realization, neuron, galaxy)
	:param targets: 2D array (neuron, galaxy)
	
	"""
	if predictions.ndim == 3:
		return np.mean(np.square(np.mean(predictions, axis=0) - targets))
	else:
		raise ValueError("Wrong pred shape")
	
	


def mse(predictions, targets):
	"""
	Standard MSE (mean square error), simply treats multiple realizations as if they were independent galaxies	
	
	:param predictions: 2D array (neuron, galaxy) or 3D array (realization, neuron, galaxy)
	:param targets: 2D array (neuron, galaxy)

	"""
	
	# This same code works for 2D or 3D predictions:
	return np.mean(np.square(predictions - targets))
	
	#return np.mean(np.square(net.run(inputs[0]) - targets))


