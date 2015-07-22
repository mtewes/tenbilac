"""
Error functions
These should work both on unmasked and on masked arrays "as expected".
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)



	
def msb(predictions, targets):
	"""
	Mean square bias
	
	:param predictions: 3D array (realization, neuron, galaxy), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, galaxy)
	
	"""
	
	if predictions.ndim == 3:
	
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, galaxy)
		return np.mean(np.square(biases))
	
	else:
		raise ValueError("Wrong pred shape")


	
def msrb(predictions, targets, rawterms=False):
	"""
	Mean square relative bias
	
	:param predictions: 3D array (realization, neuron, galaxy), should be appropiratedly masked (thus not directly the output of the net)
	:param targets: 2D array (neuron, galaxy)
	
	:param rawterms: if True, returns the "RB" of "MSRB" as (potentially masked) 2D array.
	
	"""
	
	if predictions.ndim == 3:
	
		#assert type(predictions) == np.ma.MaskedArray
		# Just as a test that this was not forgotten, no real need here.
		# No need, for this, for sure it also works without masks...
		
		biases = np.mean(predictions, axis=0) - targets # This is 2D, (label, galaxy)
		stds = np.std(predictions, axis=0) # Same shape
		
		if type(predictions) == np.ma.MaskedArray:
			reacounts = predictions.shape[0] - np.sum(predictions.mask, axis=0) + 0.0 # Number of realizations, 0.0 makes this floats # This is 2D (label, galaxy)
		elif type(predictions) == np.ndarray:
			reacounts = predictions.shape[0] + 0.0
		else:
			raise RuntimeError("Type error in predictions.")
		
		errsonbiases = stds / np.sqrt(reacounts) # 2D
		relativebiases = biases / errsonbiases # 2D
		
		if rawterms:
			#return (biases, errsonbiases)
			return relativebiases
		else:
			return np.mean(np.square(relativebiases))
		
	
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



def msre(predictions, targets):
	"""
	Weighted MSE

	"""

	if predictions.ndim == 3:
		
		stds = np.std(predictions, axis=0) # 2D, (label, galaxy)

		return np.mean(np.square((predictions - targets) / stds))
	
	else:
		raise ValueError("Wrong pred shape")

	
