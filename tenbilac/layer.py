"""
A layer holds the parameters (weights and biases) to compute the ouput of each of its neurons based on the input of the previous layer.
This means that the "input" of a neural network is *not* such a Layer object.
The "first" real Layer in a network is in fact the first hidden layer.

"""


import numpy as np

import logging
logger = logging.getLogger(__name__)

from . import act


		
class Layer():	
	def __init__(self, ni, nn, actfct=act.Sig, name="None"):
		
		"""
		:param ni: Number of inputs
		:param nn: Number of neurons
		:param actfct: Activation function
		"""

		self.ni = ni
		self.nn = nn
		self.actfct = actfct
		self.name = name
		
		self.weights = np.zeros((self.nn, self.ni)) # first index is neuron, second is input
		self.biases = np.zeros(self.nn) # each neuron has its bias
		
	
	def addnoise(self, wscale=0.1, bscale=0.1):
		"""
		Adds some noise to weights and biases
		"""
		
		self.weights += wscale * np.random.randn(self.weights.size).reshape(self.weights.shape)
		self.biases += bscale * np.random.randn(self.biases.size)
	
	
	def report(self):
		"""
		Returns a text about the weights and biases, useful for debugging
		"""
		
		txt = []
		txt.append("Layer '{name}':".format(name=self.name))
		for inn in range(self.nn):
			txt.append("    output {inn} = {act} ( input * {weights} + {bias} )".format(
				inn=inn, act=self.actfct.__class__.__name__, weights=self.weights[inn,:], bias=self.biases[inn]))
		return "\n".join(txt)
	
	
	def nparams(self):
		"""
		Retruns the number of parameters of this layer
		"""
		return self.nn * (self.ni + 1)
	
	
	def run(self, input):
		"""
		Computes output from input, as "numpyly" as possible, using only np.dot (given that
		np.tensordot seems not available for masked arrays, but we do want this to run on masked
		arrays when dealing with multiple realizations).
		This means that np.dot determines the order of indices, as following.
		
		If input is 1D,
			the input is just an array of features for a single galaxy
			the output is a 1D array with the output of each neuron
			
		If input is 2D,
			input indices: (feature, galaxy)
			output indices: (neuron, galaxy) -> so this can be fed into the next layer...
		
		If input is 3D, 
			input indices: (realization, feature, galaxy)
			output indices: (realization, neuron, galaxy)
			
		"""
		
		if input.ndim == 1:
			return self.actfct(np.dot(self.weights, input) + self.biases)
		
		elif input.ndim == 2:
			assert input.shape[0] == self.ni		
			return self.actfct(np.dot(self.weights, input) + self.biases.reshape((self.nn, 1)))
		
		elif input.ndim == 3:
			assert input.shape[1] == self.ni
			
			# Doing this:
			# self.actfct(np.dot(self.weights, input) + self.biases.reshape((self.nn, 1, 1)))
			# ... gives ouput indices (neuron, realization, galaxy)
			# We need to change the order of those indices:
				
			return np.rollaxis(self.actfct(np.dot(self.weights, input) + self.biases.reshape((self.nn, 1, 1))), 1)
		
		else:
			raise RuntimeError("Input shape error")

		
		
		
		
	

