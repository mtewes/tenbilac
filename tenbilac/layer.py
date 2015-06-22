
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
		Computes output from input
		
		If input is 1D, it's just a single galaxy
		If input is 2D, first index is feature, second index is galaxy 
		"""
		
		if len(input.shape) == 1:
			return self.actfct(np.dot(self.weights, input) + self.biases)
		
		elif len(input.shape) == 2:
			assert input.shape[0] == self.ni		
			return self.actfct(np.dot(self.weights, input) + self.biases.reshape((self.nn, 1)))
		
		else:
			raise RuntimeError("Input shape error")

		
		
		
		
	

