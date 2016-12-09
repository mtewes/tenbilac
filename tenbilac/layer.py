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
	def __init__(self, ni, nn, actfct=act.tanh, name="None", mode="sum"):
		
		"""
		:param ni: Number of inputs
		:param nn: Number of neurons
		:param actfct: Activation function
		:param mode: "sum" or "mult". If sum, the neurons perform the usual linear combinations.
			If mult, they work as "Product Units" (Durbin Rumelhart 1989).
		"""

		self.ni = abs(ni)
		self.nn = abs(nn) # We keep those positive, even if negative might have indicated "mult" earlier in the code.
		self.actfct = actfct
		self.name = name
		
		self.mode = mode
		if not self.mode in ("sum", "mult"):
			raise RuntimeError("Unknown layer mode")

		self.weights = np.zeros((self.nn, self.ni)) # first index is neuron, second is input
		self.biases = np.zeros(self.nn) # each neuron has its bias
			
		
	def __str__(self):
		return "Layer '{self.name}', mode {self.mode}, ni {self.ni}, nn {self.nn}, actfct {self.actfct.__name__}".format(self=self)
	
	def modecode(self):
		"""
		Just a little "*" to label unconventional layers in logs etc
		"""
		if self.mode == "sum":
			return ""
		else:
			return "*"
	
	def addnoise(self, wscale=0.1, bscale=0.1, multwscale=0.1, multbscale=0.1):
		"""
		Adds some noise to weights and biases
		"""
		if self.mode == "sum":
			self.weights += wscale * np.random.randn(self.weights.size).reshape(self.weights.shape)
			self.biases += bscale * np.random.randn(self.biases.size)
		elif self.mode == "mult":
			self.weights += multwscale * np.random.randn(self.weights.size).reshape(self.weights.shape)
			self.biases += multbscale * np.random.randn(self.biases.size)
		else:
			raise RuntimeError("Unknown mode")
		
	
	def setzero(self):
		"""
		Sets all weights and biases to zero
		"""
		logger.info("Setting {self.mode}-layer '{self.name}' parameters to zero...".format(self=self))
		#self.weights = np.zeros(self.weights.shape) # No, this does not work! It kills the get_params reference used to set up the Training *before* noise is added.
		#self.biases = np.zeros(self.biases.shape) # Leave this commented to prevent later attempts to change it.
		self.weights *= 0.0
		self.biases *= 0.0
	

	def setidentity(self, onlyn=None):
		"""
		Sets the weights and biases so that this layer transports all its first nn inputs to the nn outputs.
		This is meant to give a promising initial condition for training a calibration-like task. 
		
		:param onlyn: If set, limits the number of neurons to be set. By default, all are set.
		:type onlyn: int
		"""
		self.setzero()
		
		npossible = min(self.ni, self.nn) # Can't transport more inputs
		if onlyn is None:
			ngofor = npossible
		else:
			if onlyn > npossible:
				logger.warning("Cannot setidentity for more neurons (onlyn={}) than possible ({})".format(onlyn, npossible))
			ngofor = min(onlyn, npossible)
		if ngofor < 0:
			raise RuntimeError("Cannot set identity, something wrong with number of neurons.")
		
		for n in range(ngofor):
			self.weights[n, n] = 1.0
		logger.info("Setting identity for {}: it now transports {} inputs".format(str(self), ngofor))
		
	
	
	def report(self):
		"""
		Returns a text about the weights and biases, useful for debugging
		"""
		
		txt = []
		txt.append(str(self) + ":")
		for inn in range(self.nn):
			if self.mode == "sum":
				txt.append("    output {inn} = {act} ( input * {weights} + {bias} )".format(
					inn=inn, act=self.actfct.__name__, weights=self.weights[inn,:], bias=self.biases[inn]))
			elif self.mode == "mult":
				txt.append("    output {inn} = {act} ( sign * prod (input ** {weights}) +(???) {bias} )".format(
					inn=inn, act=self.actfct.__name__, weights=self.weights[inn,:], bias=self.biases[inn]))
		return "\n".join(txt)
	
	
	def nparams(self):
		"""
		Retruns the number of parameters of this layer
		"""
		return self.nn * (self.ni + 1)
	
	
	def run(self, inputs):
		"""
		Computes output from input, as "numpyly" as possible, using only np.dot (note that np.ma.dot does not
		work with 3D masked arrays, and np.tensordot seems not available for masked arrays.
		This means that np.dot determines the order of indices, as following.
		
		If inputs is 1D,
			the input is just an array of features for a single case
			the output is a 1D array with the output of each neuron
			
		If inputs is 2D,
			input indices: (feature, case)
			output indices: (neuron, case) -> so this can be fed into the next layer...
		
		If inputs is 3D, 
			input indices: (realization, feature, case)
			output indices: (realization, neuron, case)
			
		"""
		if self.mode == "sum":
			if inputs.ndim == 1:
				return self.actfct(np.dot(self.weights, inputs) + self.biases)
		
			elif inputs.ndim == 2:
				assert inputs.shape[0] == self.ni		
				return self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1)))
		
			elif inputs.ndim == 3:
				assert inputs.shape[1] == self.ni
			
				# Doing this:
				# self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1, 1)))
				# ... gives ouput indices (neuron, realization, case)
				# We need to change the order of those indices:
				
				return np.rollaxis(self.actfct(np.dot(self.weights, inputs) + self.biases.reshape((self.nn, 1, 1))), 1)
			
				# Note that np.ma.dot does not work for 3D arrays!
				# We do not care about masks at all here, just compute assuming nothing is masked.
			
			else:
				raise RuntimeError("Input shape error")

# The initial implementation of mult, works for positive inputs:
#		elif self.mode == "mult":
#			
#			if inputs.ndim == 1:
#				return self.actfct(np.prod(np.power(inputs, self.weights), axis=1) + self.biases) # Phew, that was easy, nice!
#		
#			elif inputs.ndim == 2:
#				assert inputs.shape[0] == self.ni			
#				return self.actfct(np.transpose(np.prod([np.power(inputs[:,i], self.weights) for i in range(inputs.shape[1])], axis=2) + self.biases))
#				
#			elif inputs.ndim == 3:
#				assert inputs.shape[1] == self.ni
#				return np.swapaxes(self.actfct(np.prod([[np.power(inputs[j,:,i], self.weights) for i in range(inputs.shape[2])] for j in range(inputs.shape[0])], axis=3) + self.biases), 1, 2)
#				
#			else:
#				raise RuntimeError("Input shape error")
			
# new variants for negative and positive inputs
		elif self.mode == "mult":
			
			signlim = 0.5
			self.biases *= 0.0 # For now I kill those: biases have no effect. Remains to be explored how to use them best.
			
			if inputs.ndim == 1:		
				# The test used as exponent has to be False if we want to ignore the sign, True to use it
				signs = np.prod(np.power(np.sign(inputs), np.fabs(self.weights) > signlim), axis=1)
				return signs * self.actfct(np.prod(np.power(np.fabs(inputs), self.weights), axis=1) + self.biases) # Phew, that was easy, nice!
		
			elif inputs.ndim == 2:
				assert inputs.shape[0] == self.ni
				signs = np.transpose(np.prod([np.power(np.sign(inputs[:,i]), np.fabs(self.weights) > signlim) for i in range(inputs.shape[1])], axis=2))	
				return signs * self.actfct(np.transpose(np.prod([np.power(np.fabs(inputs[:,i]), self.weights) for i in range(inputs.shape[1])], axis=2) + self.biases))
				
			elif inputs.ndim == 3:
				assert inputs.shape[1] == self.ni
				signs = np.swapaxes(np.prod([[np.power(np.sign(inputs[j,:,i]), np.fabs(self.weights) > signlim) for i in range(inputs.shape[2])] for j in range(inputs.shape[0])], axis=3), 1, 2)
				return signs * np.swapaxes(self.actfct(np.prod([[np.power(np.fabs(inputs[j,:,i]), self.weights) for i in range(inputs.shape[2])] for j in range(inputs.shape[0])], axis=3) + self.biases), 1, 2)
				
			else:
				raise RuntimeError("Input shape error")
		
		
		
		
		
	

