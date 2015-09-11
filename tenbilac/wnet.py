"""
A WNet is a custom network (in fact 2 networks in parallel), that also predicts a weight for each output parameter.
"""

import numpy as np
import scipy.optimize
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import layer
from . import net
from . import utils
from . import err
from . import act
from . import data

class WNet():
	"""
	A WNet contains 2 Nets.
	To a training, it looks like a normal Net but with twice the number of ouputs (as it returns outputs and weights).
	The appropriate error fuctions know how to combine these into a single metric.
	"""
	
	def __init__(self, ni, nhs, no=1, name=None, inames=None, onames=None, netokwargs={}, netwkwargs={}):
		"""
		:param ni: Number of input features
		:param nhs: Numbers of neurons in hidden layers
		:type nhs: tuple
		:param no: Number of ouput neurons
			
		:param name: if None, will be set automatically
		:type name: string
		
		:param inames: a list of names (strings) for the input nodes, to be used, e.g., in checkplots.
			These names have a purely decorative function, and are optional.
		:param onames: idem, for the ouptut nodes.
		
		:param netokwargs: dict of further kwargs for the constructor of the Net yielding the outputs
		:param netwkwargs: dict of further kwargs for the constructor of the Net yielding the weights
		
		
		"""
	
		self.ni = ni
		self.no = 2*no	# For every output we also have a weight. Assuming this is mandatory for the
						# way we return outputs, and so we'll stick to this and not make it more complicated.
						# However we do not rely on the fact the that two Nets are "similar"!
						# So it might be that the two nets have different numbers of params!
	
		logger.info("Building a WNet...")
		self.neto = net.Net(ni, nhs, no=no, name="neto", inames=inames, onames=onames, **netokwargs)
		self.netw = net.Net(ni, nhs, no=no, name="netw", inames=inames, onames=onames, **netwkwargs)
	
		self.name = name
		
		# We adapt the onames of the 2 networks:
		
		for i in range(self.neto.no):
			self.neto.onames[i] += "_o"
			self.netw.onames[i] += "_w"
			
	
	def __str__(self):
		"""
		A short string describing the network
		"""
		
		autotxt = "WNet" + str(self.neto)[7:]
			
		if self.name is None:
			return autotxt
		else:
			return "'{name}' {autotxt}".format(name=self.name, autotxt=autotxt)

	
	def report(self):
		"""
		Returns a text about the network parameters, useful for debugging.
		"""
		txt = ["#"*120, self.neto.report(), self.netw.report(), "#"*120]
		return "\n".join(txt)
		
	
	def save(self, filepath):
		"""
		Saves self into a pkl file
		"""
		utils.writepickle(self, filepath)		
	
	
	def nparams(self):
		"""
		Returns the number of parameters of the network
		"""
		return self.neto.nparams() + self.netw.nparams() 
		
	
	def get_params_ref(self):
		"""
		Get a single 1D numpy array containing references to all network weights and biases.
		Note that each time you call this, you loose the "connection" to the ref from any previous calls.
		
		Did not manage to get this working by reusing the equivalent functions of the two Net objects.
		So redoing from scratch.
		We put the "o" params first, then the "w" params.
		
		"""
		
		ref = np.empty(self.nparams())
		
		ind = 0
		
		# We start with the "o" params:
		for l in self.neto.layers[::-1]:
			
			ref[ind:ind+l.nn] = l.biases.flatten() # makes a copy
			ref[ind+l.nn:ind+l.nparams()] = l.weights.flatten() # makes a copy
			l.biases = ref[ind:ind+l.nn] # a view
			l.weights = ref[ind+l.nn:ind+l.nparams()].reshape(l.nn, l.ni) # a view
			ind += l.nparams()
		
		# And now the "w" params:
		assert ind == self.neto.nparams()
		
		for l in self.netw.layers[::-1]:
			
			ref[ind:ind+l.nn] = l.biases.flatten() # makes a copy
			ref[ind+l.nn:ind+l.nparams()] = l.weights.flatten() # makes a copy
			l.biases = ref[ind:ind+l.nn] # a view
			l.weights = ref[ind+l.nn:ind+l.nparams()].reshape(l.nn, l.ni) # a view
			ind += l.nparams()
	
		assert ind == self.nparams()
		return ref


	def get_paramlabels(self):
		"""
		Returns a list with labels describing the params.
		
		"""
		paramlabels = []
		paramlabels.extend(["neto_" + l for l in self.neto.get_paramlabels()])
		paramlabels.extend(["netw_" + l for l in self.netw.get_paramlabels()])
		return paramlabels
		

	def addnoise(self, **kwargs):
		"""
		Adds random noise to all parameters.
		"""
	
		self.neto.addnoise(**kwargs)
		self.netw.addnoise(**kwargs)
			
			
	def setini(self):
		"""
		Adjusts the network parameters so to approximatively get:
		- identity for neto
		- zero for netw (i.e., always return 0.0 -> a weight of e0 = 1, no matter what the input is).
		
		"""

		self.neto.setidentity()
		for l in self.netw.layers:
			l.zero()
			

	def run(self, inputs):
		"""
		Similar to Net.run().
		We compute outputs for both Nets, and combine these into a single array, mimicing a single Net with twice the number of outputs.
		So for the outputs, along the second dimension (nodes), we first have the normal outputs, and then the associated weights.
		"""
		
		if inputs.ndim != 3:
			raise ValueError("Sorry, for WNet I want 3D inputs only.")
		
		os = self.neto.run(inputs)
		ws = self.netw.run(inputs)
		
		assert os.shape == (inputs.shape[0], self.neto.no, inputs.shape[2])
		assert ws.shape == (inputs.shape[0], self.netw.no, inputs.shape[2])
		
		# And we stack the two parts 
		outputs = np.concatenate((os, ws), axis=1)
		
		assert outputs.shape == (inputs.shape[0], self.neto.no + self.netw.no, inputs.shape[2])
		
		return outputs
		
			
	
	def predict(self, inputs):
		"""
		We compute the outputs from the inputs using self.run, but here we do take care of the potential mask.
		
		This is never used during the training phase.
		
		:param inputs: a (potentially masked) 3D array
		
		:returns: a 3D array, appropriatedly masked
		
		"""
		
		# To be adapted for WNet...
		"""
		logger.info("Predicting with input = {intype} of shape {inshape}".format(
			intype=str(type(inputs)), inshape=str(inputs.shape)))

		if inputs.ndim != 3:
			raise ValueError("Sorry, I only accept 3D input")

		(inputs, outputsmask) = data.demask(inputs, no=self.no)
		
		# We can simply run the network with the unmasked inputs:
		
		logger.info("Running the actual predictions...")
		outputs = self.run(inputs)
		
		# And now mask these outputs, if required:
		
		if outputsmask is not None:
			outputs = np.ma.array(outputs, mask=outputsmask)
		
		return outputs
		"""
