"""
This is Tenbilac!
"""

import numpy as np
import scipy.optimize
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import layer
from . import utils
from . import err
from . import act
from . import data

class Tenbilac():
	"""
	Object representing a network made out of one or several hidden layers.
	"""
	
	def __init__(self, ni, nhs, no=1, onlyid=False, actfctname="tanh", name=None, inames=None, onames=None):
		"""
		:param ni: Number of input features
		:param nhs: Numbers of neurons in hidden layers
		:type nhs: tuple
		:param no: Number of ouput neurons
		:param onlyid: Set this to true if you want identity activation functions on all layers
			(useful for debugging).
			
			
		:param name: if None, will be set automatically
		:type name: string
		
		:param inames: a list of names (strings) for the input nodes, to be used, e.g., in checkplots.
			These names have a purely decorative function, and are optional.
		:param onames: idem, for the ouptut nodes.
		
		
		"""
	
		self.ni = ni
		self.nhs = nhs
		self.no = no
		self.name = name
		
		# We take care of the inames and onames:
		if inames is None:
			self.inames = ["i_"+str(i) for i in range(self.ni)]
		else:
			self.inames = inames
			if len(self.inames) != self.ni:
				raise RuntimeError("Your number of inames is wrong")
		if onames is None:
			self.onames = ["o_"+str(i) for i in range(self.no)]
		else:
			self.onames = onames
			if len(self.onames) != self.no:
				raise RuntimeError("Your number of onames is wrong")
		
		iniarch = np.array([self.ni]+self.nhs+[self.no]) # Note that we do not save this. Layers might evolve dynamically in future!

		actfct = eval("act.{0}".format(actfctname)) # We turn the string actfct option into an actual function
		
		self.layers = [] # We build a list containing only the hidden layers and the output layer
		for (i, nh) in enumerate(self.nhs):
				self.layers.append(layer.Layer(ni=iniarch[i], nn=nh, actfct=actfct, name="h"+str(i)))
		# Adding the output layer:
		self.layers.append(layer.Layer(ni=self.nhs[-1], nn=no, actfct=act.iden, name="o"))
		
		if onlyid: # Then all layers get the Id activation function:
			for l in self.layers:
				l.actfct = act.iden
				
		logger.info("Built " + str(self))


	
	def __str__(self):
		"""
		A short string describing the network
		"""
		#return "Tenbilac with architecture {self.arch} and {nparams} params".format(self=self, nparams=self.nparams())
		#archtxt = str(self.ni) + "|" + "|".join(["{n}/{actfct}".format(n=l.nn, actfct=l.actfct.__name__) for l in self.layers])
		archtxt = str(self.ni) + "|" + "|".join(["{n}/{actfct}".format(n=l.nn, actfct=l.actfct.__name__) for l in self.layers])
		#autotxt = "[{archtxt}]({nparams})".format(archtxt=archtxt, nparams=self.nparams())
		autotxt = "[{archtxt}={nparams}]".format(archtxt=archtxt, nparams=self.nparams())
		
		if self.name is None:
			return autotxt
		else:
			return "'name' {autotxt}".format(name=name, autotxt=autotxt)


	
	def report(self):
		"""
		Returns a text about the network parameters, useful for debugging.
		"""
		txt = ["="*120, str(self)]
		for l in self.layers:
			txt.append(l.report())
		txt.append("="*120)
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
		return sum([l.nparams() for l in self.layers])
		
	
	def get_params_ref(self, schema=2):
		"""
		Get a single 1D numpy array containing references to all network weights and biases.
		Note that each time you call this, you loose the "connection" to the ref from any previous calls.
		
		:param schema: different ways to arrange the weights and biases in the output array.
		
		"""
		
		ref = np.empty(self.nparams())
		ind = 0
		
		if schema == 1: # First layer first, weights and biases.
		
			for l in self.layers:
				ref[ind:ind+(l.nn*l.ni)] = l.weights.flatten() # makes a copy
				ref[ind+(l.nn*l.ni):ind+l.nparams()] = l.biases.flatten() # makes a copy
				l.weights = ref[ind:ind+(l.nn*l.ni)].reshape(l.nn, l.ni) # a view
				l.biases = ref[ind+(l.nn*l.ni):ind+l.nparams()] # a view
				ind += l.nparams()
		
		elif schema == 2: # Starting at the end, biases before weights
		
			for l in self.layers[::-1]:
			
				ref[ind:ind+l.nn] = l.biases.flatten() # makes a copy
				ref[ind+l.nn:ind+l.nparams()] = l.weights.flatten() # makes a copy
				l.biases = ref[ind:ind+l.nn] # a view
				l.weights = ref[ind+l.nn:ind+l.nparams()].reshape(l.nn, l.ni) # a view
				ind += l.nparams()
			
		else:
			raise ValueError("Unknown schema")
			
		
		# Note that such tricks do not work, as indexing by indices creates copies:
		#indices = np.arange(self.nparams())
		#np.random.shuffle(indices)
		#return ref[indices]

		assert ref.size == self.nparams()
		return ref



	def get_paramlabels(self, schema=2):
		"""
		Returns a list with labels describing the params. This is for humans and plots.
		Note that plots might expect these labels to have particular formats.
		"""
			
		paramlabels=[]
		ind = 0
		
		if schema == 2:
			for l in self.layers[::-1]:
				
				paramlabels.extend(l.nn*["layer-{l.name}_bias".format(l=l)])
				paramlabels.extend(l.nn*l.ni*["layer-{l.name}_weight".format(l=l)])
		
		assert len(paramlabels) == self.nparams()
		
		return paramlabels
		


	def addnoise(self, **kwargs):
		"""
		Adds random noise to all parameters.
		"""
		
		logger.info("Adding noise to network parameters...")
		
		for l in self.layers:
			l.addnoise(**kwargs)
			
			
	def setidentity(self):
		"""
		Adjusts the network parameters so to approximatively get an identity relation
		between the ith output and the ith input (for each i in the outputs).
		
		This should be a good starting position for "calibration" tasks. Example: first
		input feature is observed galaxy ellipticity g11, and first output is true g1.
		"""

		for l in self.layers:
			l.zero() # Sets everything to zero
			if l.nn < self.no or self.ni < self.no:
				raise RuntimeError("Network is too small for setting identity!")
			
		for io in range(self.no):
			for l in self.layers:
				l.weights[io, io] = 1.0 # Now we set selected weights to 1.0 (leaving biases at 0.0)
			
		logger.info("Set identity weights")
			

	def run(self, inputs):
		"""
		Propagates input through the network "as fast as possible".
		This works for 1D, 2D, and 3D inputs, see layer.run().
		Note that this forward-running does not care about the fact that some of the inputs might be masked!
		Use predict() if you have masked arrays.
		"""
		
		output = inputs
		for l in self.layers:
			output = l.run(output)
		return output
	
			
	
	def predict(self, inputs):
		"""
		We compute the outputs from the inputs using self.run, but here we do take care of the potential mask.
		
		This is never used during the training phase.
		
		:param inputs: a (potentially masked) 3D array
		
		:returns: a 3D array, appropriatedly masked
		
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
		
