"""
A MultNet is a variant of a Net, with a first hidden layer performing multiplications.
This first layer needs special initialization, and is not optimized at the same pace as the second part.


Things to keep in mind:
 - in case you call Net.setidentity, you WANT to call MultNet.multini again to set the initial params of the first mult layer.
 - you probably don't want to add noise to the mult layer if you're not optimizing it. One way of doing this is to use multwscale=0.0, multbscale=0.0.

"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

from . import net
from . import layer
from . import utils
from . import act
from . import data

class MultNet(net.Net):


	def __init__(self, ni, nhs, mwlist=None, **kwargs):
		"""
		:param nhs: specify only the sum-hidden-layers, the mult-layer is inserted automatically!
		"""
		
		# MultNet specific tests:
		for n in nhs:
			if n < 0:
				raise RuntimeError("Would be fine for me, but is probably a mistake.")
	
		self.mwlist = mwlist
		nmult = ni + len(mwlist)
		net.Net.__init__(self, ni, [-nmult]+nhs, **kwargs)
	
		# Yes, it makes sense to directly set the weights of the first layer, as mwlist is given at this stage.
		self.multini()
		

	
	def multini(self):
		"""
		Initialize the mult layer in a very custom way.
		The first ni neurons are set to simply transport.
		The settings for the other ones are controlled by mwlist.
		
		:param mwlist: list of tuples containing the weights for each neuron of the mult-layer in addition
			to the "initial-identity" neurons.
			Example: mwlist=[(1, 1)] adds a single neuron that multiplies the first two inputs.
		
		"""
		
		logger.info("Initializing mult-layer with mwlist={}...".format(self.mwlist))
		
		if self.mwlist is None:
			raise RuntimeError("Weird: multini with mwlist None")
		
		self.layers[0].setzero()
		self.layers[0].setidentity(onlyn=self.ni)
		
		# And now the custom part, for the neurons which come after self.ni :
		for (i, item) in enumerate(self.mwlist):
			if len(item) > self.ni:
				raise RuntimeError("One item of your mwlist has too many weights!")
			for (inputi, weight) in enumerate(item):
				self.layers[0].weights[self.ni+i,inputi] = weight
			
		
		
		
	
	def get_paramslice(self, mode=None):
		"""
		Returns a slice object describing the parameters that should be optimized, in the context of a given mode.
		
		:param mode: If "sum", all parameters of the sum-layers are included.
			If "mult", only the parameters of the first mult-layer are in.
		
		"""
		
		logger.info("Preparing a paramslice to mode '{}'...".format(mode))
		
		ntotparams = self.nparams()
		nmultparams = self.layers[0].nparams()
		nsumparams = ntotparams - nmultparams
		
		if mode == "mult":
			# Those params, of the first layer, are the **last** in the paramslist, so we skip the first part:
			return slice(nsumparams, ntotparams)
		if mode == "sum":
			# We use only the first params:
			return slice(0, nsumparams)
		elif mode == None: # Empty slice, use all params
			return slice(None)	
		else:
			raise ValueError("Unknown mode!")
		
		
