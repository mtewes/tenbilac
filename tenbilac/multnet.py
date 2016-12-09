"""
A MultNet is a variant of a Net, with a first hidden layer performing multiplications.
This first layer needs special initialization, and is not optimized at the same pace as the second part.
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
	
		nmult = ni + len(mwlist)
		net.Net.__init__(self, ni, [-nmult]+nhs, **kwargs)
	
		self.multini(mwlist)
		

	
	def multini(self, mwlist):
		"""
		Initialize the mult layer in a very custom way.
		The first ni neurons are set to simply transport.
		The settings for the other ones are controlled by mwlist.
		
		:param mwlist: list of tuples containing the weights for each neuron of the mult-layer in addition
			to the "initial-identity" neurons.
			Example: mwlist=[(1, 1)] adds a single neuron that multiplies the first two inputs.
		
		"""
		
		self.layers[0].setzero()
		self.layers[0].setidentity(onlyn=self.ni)
		
		# And now the custom part
		for (i, item) in enumerate(mwlist):
			if len(item) > self.ni:
				raise RuntimeError("One item of your mwlist has too many weights!")
			for (inputi, weight) in enumerate(item):
				self.layers[0].weights[self.ni+i,inputi] = weight
			
		
	def get_paramslice(self, mode=None):
		"""
		Returns a slice object describing the parameters that should be optimized.
		
		:param mode: If "sum", all parameters of the sum-layers are included.
			If "mult", only the parameters of the first mult-layer are in.
		
		"""
		
		if mode == "sum":
			
			
			logger.info("Set paramslice to mode '{}'".format(mode))
		else:
			raise ValueError("Unknown mode!")
		
		
