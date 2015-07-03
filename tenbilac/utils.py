"""
General helpers
"""

import numpy as np
import os
import cPickle as pickle
import gzip


import logging
logger = logging.getLogger(__name__)


class Normer:
	"""
	Object providing methods to "normalize" data, linearly rescaling it to be within 0 and 1 (type="01"),
	-1 and 1 (type="-11"), or around 0 with a std of 1 (type="std").
	
	For each feature, the same normalization is applied to all realizations of all galaxies.
	The details of the normalization are kept in the Normer object, so that one can simply use the same
	object to denorm stuff afterwards.
	
	This works with inputs (3D or 2D) and outputs or targets (2D), with indices
	(realization, feature, galaxy) or (feature, galaxy).
	
	"""

	def __init__(self, x, type="01"):
		"""
		Does *not* normalize anything, just determines the normalization parameters!
		"""

		self.type = type
		
		x = np.asfarray(x)
		if x.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")
			
		if type in ["01", "-11"]:
		
			if x.ndim == 3:# If we have several realizations:
				min = np.min(np.min(x, axis=0), axis=1) # Computes the min along the first and thrid axis.
				dist = np.max(np.max(x, axis=0), axis=1) - min
			elif x.ndim == 2:
				min = np.min(x, axis=1) # Only along the second axes (i.e., "galaxies")
				dist = np.max(x, axis=1) - min
							
			self.a = min
			self.b = dist
				
		elif type == "std":
			
			if x.ndim == 3: # Use rollaxis to reshape array ?
				raise ValueError("Not yet implemented")
				
			avg = np.mean(x, axis=1) # Along galaxies
			std = np.std(x, axis=1)
						
			self.a = avg
			self.b = std
			
		else:
			raise RuntimeError("Unknown Normer type")		
		
		logger.info(str(self))
		

	def __str__(self):
		return "Normer of type '{self.type}': a={self.a}, b={self.b}".format(self=self)


	def __call__(self, x):
		"""
		Returns the normalized data.
		"""
		res = np.asfarray(x)
		if res.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")

		assert self.a.ndim == 1
		assert self.b.ndim == 1
		
		if res.shape[-2] != self.a.shape[0]:
			raise RuntimeError("Number of features does not match!")
			
		atiled = np.tile(self.a.reshape(self.a.size, 1), (1, res.shape[-1]))
		btiled = np.tile(self.b.reshape(self.b.size, 1), (1, res.shape[-1]))			
		res = (res - atiled) / btiled
					
		if self.type == "-11":
			res = 2.0*res - 1.0
		
		return res

	def denorm(self, x):
		"""
		Denorms the data
		"""
		res = np.asfarray(x)
		if res.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")

		assert self.a.ndim == 1
		assert self.b.ndim == 1
		
		if self.type == "-11":
			res = (res + 1.0) / 2.0

		if res.shape[-2] != self.a.shape[0]:
			raise RuntimeError("Number of features does not match!")
			
		atiled = np.tile(self.a.reshape(self.a.size, 1), (1, res.shape[-1]))
		btiled = np.tile(self.b.reshape(self.b.size, 1), (1, res.shape[-1]))			
		res = res * btiled + atiled
		
		return res




def writepickle(obj, filepath, protocol = -1):
	"""
	I write your python object obj into a pickle file at filepath.
	If filepath ends with .gz, I'll use gzip to compress the pickle.
	Leave protocol = -1 : I'll use the latest binary protocol of pickle.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath, 'wb')
	else:
		pkl_file = open(filepath, 'wb')
	
	pickle.dump(obj, pkl_file, protocol)
	pkl_file.close()
	logger.info("Wrote %s" % filepath)
	
	
def readpickle(filepath):
	"""
	I read a pickle file and return whatever object it contains.
	If the filepath ends with .gz, I'll unzip the pickle file.
	"""
	if os.path.splitext(filepath)[1] == ".gz":
		pkl_file = gzip.open(filepath,'rb')
	else:
		pkl_file = open(filepath, 'rb')
	obj = pickle.load(pkl_file)
	pkl_file.close()
	logger.info("Read %s" % filepath)
	return obj
