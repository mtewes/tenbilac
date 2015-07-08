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
	
	All methods correctly work with masked arrays.
	
	"""

	def __init__(self, x, type="-11"):
		"""
		Does *not* normalize anything, just determines the normalization parameters!
		"""

		self.type = type
		
		if isinstance(x, np.ma.MaskedArray):
			logger.info("Building Normer with a masked array of shape {0} and {1} masked values".format(str(x.shape), np.sum(x.mask)))
			#np.ma.set_fill_value(x, 1.e20) # To notice things if the array gets filled by error.
		elif isinstance(x, np.ndarray):
			logger.info("Building Normer with an unmasked array of shape {0}".format((x.shape)))
		else:
			raise ValueError("x is not a numpy array")
		
		if x.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")
			
		if x.shape[-2] > 100: # This is the number of features
			raise RuntimeError("Looks like you have more than 100 features or labels, this is probably a array format error !")
			
		if type in ["01", "-11"]:
		
			if x.ndim == 3:# If we have several realizations:
				mins = np.min(np.min(x, axis=0), axis=1) # Computes the min along the first and thrid axis.
				dists = np.max(np.max(x, axis=0), axis=1) - mins
				
				# All these np.min, np.max, np.mean, np.std work as expected also with masked arrays.
			elif x.ndim == 2:
				mins = np.min(x, axis=1) # Only along the second axes (i.e., "galaxies")
				dists = np.max(x, axis=1) - mins
							
			self.a = mins
			self.b = dists
				
		elif type == "std":
			
			if x.ndim == 3: # Use rollaxis to reshape array ?
				raise ValueError("Not yet implemented")
				
			avgs = np.mean(x, axis=1) # Along galaxies
			stds = np.std(x, axis=1)
						
			self.a = avgs
			self.b = stds
			
		else:
			raise RuntimeError("Unknown Normer type")		
		
		logger.info(str(self))
		

	def __str__(self):
		return "Normer of type '{self.type}': a={self.a}, b={self.b}".format(self=self)


	def __call__(self, x):
		"""
		Returns the normalized data.
		"""
		
		if x.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")

		assert self.a.ndim == 1
		assert self.b.ndim == 1
		
		if x.shape[-2] != self.a.shape[0]:
			raise RuntimeError("Number of features does not match!")
			
		atiled = np.tile(self.a.reshape(self.a.size, 1), (1, x.shape[-1]))
		btiled = np.tile(self.b.reshape(self.b.size, 1), (1, x.shape[-1]))			
		res = (x - atiled) / btiled
					
		if self.type == "-11":
			res = 2.0*res - 1.0
		
		return res

	def denorm(self, x):
		"""
		Denorms the data
		"""
		
		if x.ndim not in (2, 3):
			raise ValueError("Cannot handle this array shape")

		assert self.a.ndim == 1
		assert self.b.ndim == 1
		
		if self.type == "-11":
			res = (x + 1.0) / 2.0
		else:
			res = x + 0.0

		if res.shape[-2] != self.a.shape[0]:
			raise RuntimeError("Number of features does not match!")
			
		atiled = np.tile(self.a.reshape(self.a.size, 1), (1, res.shape[-1]))
		btiled = np.tile(self.b.reshape(self.b.size, 1), (1, res.shape[-1]))			
		res = res * btiled + atiled
		
		return res



def demask(data, no=1):
	"""
	Function that "splits" a potentially masked input 3D array into unmasked input and some appropriate mask
	that can be applied to the output.
	This allows us to write the neural network itself as if no data was masked, as long as the cost function
	is aware of the mask.
	
	The whole point: if any feature of a realization is maksed,
	the full realization should be disregarded.
	
	:param data: 3D numpy array (rea, feature, case), typically input for training or prediction.
	:param no: The number of outputs of the network (only used to properly format the returned mask).
	
	:returns: a tuple (filleddata, outputsmask), where filledata has exactly the same shape as data,
		and outputsmask is 3D but with only "no" feature dimensions (rea, =no, case)
		If the input data is not masked, the returned outputsmask is "None".
	
	"""
	assert data.ndim == 3
	
	if isinstance(data, np.ma.MaskedArray):
			
		assert data.mask.ndim == 3
		
		outputsmask = np.any(data.mask, axis=1) # This is 2D (rea, gal)
		
		# Let's also compute a mask for galaxies, just to see how many are affected:
		galmask = np.any(outputsmask, axis=0) # This is 1D (gal)
		galmaskall = np.all(outputsmask, axis=0) # This is 1D (gal)
			
		txt = []
			
		txt.append("In total {0} realizations ({1:.2%}) will be disregarded due to {2} masked features.".format(np.sum(outputsmask), float(np.sum(outputsmask))/float(outputsmask.size), np.sum(data.mask)))
		txt.append("This affects {0} ({1:.2%}) of the {2} cases,".format(np.sum(galmask), float(np.sum(galmask))/float(galmask.size), galmask.size))
		txt.append("and {0} ({1:.2%}) of the cases have no useable realizations at all.".format(np.sum(galmaskall), float(np.sum(galmaskall))/float(galmaskall.size)))
			
		logger.info(" ".join(txt))
			
		# Now we inflate this outputsmask to make it 3D (rea, label, gal)
		# Values are the same for all labels, but this is required for easy use in the error functions.
		outputsmask = np.swapaxes(np.tile(outputsmask, (no, 1, 1)), 0, 1)
			
		filleddata = data.filled(fill_value=0.0) # Gives us a plain ndarray without mask.
		assert type(filleddata) == np.ndarray
			
	else:
		assert type(data) == np.ndarray
		logger.info("The data has no mask, so nothing to demask...")
		filleddata = data
		outputsmask = None
	
	return (filleddata, outputsmask)
	
	


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
