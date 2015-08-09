"""
Classes to hold and manipulate data
"""

import numpy as np


import logging
logger = logging.getLogger(__name__)


class Normer:
	"""
	A Normer provides methods to "normalize" and later "denormalize" a numpy array, linearly rescaling it to be within 0 and 1 (type="01"),
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
			# For this, we need to compute the min and max value for every feature, along the realizations and cases.
		
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
			
			if x.ndim == 3: # First using rollaxis to reshape array instead of using fancy reshape modes seemed safer...
				xreshaped = np.rollaxis(x, axis=1) # brings the feature as first index.
				xreshaped = np.reshape(xreshaped, (x.shape[1], -1)) # mixes reas and cases in second index.
				avgs = np.mean(xreshaped, axis=1) # along reas and cases
				stds = np.std(xreshaped, axis=1)
				
			elif x.ndim == 2: # Easy !
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



def demask(indata, no=1):
	"""
	Function that "splits" a potentially masked input 3D array into unmasked input and some appropriate mask
	that can be applied to the output.
	This allows us to write the neural network itself as if no data was masked, as long as the cost function
	is aware of the mask.
	
	The whole point: if any feature of a realization is maksed,
	the full realization should be disregarded.
	
	:param indata: 3D numpy array (rea, feature, case), typically input for training or prediction.
	:param no: The number of outputs of the network (only used to properly format the returned mask).
	
	:returns: a tuple (filleddata, outputsmask), where filledata has exactly the same shape as data,
		and outputsmask is 3D but with only "no" feature dimensions (rea, =no, case)
		If the input data is not masked, the returned outputsmask is "None".
	
	"""
	assert indata.ndim == 3
	
	if isinstance(indata, np.ma.MaskedArray):
			
		assert indata.mask.ndim == 3
		
		outputsmask = np.any(indata.mask, axis=1) # This is 2D (rea, gal)
		
		# Let's also compute a mask for galaxies, just to see how many are affected:
		galmask = np.any(outputsmask, axis=0) # This is 1D (gal)
		galmaskall = np.all(outputsmask, axis=0) # This is 1D (gal)
			
		txt = []
			
		txt.append("In total {0} realizations ({1:.2%}) will be disregarded due to {2} masked features.".format(np.sum(outputsmask), float(np.sum(outputsmask))/float(outputsmask.size), np.sum(indata.mask)))
		txt.append("This affects {0} ({1:.2%}) of the {2} cases,".format(np.sum(galmask), float(np.sum(galmask))/float(galmask.size), galmask.size))
		txt.append("and {0} ({1:.2%}) of the cases have no useable realizations at all.".format(np.sum(galmaskall), float(np.sum(galmaskall))/float(galmaskall.size)))
			
		logger.info(" ".join(txt))
			
		# Now we inflate this outputsmask to make it 3D (rea, label, gal)
		# Values are the same for all labels, but this is required for easy use in the error functions.
		outputsmask = np.swapaxes(np.tile(outputsmask, (no, 1, 1)), 0, 1)
			
		filleddata = indata.filled(fill_value=0.0) # Gives us a plain ndarray without mask.
		assert type(filleddata) == np.ndarray
			
	else:
		assert type(indata) == np.ndarray
		logger.info("The data has no mask, so nothing to demask...")
		filleddata = indata
		outputsmask = None
	
	return (filleddata, outputsmask)
	
	




class Traindata:
	"""
	A class to hold training data so that it can be efficiently used by Tenbilac.
	It has methods to set validation data, swap minibatches etc.
	
	Here we avoid masked arrays. Instead, we carry around boolean masks to be used by the error functions
	after the unmasked arrays were propagated through the network.
	# We will "run" the network without paying attention to the masks.
	# Instead, we now manually generate a mask for the ouputs, so that the errorfunction can disregard the masked realizations.
	# Indeed all this masking stays the same for given training data, no need to compute this at every iteration...
	
	Naming conventions:
	fulltraininputs = the full training set
	valinputs = the full validation set
	traininputs = the current "mini batch" subset of the full training set (or the full training set, if no minibatch is set)
		
	"""



	def __init__(self, inputs, targets, valfrac=0.5, shuffle=True):
		"""
		
		:param inputs: masked 3D array with the inputs, I'll take care of demasking it.
		:param targets: 2D array with the targets (should not be masked, as targets should all be known...)
		
		:param valfrac: Fraction of training data which should be used for the validation
	
		"""
		
		logger.info("Setting up the training data...")
		
		if inputs.ndim != 3 and targets.ndim != 2:
			raise ValueError("Sorry, for training I only accept 3D input and 2D targets.")
		if type(targets) != np.ndarray:
			raise ValueError("Sorry, targets should not be masked")
	
		# We split the mask apart:
		(nomaskinputs, outputsmask) = demask(inputs, no=targets.shape[0])
		

		# Now we cut away part of the data for validation purposes, and shuffle before doing so.
		
		ncases = inputs.shape[2]
		nvalcases = int(valfrac * ncases)
		if nvalcases <= 0:
			raise RuntimeError("Please allow for some validation cases.")
		ntraincases = ncases - nvalcases

		if shuffle:
			logger.info("Shuffling training data and selecting {nvalcases} among {ncases} cases for validation...".format(ncases=ncases, nvalcases=nvalcases))
			caseindexes = np.arange(ncases)
			np.random.shuffle(caseindexes)
			trainindexes = caseindexes[0:ntraincases]
			valindexes = caseindexes[ntraincases:ncases]
			
			self.fulltraininputs = nomaskinputs[:,:,trainindexes]
			self.valinputs = nomaskinputs[:,:,valindexes]
			self.fulltraintargets = targets[:,trainindexes]
			self.valtargets = targets[:,valindexes]
			
			if outputsmask is None:
				self.fulltrainoutputsmask = None
				self.valoutputsmask = None
			else:
				self.fulltrainoutputsmask = outputsmask[:,:,trainindexes]
				self.valoutputsmask = outputsmask[:,:,valindexes]
				
			
		else: # then we just slice the arrays:
			logger.info("Selecting the last {nvalcases} among {ncases} cases for validation (no shuffling)".format(ncases=ncases, nvalcases=nvalcases))
			
			self.fulltraininputs = nomaskinputs[:,:,0:ntraincases]
			self.valinputs = nomaskinputs[:,:,ntraincases:ncases]
			self.fulltraintargets = targets[:,0:ntraincases]
			self.valtargets = targets[:,ntraincases:ncases]
			
			if outputsmask is None:
				self.fulltrainoutputsmask = None
				self.valoutputsmask = None
			else:
				self.fulltrainoutputsmask = outputsmask[:,:,0:ntraincases]
				self.valoutputsmask = outputsmask[:,:,ntraincases:ncases]
			
		# Let's check that all this looks good:
		assert self.fulltraininputs.shape[2] == ntraincases
		assert self.valinputs.shape[2] == nvalcases			
		assert self.fulltraintargets.shape[1] == ntraincases
		assert self.valtargets.shape[1] == nvalcases			
		if outputsmask is not None:
			assert self.fulltrainoutputsmask.shape[2] == ntraincases
			assert self.valoutputsmask.shape[2] == nvalcases			

		# By default we set the full training set as batch:
		self.fullbatch()


	def __str__(self):
		
		if self.getntrain() == self.getnfulltrain():
			ntraintxt = "{nfulltrain}".format(nfulltrain=self.getnfulltrain())	
		else:
			ntraintxt = "{nfulltrain}({ntrain})".format(nfulltrain=self.getnfulltrain(), ntrain=self.getntrain())	
		
		return "{nrea}*{ntraintxt}|{nval}".format(ntraintxt=ntraintxt, nval=self.getnval(), nrea=self.getnrea())



	def getnrea(self):
		"""
		The number of realizations
		"""
		return self.fulltraininputs.shape[0]

	def getni(self):
		"""
		The number of "features", i.e. input nodes of the ANN
		"""
		return self.fulltraininputs.shape[1]

	def getno(self):
		"""
		The number of labels, i.e. output nodes of the ANN
		"""
		return self.fulltraintargets.shape[0]

	def getntrain(self):
		"""
		Number of training cases in batch
		"""
		return self.traininputs.shape[2]
	
	def getnfulltrain(self):
		"""
		Total number of training cases (not validation cases!)
		"""
		return self.fulltraininputs.shape[2]

	def getnval(self):
		"""
		Number of validation cases
		"""
		return self.valinputs.shape[2]


	def fullbatch(self):
		"""
		Sets the full training sample as batch training data.
		"""
		logger.info("Setting the full training set to be used as batch.")
		self.traininputs = self.fulltraininputs
		self.trainoutputsmask = self.fulltrainoutputsmask # even if None, this works
		self.traintargets = self.fulltraintargets	


	def random_minibatch(self, mbsize=10):
		"""
		Selects a random minibatch of the full training set
		:param mbsize: if None, will call fullbatch.
		"""
		
		if mbsize is None:
			self.fullbatch()
			return
		
		nfulltrain = self.getnfulltrain()
		if mbsize > nfulltrain:
			raise RuntimeError("Cannot select {mbsize} among {nfulltrain}".format(mbsize=mbsize, nfulltrain=nfulltrain))
		
		
		logger.info("Randomly seleting new minibatch of {mbsize} among {nfulltrain} cases...".format(mbsize=mbsize, nfulltrain=nfulltrain))
		caseindexes = np.arange(nfulltrain)
		np.random.shuffle(caseindexes)
		caseindexes = caseindexes[0:mbsize]
			
		self.traininputs = self.fulltraininputs[:,:,caseindexes]
		self.traintargets = self.fulltraintargets[:,caseindexes]
		
		if self.fulltrainoutputsmask is not None:
			self.trainoutputsmask = self.fulltrainoutputsmask[:,:,caseindexes] # Yes, outputsmask is 3D
		else:
			self.trainoutputsmask = None
		

	


