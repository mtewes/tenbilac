"""
Little helpers
"""

import numpy as np
import os
import cPickle as pickle
import gzip


import logging
logger = logging.getLogger(__name__)


class Normer:
	"""
	Class helping to "normalize" data, linearly rescaling it to be within 0 and 1 (type="01"),
	-1 and 1 (type="-11", or around 0 with a std of 1 (type="std").
	"""

	def __init__(self, x, type="01"):

		self.type = type
		
		x = np.asfarray(x)
		
		if x.ndim != 2:
			raise ValueError('x must have 2 dimensions')
			
		if type == "01":
			min = np.min(x, axis=0)
			dist = np.max(x, axis=0) - min
			min.shape = (1, min.size)
			dist.shape = (1, dist.size)
			self.a = min
			self.b = dist
			
		elif type == "std":
			avg = np.mean(x, axis=0)
			std = np.std(x, axis=0)
			avg.shape = (1, avg.size)
			std.shape = (1, std.size)
			self.a = avg
			self.b = std
	
		else:
			raise RuntimeError("Unknown Normer type")		
		
		logger.info(str(self))
		

	def __str__(self):
		return "Normer of type '{self.type}': a={self.a}, b={self.b}".format(self=self)


	def __call__(self, x):
		x = np.asfarray(x)
		res = (x - self.a) / self.b
		return res

	def renorm(self, x):
		x = np.asfarray(x)
		res = x * self.b + self.a
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
