"""
General helpers
"""

import numpy as np
import os
import cPickle as pickle
import gzip


import logging
logger = logging.getLogger(__name__)



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

def sigma_clip_plus(arr, maxdev, get_indices=False):
	"""
	Removes iteratively all points that are upwards of 

		mean(arr) + maxdev * std(arr)

	:param arr: a numpy array (1D) containing the data points.
	:param maxdev: maximum allowed deviation
	:param get_indices: if True returns arr, keys otherwise just keys
	"""

	lenp = 1
	lena = 0

	keys = np.arange(len(arr))
	arr = np.asarray(arr)

	while lenp > lena:
		_thr = np.mean(arr) + maxdev * np.std(arr)
		lenp = np.size(arr)
		sel = arr < _thr
		keys = keys[sel]
		arr = arr[sel]
		lena = np.size(arr)
		
		if lena == 1:
			break

	if get_indices:
		return arr, keys
	else:
		return arr
