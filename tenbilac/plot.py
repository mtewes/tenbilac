"""
Plots directly related to tenbilac objects
"""

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)



def checkdata(data, filepath=None):
	"""
	Simply plots histograms of the different features
	Checks normalization
	
	:param data: 2D or 3D numpy array with (feature, gal) or (rea, feature, gal).
	:type data: numpy array
	
	"""

	fig = plt.figure(figsize=(10, 10))

	if data.ndim == 3:

		for i in range(data.shape[1]):
			
			ax = fig.add_subplot(3, 3, i)
			vals = data[:,i,:].flatten()
			ran = (np.min(vals), np.max(vals))
			
			ax.hist(vals, bins=100, range=ran, color="gray")
			#ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
			#ax.set_ylabel(r"$d$", fontsize=18)
			#ax.set_xlim(-1.2, 2.4)
			#ax.set_ylim(1.6, 3.1)

	plt.tight_layout()
	plt.show()	



def errorcurve(train, filepath=None):
	"""
	Simple plot of the error curve generated during the training of a network
	
	"""
	
	fig = plt.figure(figsize=(10, 10))

	
	# The cost function calls:
	opterrs = np.array(train.opterrs)
	optcalls = np.arange(len(train.opterrs)) + 1
	plt.plot(optcalls, opterrs, "r-")
	
	# The "iterations":
	optiterrs = np.array(train.optiterrs)
	optitcalls = np.array(train.optitcalls)
	optits = np.arange(len(train.optiterrs)) + 1
	
	plt.plot(optitcalls, optiterrs, "b.")
	ax = plt.gca()
	for (optit, optiterr, optitcall) in zip(optits, optiterrs, optitcalls):
		if optit % 5 == 0:
		    ax.annotate("{0}".format(optit), xy=(optitcall, optiterr))


	ax.set_yscale('log')
	plt.xlabel("Cost function call")
	plt.ylabel("Cost function value")

	plt.tight_layout()
	plt.show()	

	
	
def paramscurve(train, filepath=None):
	"""
	Visualization of the evolution of the network parameters during the training
	
	"""
	
	fig = plt.figure(figsize=(10, 10))
	
	
	optparams = np.array(train.optitparams)
	opterrs = train.optiterrs
	optits = np.arange(len(train.optiterrs)) + 1
	
	ax = plt.subplot(2, 1, 1)
	ax.plot(optits, opterrs)
	ax.set_yscale('log')
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Cost function value ({0})".format(train.errfctname))
	
	ax = plt.subplot(2, 1, 2)
	assert optparams.shape[1] == train.net.nparams()
	for paramindex in range(train.net.nparams()):
		ax.plot(optits, optparams[:,paramindex])
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Network parameter value")
	
	plt.tight_layout()
	plt.show()	


	
	
	
	
	
	
	
	

