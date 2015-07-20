"""
Plots directly related to tenbilac objects
"""

import numpy as np
import itertools
import re
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines

from . import err

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
	Visualization of the evolution of the network parameters during the training, iteration per iteration
	
	"""
	
	# Getting the data 
	optitparams = np.array(train.optitparams)
	optiterrs_train = np.array(train.optiterrs_train)
	optiterrs_val = np.array(train.optiterrs_val)
	optits = np.arange(len(train.optitparams)) + 1
	optbatchchangeits = getattr(train, "optbatchchangeits", [])
	
	# The cpu durations
	optittimes = np.array(train.optittimes)
	cumoptittimes = np.cumsum(optittimes)
	assert cumoptittimes.size == optittimes.size
	
	if optittimes.size > 10:
		labelstep = int(float(optittimes.size)/10.0)
		timeindices = range(labelstep, optittimes.size, labelstep)
	else:
		timeindices = []
	
	
	paramlabels = train.net.get_paramlabels()
	# Now we find out how many layers we have, to be able to properly attribute colors
	#layernames = [re.match("layer-(.*)_(.*)", label).group(1) for label in paramlabels]
	#difflayernames = list(set(layernames))
	#colors = itertools.cycle(["r", "black", "b", "g"])	
	
	
	fig = plt.figure(figsize=(10, 10))
	ax = plt.subplot(2, 1, 1)
	
	for optbatchchangeit in optbatchchangeits:
		ax.axvline(optbatchchangeit, color="gray")

	ax.plot(optits, optiterrs_train, ls="-", color="black", label="Training batch")
	ax.plot(optits, optiterrs_val, ls="--", color="red", label="Validation set")
	
	#for i in timeindices:
	#	ax.annotate("{0:.1f}".format(cumoptittimes[i]), xy=(optits[i], optiterrs_val[i]), xytext=(0, 10), textcoords='offset points')
	
	ax.set_yscale('log')
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Cost function value ({0})".format(train.errfctname))
	ax.legend()
	ax.set_title(train.title())
	
	ax = plt.subplot(2, 1, 2)
	
	assert optitparams.shape[1] == train.net.nparams()
	for paramindex in range(train.net.nparams()):
		label = paramlabels[paramindex]
		if label.endswith("_bias"):
			ls = "--"
		elif label.endswith("_weight"):
			ls = "-"
		layername = re.match("layer-(.*)_(.*)", label).group(1)
		if layername == "o":
			color="black"
		else:
			color="blue"
		
		pla = ax.plot(optits, optitparams[:,paramindex], ls=ls, color=color)
	ax.set_xlabel("Iteration")
	ax.set_ylabel("Network parameter value")
	
	# Now creating the legend
	
	black_patch = matplotlib.patches.Patch(color='black', label='Output layer')
	red_patch = matplotlib.patches.Patch(color='blue', label='Hidden layers')
	line = matplotlib.lines.Line2D([], [], color='black', marker='', ls="-", label='Weight')
	dashed = matplotlib.lines.Line2D([], [], color='black', marker='', ls="--", label='Bias')
	
	ax.legend(handles=[line, dashed, black_patch, red_patch])
	
	plt.tight_layout()
	plt.show()	






def outdistribs(train, filepath=None):
	"""
	
	"""
	fig = plt.figure(figsize=(18, 5*train.net.no))
	
	dat = train.dat
	net = train.net
	
	trainoutputs = np.ma.array(net.run(dat.traininputs), mask=dat.trainoutputsmask)
	valoutputs =  np.ma.array(net.run(dat.valinputs), mask=dat.valoutputsmask)
	
	trainerrors = trainoutputs - dat.traintargets # 3D - 2D = 3D
	valerrors = valoutputs - dat.valtargets
	
	valmsrbterms = err.msrb(valoutputs, dat.valtargets, rawterms=True)
	trainmsrbterms = err.msrb(trainoutputs, dat.traintargets, rawterms=True)
	
	
	for io in range(train.net.no):
		
		# Subplots: (lines, columns, number)
		ax = plt.subplot(train.net.no, 2, io+1)
		
		
		# We collect the stuff to be plotted as 1D arrays:
		
		thiso_valoutputs = np.ravel(valoutputs[:,io,:])
		thiso_trainoutputs = np.ravel(trainoutputs[:,io,:])
		
		thiso_valtargets = np.ravel(dat.valtargets[io,:])
		thiso_traintargets = np.ravel(dat.traintargets[io,:])
		
		thiso_valerrors = np.ravel(valerrors[:,io,:])
		thiso_trainerrors = np.ravel(trainerrors[:,io,:])
		
		
		# A bit more involved: we compute the terms that go into the MSRB.
		
		thiso_valmsrbterms = np.ravel(valmsrbterms[io,:])
		thiso_trainmsrbterms = np.ravel(trainmsrbterms[io,:])
		
		
		# And now we plot the panels side by side:
		ncol = 4
		
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+1)
		ax.hist(thiso_valoutputs, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainoutputs, bins=50, histtype="step", color="black", label="Training batch")
		ax.set_yscale('log')
		ax.set_ylabel("Counts (mixing all realizations and cases)")
		ax.set_xlabel("Output '{0}'".format(net.onames[io]))
		
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+2)
		ax.hist(thiso_valtargets, bins=10, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_traintargets, bins=10, histtype="step", color="black", label="Training batch")
		ax.set_ylabel("Counts (mixing all cases)")
		ax.set_xlabel("Targets for '{0}'".format(net.onames[io]))
		
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+3)
		ax.hist(thiso_valerrors, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainerrors, bins=50, histtype="step", color="black", label="Training batch")
		ax.set_ylabel("Counts (mixing all realizations and cases)")
		ax.set_xlabel("Errors of '{0}'".format(net.onames[io]))
		ax.set_yscale('log')
		
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+4)
		ax.hist(thiso_valmsrbterms, bins=10, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainmsrbterms, bins=10, histtype="step", color="black", label="Training batch")
		ax.set_ylabel("Counts (mixing all cases)")
		ax.set_xlabel("Relative biases of '{0}'".format(net.onames[io]))
		ax.set_yscale('log')
	
	
	plt.tight_layout()
	plt.show()	

	
	
def errors():
	"""
	Viz of the individual prediction errors per case ?
	
	"""
	
	
	

