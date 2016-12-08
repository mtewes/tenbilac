"""
Plots directly related to tenbilac objects
"""

import numpy as np
import itertools
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import err
from . import net
from . import wnet

import logging
logger = logging.getLogger(__name__)



def errevo(ax, train, showtimes=True):
	"""
	Plots the evolution of the error curve during the training, iteration per iteration.
	
	:param ax: matplotlib axes
	:param showtimes: If True, some training times are written on the curve, in minutes.
	"""

	logger.info("Preparing error evolution plot for {train}".format(train=str(train)))
	# Preparint the data:
	optiterrs_train = np.array(train.optiterrs_train)
	optiterrs_val = np.array(train.optiterrs_val)
	optits = np.arange(len(train.optitparams))
	optbatchchangeits = getattr(train, "optbatchchangeits", [])
	
	# The cpu durations
	optittimes = np.array(train.optittimes)
	cumoptittimes = np.cumsum(optittimes)/60.0 # minutes
	assert cumoptittimes.size == optittimes.size
	
	if optittimes.size > 10:
		labelstep = int(float(optittimes.size)/10.0)
		timeindices = range(labelstep, optittimes.size, labelstep)
	else:
		timeindices = []

	for optbatchchangeit in optbatchchangeits:
		ax.axvline(optbatchchangeit, color="gray")

	ax.plot(optits, optiterrs_train, ls="-", color="black", label="Training batch")
	ax.plot(optits, optiterrs_val, ls="--", color="red", label="Validation set")
	
	if showtimes:
		for i in timeindices:
			ax.annotate("{0:.1f}".format(cumoptittimes[i]), xy=(optits[i], optiterrs_val[i]), xytext=(0, 10), textcoords='offset points')
	
	ax.set_yscale('log')
	ax.set_xlabel("Iteration")
	ax.set_xlim((optits[0], optits[-1]))
	ax.set_ylabel("Cost function value ({0})".format(train.errfctname))
	ax.legend()
	ax.set_title(train.title())





def paramsevo(ax, train, wnetpart=None):
	"""
	Plots the evolution of the actual network parameters.
	
	""" 
	
	optits = np.arange(len(train.optitparams))
	
	if isinstance(train.net, wnet.WNet):
		if wnetpart == "o":
			mynet = train.net.neto
			optitparams = np.array(train.optitparams)[:,:train.net.neto.nparams()]
			
		elif wnetpart == "w":
			mynet = train.net.netw
			optitparams = np.array(train.optitparams)[:,train.net.neto.nparams():]
		else:
			raise ValueError("This is a WNet, please specify a wnetpart")
	else: # We have a normal Net:
		mynet = train.net
		optitparams = np.array(train.optitparams)
	
	paramlabels = mynet.get_paramlabels()
	

	assert optitparams.shape[1] == mynet.nparams()
	for paramindex in range(mynet.nparams()):
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
	ax.set_xlim((optits[0], optits[-1]))
	ax.set_ylabel("Network parameter value")
	
	# Now creating the legend
	
	black_patch = matplotlib.patches.Patch(color='black', label='Output layer')
	red_patch = matplotlib.patches.Patch(color='blue', label='Hidden layers')
	line = matplotlib.lines.Line2D([], [], color='black', marker='', ls="-", label='Weight')
	dashed = matplotlib.lines.Line2D([], [], color='black', marker='', ls="--", label='Bias')
	
	ax.legend(handles=[line, dashed, black_patch, red_patch])



	
def sumevo(train, filepath=None, showtimes=True):
	"""
	Visualization of the evolution of the network parameters and error during the training,
	iteration per iteration
	
	:param showtimes: If True, some training times are written on the curve, in minutes.
	"""
	
	
	fig = plt.figure(figsize=(10, 10))
	ax = plt.subplot(2, 1, 1)
	errevo(ax, train, showtimes=showtimes)
	
	ax = plt.subplot(2, 1, 2)
	paramsevo(ax, train, wnetpart=None)
	
	plt.tight_layout()
	if filepath is None:
		plt.show()	
	else:
		logger.info("Writing paramscurve to {0}".format(filepath))
		plt.savefig(filepath)
	plt.close() # Important, otherwise it's still around for the next plt.show()



def biasevo(train, filepath=None):
	"""
	Viz of the evolution of the individual bias terms at each iteration (massive!)
	"""
	
	nsnaps = len(train.biassnaps_it)
	if nsnaps < 1:
		logger.warning("No snapshots to biasevo plot!")
		return
		
		
	trainbiases = np.array(train.biassnaps_train) # Members of this list have shape (outputs, cases). Now its (iterations, outputs, cases)
	valbiases = np.array(train.biassnaps_val) # Members of this list have shape (outputs, cases). Now its (iterations, outputs, cases)
	
	biases = np.dstack((trainbiases, valbiases)) # shape is (iteration, output-neuron, case)
	
	assert nsnaps == biases.shape[0]
	ncas = biases.shape[2]
	
	# For colors, we get the input data
	if train.dat is None:
		logger.warning("Need dat for biasevo plot!")
		return
	meantraininputs = np.mean(train.dat.fulltraininputs, axis=0) # shape is (feature, case)
	meanvalinputs = np.mean(train.dat.valinputs, axis=0) # shape is (feature, case)
	meaninputs = np.hstack((meantraininputs, meanvalinputs)) # shape is still (feature, case)
	
	assert meaninputs.shape[1] == biases.shape[2]
	
	logger.info("Preparing biasevo plot with {} cases and {} snapshots...".format(ncas, nsnaps))
	

	fig = plt.figure(figsize=(8*train.net.no, 5*train.net.ni))
	
	for io in range(train.net.no):
		for ii in range(train.net.ni):
			ax = plt.subplot(train.net.ni, train.net.no, io*train.net.ni+1+ii)
			
			#for optbatchchangeit in optbatchchangeits:
			#	ax.axvline(optbatchchangeit, color="gray", zorder=-20)
			ax.axhline(0.0, color="gray", zorder=-20)
			
			cmap = matplotlib.cm.get_cmap("jet")
			sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(meaninputs[ii,:]), vmax=np.max(meaninputs[ii,:])))
			
			for icas in range(ncas):
				plt.plot(train.biassnaps_it, biases[:,io, icas], color=sm.to_rgba(meaninputs[ii, icas]), alpha=0.1)
		
			ax.set_xlabel("Iteration (only snapshots at minibatch-changes are shown)")
			ax.set_ylabel("Bias on target '{}'".format(train.net.onames[io]))
	
			ax.set_yscale("symlog", linthreshy=1.e-3, linscaley=1)
			
			# fake up the array of the scalar mappable. Urgh...
			#sm._A = []
			sm.set_array([])
			cax = plt.colorbar(sm)
			cax.set_label("Feature '{}' (mean across reas)".format(train.net.inames[ii]))
			
	
	plt.tight_layout()
	if filepath is None:
		plt.show()	
	else:
		logger.info("Writing biasevo to {0}".format(filepath))
		plt.savefig(filepath)
	plt.close() # Important, otherwise it's still around for the next plt.show()

	

def outdistribs(train, filepath=None):
	"""
	Viz of the output 
	"""
	ncol = 7
	fig = plt.figure(figsize=(4*ncol, 3.5*train.net.no))
	
	dat = train.dat
	net = train.net
	assert dat is not None
	
	logger.info("Computing predictions...")
	trainoutputs = np.ma.array(net.run(dat.traininputs), mask=dat.trainoutputsmask) # masked, 3D
	valoutputs =  np.ma.array(net.run(dat.valinputs), mask=dat.valoutputsmask)
	logger.info("Done")

	trainerrors = trainoutputs - dat.traintargets # 3D - 2D = 3D
	valerrors = valoutputs - dat.valtargets

	trainbiases = np.mean(trainerrors, axis=0) # 2D (node, case)
	valbiases = np.mean(valerrors, axis=0)
	
	trainstds = np.std(trainoutputs, axis=0) # 2D
	valstds = np.std(valoutputs, axis=0)
	
	valmsrbterms = err.msrb(valoutputs, dat.valtargets, rawterms=True) # 2D
	trainmsrbterms = err.msrb(trainoutputs, dat.traintargets, rawterms=True)
	
	for io in range(train.net.no):
		
		# Subplots: (lines, columns, number)
		# We collect the stuff to be plotted as 1D arrays:
		
		# Warning, ravel does ignore the mask, so we use flatten.
		
		# The simple outputs
		thiso_valoutputs = valoutputs[:,io,:].flatten().compressed()
		thiso_trainoutputs = trainoutputs[:,io,:].flatten().compressed()
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+1)
		ax.hist(thiso_valoutputs, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainoutputs, bins=50, histtype="step", color="black", label="Training batch")
		ax.set_yscale('log')
		ax.set_ylabel("Counts (reas and cases)")
		ax.set_xlabel("Pred. output for '{0}'".format(net.onames[io]))
		
		# The targets
		thiso_valtargets = dat.valtargets[io,:].flatten()
		thiso_traintargets = dat.traintargets[io,:].flatten()
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+2)
		ax.hist(thiso_valtargets, bins=20, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_traintargets, bins=20, histtype="step", color="black", label="Training batch")
		ax.set_ylabel("Counts (cases)")
		ax.set_xlabel("Targets '{0}'".format(net.onames[io]))

		# The prediction errors
		thiso_valerrors = valerrors[:,io,:].flatten().compressed()
		thiso_trainerrors = trainerrors[:,io,:].flatten().compressed()
		histrange = (-1.5, 1.5)
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+3)
		ax.hist(thiso_valerrors, bins=50, histtype="step", color="red", label="Validation set", range=histrange)
		ax.hist(thiso_trainerrors, bins=50, histtype="step", color="black", label="Training batch", range=histrange)
		ax.set_ylabel("Counts (reas and cases)")
		ax.set_xlabel("Errors of pred. '{0}'".format(net.onames[io]))
		ax.set_yscale('log')
		ax.set_xlim(histrange)

		# The biases
		thiso_valbiases = valbiases[io,:].flatten().compressed()
		thiso_trainbiases = trainbiases[io,:].flatten().compressed()
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+4)
		ax.hist(thiso_valbiases, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainbiases, bins=50, histtype="step", color="black", label="Training batch")	
		ax.set_yscale('log')	
		ax.set_ylabel("Counts (cases)")
		ax.set_xlabel("Bias of pred. for '{0}'".format(net.onames[io]))

		# The stds
		thiso_valstds = valstds[io,:].flatten().compressed()
		thiso_trainstds = trainstds[io,:].flatten().compressed()
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+5)
		ax.hist(thiso_valstds, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainstds, bins=50, histtype="step", color="black", label="Training batch")	
		ax.set_yscale('log')	
		ax.set_ylabel("Counts (cases)")
		ax.set_xlabel("STD of pred. for '{0}'".format(net.onames[io]))


		# Against each other
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+6)
		assert thiso_trainstds.size == thiso_trainbiases.size # Indeed even if they were masked, their masks should be identical
		assert thiso_valstds.size == thiso_valbiases.size
		ax.plot(thiso_valbiases, thiso_valstds, marker=".", ms=2, ls="None", color="red", label="Validation set", rasterized=True)
		ax.plot(thiso_trainbiases, thiso_trainstds, marker=".", ms=2, ls="None", color="black", label="Training batch", rasterized=True)	
		ax.set_ylabel("STD of pred. for '{0}'".format(net.onames[io]))
		ax.set_xlabel("Bias of pred. for '{0}'".format(net.onames[io]))

		
		# The MSRB terms:
		thiso_valmsrbterms =valmsrbterms[io,:].flatten().compressed()
		thiso_trainmsrbterms = trainmsrbterms[io,:].flatten().compressed()
		ax = plt.subplot(train.net.no, ncol, (io*ncol)+7)
		ax.hist(thiso_valmsrbterms, bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(thiso_trainmsrbterms, bins=50, histtype="step", color="black", label="Training batch")
		ax.set_ylabel("Counts (cases)")
		ax.set_xlabel("Relative biases of '{0}'".format(net.onames[io]))
		ax.set_yscale('log')
	
		
		
	plt.tight_layout()
	if filepath is None:
		plt.show()	
	else:
		logger.info("Writing outdistribs to {0}".format(filepath))
		plt.savefig(filepath)
	plt.close() # Important, otherwise it's still around for the next plt.show()

	
	
def errorinputs(train, filepath=None, io=0):
	"""
	Viz of the prediction errors as function of inputs
	
	:param io: the index of the ouput I should use. If you have only one neuron, this is 0.
	
	"""
	nlines = 3
	
	fig = plt.figure(figsize=(3.3*train.net.ni, 3.5*nlines))
	plt.figtext(0.5, 1.0, train.title(), ha="center", va="top")
	
	dat = train.dat
	net = train.net
	assert dat is not None
	
	logger.info("Computing predictions...")
	trainoutputs = np.ma.array(net.run(dat.traininputs), mask=dat.trainoutputsmask)
	valoutputs =  np.ma.array(net.run(dat.valinputs), mask=dat.valoutputsmask)
	logger.info("Done")
	
	trainerrors = trainoutputs - dat.traintargets # 3D - 2D = 3D: (rea, label, case)
	valerrors = valoutputs - dat.valtargets # idem
	
	#valmsrbterms = err.msrb(valoutputs, dat.valtargets, rawterms=True)
	#trainmsrbterms = err.msrb(trainoutputs, dat.traintargets, rawterms=True)
	
		
	for ii in range(train.net.ni):
			
			
			
		ax = plt.subplot(nlines, train.net.ni, ii+1)
			
		ax.hist(np.ravel(dat.valinputs[:,ii,:]), bins=50, histtype="step", color="red", label="Validation set")
		ax.hist(np.ravel(dat.traininputs[:,ii,:]), bins=50, histtype="step", color="black", label="Training batch")
		ax.set_yscale('log')
		if ii == 0:
			ax.set_ylabel("Counts (reas + cases, masked = 0)")
		#if ii == int(train.net.ni/2):
		#	ax.set_title(train.title())
		#ax.set_xlabel("Input '{0}'".format(net.inames[ii]))
		ax.set_xticklabels([]) # Hide x tick labels
		ax.set_xlim(-1.0, 1.0)
		# in this plot the masked inputs are shown with a value of 0, this is desired and expected.
	
		
		# Now we viz the biases and stds of the predictions (along the reas)
		
		# The biases and stds on the validation set:
		valbiases = np.mean(valerrors, axis=0)[io,:] # 1D (case)
		valstds = np.std(valoutputs, axis=0)[io,:]
		valbiases = np.tile(valbiases, (dat.valinputs.shape[0], 1)) # Inflated to 2D (rea, case), with all reas having the same value.
		valstds = np.tile(valstds, (dat.valinputs.shape[0], 1))
		
		
		# We want to get a version of these particular inputs with a mask, selecting only one node
		valinputs = dat.valinputs[:,ii:ii+1,:] # 3D : (rea, 1, case)
		valinputs = np.ma.array(valinputs, mask=dat.valoutputsmask) # Here an input is masked if any other input of that rea was masked.
		assert valinputs.shape[1] == 1
		valinputs = valinputs[:,0,:] # 2D : (rea, case)
		
		assert valbiases.shape == valinputs.shape
		assert valstds.shape == valinputs.shape
		

		# For the plot, we only use points for which the inputs are unmasked 
		valbiases = valbiases.flatten()
		valstds = valstds.flatten()
		valinputs = valinputs.flatten()
		valbiases = np.ma.array(valbiases, mask=valinputs.mask) # copying the mask
		valstds = np.ma.array(valstds, mask=valinputs.mask)
		valbiases = valbiases.compressed() # Getting rid of the masked elements
		valstds = valstds.compressed()
		valinputs = valinputs.compressed()
		
		assert valinputs.size == valbiases.size
		assert valinputs.size == valstds.size
		
		
		ax = plt.subplot(nlines, train.net.ni, train.net.ni+ii+1)
		ax.plot(valinputs, valbiases, marker=".", color="red", ls="None", ms=1)
		ax.axhline(0.0, color="black", lw=1, ls="--")
		if ii == 0:
			ax.set_ylabel("Bias of pred. for '{0}'".format(net.onames[io]))
		else:
			ax.set_yticklabels([]) # Hide y tick labels
		#ax.set_xlabel("Input '{0}'".format(net.inames[ii]))
		ax.set_xlim(-1.0, 1.0)
		ax.set_xticklabels([]) # Hide x tick labels
		
		ax = plt.subplot(nlines, train.net.ni, 2*train.net.ni+ii+1)
		ax.plot(valinputs, valstds, marker=".", color="red", ls="None", ms=1)
		if ii == 0:
			ax.set_ylabel("STD of pred. for '{0}'".format(net.onames[io]))
		else:
			ax.set_yticklabels([]) # Hide y tick labels
		ax.set_xlabel("Input '{0}'".format(net.inames[ii]))
		ax.set_xlim(-1.0, 1.0)
		
		
	plt.tight_layout()
	if filepath is None:
		plt.show()	
	else:
		logger.info("Writing errorinputs to {0}".format(filepath))
		plt.savefig(filepath)
	plt.close() # Important, otherwise it's still around for the next plt.show()

def draw_link(ax, start, end, **kwargs):
	"""
	Computes a Bezier curve between two points (`start` and `end`) on an axis `ax`
	
	:param ax: the axis to draw on
	:param start: The starting point, must be an array [Sx, Sy]
	:param end: The end point of the curve, must be an array [Ex, Ey]
	
	Any additional kawrgs are directly passed to `patches.PathPatch` to control the line style.
	"""
	
	verts = [
			(start[0], start[1]),  # P0
			(start[0]+0.5, start[1]), # P1
			(end[0]-0.5, end[1]), # P2
			(end[0], end[1]), # P3
    ]

	codes = [Path.MOVETO,
         Path.CURVE4,
         Path.CURVE4,
         Path.CURVE4,
         ]
	
	path = Path(verts, codes)
	patch = patches.PathPatch(path, facecolor='None', **kwargs)
	ax.add_patch(patch)
	
def scale_bias(b, scale=20.):
	"""
	Returns the scaled biased for the `plot.netviz` plot.
	
	:param b: the bias to scale
	:param scale: the scale parameter to be applied on the absolute value of `b`. Default=20
	"""
	return scale * np.abs(b)
	
def scale_weight(w, scale=2.):
	"""
	Returns the scaled weight for the `plot.netviz` plot.
	
	:param w: the weight to scale
	:param scale: the scale parameter to be applied on the absolute value of `w`. Default=2
	"""
	return scale * np.abs(w)

def get_color(v, pos="orange", neg="navy"):
	"""
	Returns a color code for the scalar value `v`.
	
	:param v: scalar value
	:param pos: name or color-code for the positive values
	:param neg: name or color-code for the negative values
	"""
	if v > 0:
		c = pos
	else:
		c = neg
	return c

def netviz(train, title="default", legend=True, filepath=None):
	"""
	Draws a visualisation of the network in the style of the `Tensorflow` playground.
	
	:param train: the Train class for the network
	:param title: The string to display as title, if "default" writes activation fct + err fct. Default=`default`
	:param legend: Whether to show the legend for the tickness of the lines and points. Default=`True`
	:param filepath: The file path to save the data to. If `None` (default) shows the figures.
	"""
	
	net = train.net
	
	nmax = np.amax([net.ni, net.no, np.amax(net.nhs)]) * 1.
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	if title == "default":
		title = "".join([r"$\mathrm{{{n}/{actfct}\ }}$".format(n=l.nn, actfct=l.actfct.__name__) for l in net.layers])
		title += r"$; \mathrm{{{}}}$".format(train.get_costfctname())
	plt.title(title)
	
	plt_kwargs = {'marker':'s', 's':40, 'c':'k', 'zorder':1}

	for ii, l in enumerate(net.layers):
		dy = nmax / 2 - (l.ni * 1.) / 2
		lnis = np.arange(l.ni) + dy
		
		plt.scatter(np.zeros_like(lnis) + ii, lnis, **plt_kwargs)
		
		if ii == 0 :
			# Write the name of the features
			inames = [r"$\mathrm{{{}}}$".format(i) for i in net.inames]
			for iii, inp in enumerate(inames):
				plt.text(ii-0.08, iii+dy, inp.replace("_", "\_"), horizontalalignment='right', verticalalignment='center')
		
		if ii >= len(net.layers) - 1: 
			flnis = np.arange(net.no)
			dyf = (nmax - (net.no * 1.)) * 0.5
		else:
			flnis = np.arange(net.layers[ii+1].ni)
			dyf = (nmax - (net.layers[ii+1].ni * 1.)) * 0.5
		
		for iw, w in enumerate(l.weights):
			# Draw the weights
			for il, link in enumerate(w):
				draw_link(ax, start=[ii, lnis[il]], 
						end=[ii+1, flnis[iw]+dyf], lw=scale_weight(link), edgecolor=get_color(link), zorder=-1)
			
			# Draw the biases	
			plt.scatter([ii+1.], [flnis[iw]+dyf+0.1], c=get_color(l.biases[iw]), edgecolors="None", s=scale_bias(l.biases[iw]))
	
	# Draw output
	nos = np.arange(net.no)
	plt.scatter(np.zeros_like(nos) + ii + 1, nos+dyf, **plt_kwargs)
	
	# Name the output
	onames = [r"$\mathrm{{{}}}$".format(i) for i in net.onames]
	for iii, inp in enumerate(onames):
		plt.annotate(inp.replace("_", "\_"), xy=(ii+1.08, iii+dyf), horizontalalignment='left', verticalalignment='center')
		
	# Draw legend
	if legend:
		ws = [-1.,-0.5,0.5,1.]
		for iw, w in enumerate(ws):
			yy = nos[-1]+dyf + 0.75 + iw * 0.2
			plt.annotate(r"$%1.1f$" % w, xy=(ii+0.95, yy), horizontalalignment='right', verticalalignment='center')
			plt.scatter([ii+1], [yy], c=get_color(w), edgecolors="None", s=scale_bias(w))
			plt.plot([ii+1.07, ii+1.47], [yy, yy], c=get_color(w), lw=scale_weight(w))
	
	# Taking care of a few things
	plt.xlim([-0.23 * (len(net.nhs) + 2),ii+1.7])
	plt.tight_layout()
	plt.axis('off')
	
	if filepath is None:
		plt.show()	
	else:
		logger.info("Writing netviz to {0}".format(filepath))
		plt.savefig(filepath)
	plt.close() # Important, otherwise it's still around for the next plt.show()


#def checkdata(data, filepath=None):
#	"""
#	Simply plots histograms of the different features
#	Checks normalization
#	
#	:param data: 2D or 3D numpy array with (feature, case) or (rea, feature, case).
#	:type data: numpy array
#	
#	"""
#
#	fig = plt.figure(figsize=(10, 10))
#
#	if data.ndim == 3:
#
#		for i in range(data.shape[1]):
#			
#			ax = fig.add_subplot(3, 3, i)
#			vals = data[:,i,:].flatten()
#			ran = (np.min(vals), np.max(vals))
#			
#			ax.hist(vals, bins=100, range=ran, color="gray")
#			#ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
#			#ax.set_ylabel(r"$d$", fontsize=18)
#			#ax.set_xlim(-1.2, 2.4)
#			#ax.set_ylim(1.6, 3.1)
#
#	plt.tight_layout()
#	plt.show()
#
#def errorcurve(train, filepath=None):
#	"""
#	Simple plot of the error curve generated during the training of a network
#	
#	"""
#	
#	fig = plt.figure(figsize=(10, 10))
#
#	
#	# The cost function calls:
#	opterrs = np.array(train.opterrs)
#	optcalls = np.arange(len(train.opterrs)) + 1
#	plt.plot(optcalls, opterrs, "r-")
#	
#	# The "iterations":
#	optiterrs = np.array(train.optiterrs)
#	optitcalls = np.array(train.optitcalls)
#	optits = np.arange(len(train.optiterrs)) + 1
#	
#	plt.plot(optitcalls, optiterrs, "b.")
#	ax = plt.gca()
#	for (optit, optiterr, optitcall) in zip(optits, optiterrs, optitcalls):
#		if optit % 5 == 0:
#		    ax.annotate("{0}".format(optit), xy=(optitcall, optiterr))
#
#
#	ax.set_yscale('log')
#	plt.xlabel("Cost function call")
#	plt.ylabel("Cost function value")
#
#	plt.tight_layout()
#	plt.show()	
