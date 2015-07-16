"""
Training a network happens here
"""

import numpy as np
import scipy.optimize
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

from . import layer
from . import utils
from . import err
from . import act



class Training:
	"""
	Holds together everthing related to the process of training a Tenbilac.
	"""

	
	def __init__(self, net, inputs, targets, valfrac=0.1, shuffle=True, errfctname="msrb", itersavepath=None, verbose=False, name=None):
		"""
		
		
		:param valfrac: Fraction of training data which should be used for the validation
		
		
		Sets up
		- inputs and targets (with respective masks)
		- housekeeping lists
		- error function
		
		
		# Naming conventions:
		fulltraininputs = the full training set
		valinputs = the full validation set
		traininputs = a potential "mini batch" subset of the full training set
		
		
		"""

		if inputs.ndim != 3 and targets.ndim != 2:
			raise ValueError("Sorry, for training I only accept 3D input and 2D targets.")
		logger.info("Setting up the training with {ncases} cases and {nreas} realizations...".format(ncases=inputs.shape[2], nreas=inputs.shape[0]))
		#logger.info("Training data: inputs = {intype} of shape {inshape} and targets = {tartype} of shape {tarshape}".format(
		#	intype=str(type(inputs)), inshape=str(inputs.shape), tartype=str(type(targets)), tarshape=str(targets.shape)))
		
		# The acutal net is an attribute of a training:
		
		self.net = net
		self.name = name
		self.params = self.net.get_params_ref(schema=2) # Fast connection to the network parameters
			
		# We will "run" the network without paying attention to the masks.
		# Instead, we now manually generate a mask for the ouputs, so that the errorfunction can disregard the masked realizations.
		# Indeed all this masking stays the same for given training data, no need to compute this at every iteration...
		
		(nomaskinputs, outputsmask) = utils.demask(inputs, no=self.net.no)
		assert type(targets) == np.ndarray # This should not be masked
		
		# Now we cut away part of this for validation purposes, and shuffle before doing so.
		
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


				
		# By default, without batches, the full data will get used:
		self.fullbatch()
		
		# Setting up the cost function
		self.errfctname = errfctname
		
		# We initialize some counters for the optimization:
		self.optit = 0 # The iteration counter
		self.optcall = 0 # The cost function call counter
		self.optitcall = 0 # Idem, but gets reset at each new iteration
		self.opterr_train = np.inf # The current cost function value on the training set
		self.opterr_val = np.inf # The current cost function value on the validation set
		
		# And some lists describing the optimization:
		self.opterrs = [] # The cost function value on the training set at each (!) call
		
		self.optiterrs = [] # The cost function value at each iteration
		self.optitparams = [] # A copy of the network parameters at each iteration
		self.optitcalls = [] # The cost function call counter at each iteration
		self.optittimes = [] # Time taken for iteration, in seconds
		
		self.verbose = verbose
		self.itersavepath = itersavepath
		
		logger.info("Setup {self}".format(self=self))
		
		# And let's test this out before we start, so that it fails fast in case of a problem:
		if self.itersavepath is not None:
			self.save(self.itersavepath)
			

	def __str__(self):
		#return "Training using {self.errfctname} on {ncases} cases with {nrea} realizations".format(self=self, ncases=self.fullinputs.shape[2], nrea=self.fullinputs.shape[0])
		autotxt = "T_{self.net}({self.errfctname}/{nrea}*{ntraincases}|{nvalcases})".format(
			self=self, ntraincases=self.fulltraininputs.shape[2], 
			nvalcases=self.valinputs.shape[2],
			nrea=self.fulltraininputs.shape[0])
		return autotxt
	
	

	def save(self, filepath):
		"""
		Saves self into a pkl file
		"""
		utils.writepickle(self, filepath)		



	def fullbatch(self):
		"""
		Sets the full training sample as batch training data.
		"""
		self.traininputs = self.fulltraininputs
		self.trainoutputsmask = self.fulltrainoutputsmask # even if None, this works
		self.traintargets = self.fulltraintargets	
		


	def random_minibatch(self, size=10):
		"""
		Selects a random minibatch of the full training set
		"""
		
		ncases = self.fulltraininputs.shape[2]
		if size > ncases:
			raise RuntimeError("Cannot select {size} among {ncases}".format(size=size, ncases=ncases))
		
		
		logger.info("Randomly seleting new minibatch of {size} among {ncases} cases...".format(size=size, ncases=ncases))
		caseindexes = np.arange(ncases)
		np.random.shuffle(caseindexes)
		caseindexes = caseindexes[0:size]
			
		self.traininputs = self.fulltraininputs[:,:,caseindexes]
		self.traintargets = self.fulltraintargets[:,caseindexes]
		
		if self.fulltrainoutputsmask is not None:
			self.trainoutputsmask = self.fulltrainoutputsmask[:,:,caseindexes] # Yes, outputsmask is 3D
		else:
			self.trainoutputsmask = None
		
		
		"""
		if batchsize is not None:
			self.batchsize = batchsize
			self.nbatch = int(inputs.shape[2] // batchsize) # floor division: number of batch samples we can do
			self.batchlist = [(batchi*batchsize, (batchi+1)*batchsize) for batchi in range(self.nbatch)]
			self.batchi = 0 # We will start with the first batch
			
			logger.info("Will work with {self.nbatch} batches, each containing {self.batchsize} training cases.".format(self=self))
		else:
			self.batchsize = None
		"""
		
		
	def start(self):
		"""
		Called a the beginning of a training 
		"""
		self.testcost()
		self.iterationstarttime = datetime.now()
		self.optitcall = 0
		
	
	def end(self):
		"""
		Called at the end of a training
		"""
		self.optitcall = 0
		logger.info("Total training time: {0:.2f} s".format(np.sum(self.optittimes)))
		

	def callback(self, *args):
		"""
		Function called by the optimizer after each "iteration".
		Print out some info about the training progress,
		saves status of the counters,
		and optionally writes the network itself to disk.
		"""
		#print args
		#exit()
		
		self.optit += 1
		now = datetime.now()
		secondstaken = (now - self.iterationstarttime).total_seconds()
		callstaken = self.optitcall 
		
		self.optittimes.append(secondstaken)
		self.optiterrs.append(self.opterr)
		self.optitcalls.append(self.optcall)
		self.optitparams.append(args[0])
		
		callstaken = self.optitcall 
		
		logger.info("Iteration {self.optit:4d}, {self.errfctname} = {self.opterr:.8e}, took {time:.4f} s for {calls} calls ({avg:.4f} s per call)".format(self=self, time=secondstaken, calls=callstaken, avg=float(secondstaken)/float(callstaken)))
		
		
		if self.itersavepath != None:
			self.save(self.itersavepath)
		
		# We reset the iteration counters:
		self.iterationstarttime = now
		self.optitcall = 0 
			
		# And now we take care of getting a new batch
		#self.randombatch()
		
		
		
	def cost(self, p):
		"""
		The as-fast-as-possible function to compute the error based on parameters p.
		This gets called repeatedly by the optimizers.
		"""
	
		errfct = eval("err.{0}".format(self.errfctname))
	
		self.params[:] = p # Updates the network parameters
		outputs = self.net.run(self.traininputs) # This is not a masked array!
		if self.trainoutputsmask is None:
			err = errfct(outputs, self.traintargets)
		else:
			err = errfct(np.ma.array(outputs, mask=self.trainoutputsmask), self.traintargets)
			
		self.opterr = err
		self.optcall += 1
		self.optitcall += 1
		self.opterrs.append(err)
		
		if self.verbose:
			logger.debug("Iteration {self.optit:4d}, call number {self.optcall:8d}: cost = {self.opterr:.8e}".format(self=self))
			logger.debug("\n" + self.net.report())
			
		return err


	def currentcost(self):
		return self.cost(p=self.params)


	def testcost(self):
		"""
		Calls the cost function and logs some info.
		"""
		
		logger.info("Testing cost function call...")
		starttime = datetime.now()
		err = self.currentcost()
		endtime = datetime.now()
		took = (endtime - starttime).total_seconds()
		
		logger.info("Done in {took:.4f} seconds. Current state: {self.errfctname} = {self.opterr:.8e}".format(self=self, took=took))
	
	
	
	def minibatch_bfgs(self, size=100, nloops=10, maxiter=10):
		
		for loopi in range(nloops):
			self.random_minibatch(size=size)
			self.bfgs(maxiter=maxiter)
			
			
	

	def bfgs(self, maxiter=100, gtol=1e-8):
		
		self.start()
		logger.info("Starting BFGS for {0} iterations (maximum)...".format(maxiter))
		
		optres = scipy.optimize.fmin_bfgs(
			self.cost, self.params,
			fprime=None,
			maxiter=maxiter, gtol=gtol,
			full_output=True, disp=True, retall=False, callback=self.callback)
		
		if len(optres) == 7:
			(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag) = optres
			self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
			logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
		else:
			logger.warning("Optimization output is fishy")
		
		self.end()
	


	def cg(self, maxiter):
		
		self.start()
		logger.info("Starting CG for {0} iterations (maximum)...".format(maxiter))
		
		optres = scipy.optimize.fmin_cg(
			self.cost, self.params,
			fprime=None, gtol=1e-05,
			maxiter=maxiter, full_output=True, disp=True, retall=False, callback=self.callback)
			
		if len(optres) == 5:
			(xopt, fopt, func_calls, grad_calls, warnflag) = optres
			self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
			logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
		else:
			logger.warning("Optimization output is fishy")
	
		self.end()
	





#	def anneal(self, maxiter=100):
#		
#		self.testcost()
#		logger.info("Starting annealing for {0} iterations (maximum)...".format(maxiter))
#	
#		optres = scipy.optimize.basinhopping(
#			self.cost, self.params, 
#			niter=maxiter, T=0.001, stepsize=0.1, minimizer_kwargs=None, take_step=None, accept_test=None,
#			callback=self.callback, interval=100, disp=True, niter_success=None)
#			
#			# Warning : interval is not the callback interval, but the step size update interval.
#
#		print optres
#		
#		print len(optres)
		
#	def fmin(self, maxiter=100):	# One iteration per call
#		self.testcost()
#		logger.info("Starting fmin for {0} iterations (maximum)...".format(maxiter))
#		
#		optres = scipy.optimize.fmin(
#			self.cost, self.params,
#			xtol=0.0001, ftol=0.0001, maxiter=maxiter, maxfun=None,
#			full_output=True, disp=True, retall=True, callback=self.callback)
#		
#		print optres

	
#		"""
#		optres = scipy.optimize.fmin_powell(
#			cost, params,
#			maxiter=maxiter, ftol=1e-06,
#			full_output=True, disp=True, retall=True, callback=self.optcallback)
#		"""
#		"""
#		optres = scipy.optimize.fmin(
#			cost, params,
#			xtol=0.0001, ftol=0.0001, maxiter=maxiter, maxfun=None,
#			full_output=True, disp=True, retall=True, callback=self.optcallback)
#		"""
#		"""
#		optres = scipy.optimize.minimize(
#			cost, params, method="Anneal",
#			jac=None, hess=None, hessp=None, bounds=None, constraints=(),
#			tol=None, callback=self.optcallback, options={"maxiter":maxiter, "disp":True})
#		"""
#		
#		"""
#		optres = scipy.optimize.basinhopping(
#			cost, params, 
#			niter=maxiter, T=0.001, stepsize=1.0, minimizer_kwargs=None, take_step=None, accept_test=None,
#			callback=self.optcallback, interval=50, disp=True, niter_success=None)
#		"""
