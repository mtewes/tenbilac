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

	
	def __init__(self, net, inputs, targets, errfctname="msrb", itersavepath=None, verbose=False):
		"""
		
		Sets up
		- inputs and targets (with respective masks)
		- housekeeping lists
		- error function
		
		"""
		if inputs.ndim != 3 and targets.ndim != 2:
			raise ValueError("Sorry, for training I only accept 3D input and 2D targets.")
		logger.info("Setting up the training with {ncases} and {nreas} realizations...".format(ncases=inputs.shape[2], nreas=inputs.shape[0]))
		#logger.info("Training data: inputs = {intype} of shape {inshape} and targets = {tartype} of shape {tarshape}".format(
		#	intype=str(type(inputs)), inshape=str(inputs.shape), tartype=str(type(targets)), tarshape=str(targets.shape)))
		
		# The acutal net is an attribute of a training:
		self.net = net
		self.params = self.net.get_params_ref(schema=2) # Fast connection to the network parameters
		
		# We will "run" the network without paying attention to the masks.
		# Instead, we now manually generate a mask for the ouputs, so that the errorfunction can disregard the masked realizations.
		# Indeed all this masking stays the same for given training data, no need to compute this at every iteration...
		
		(self.fullinputs, self.fulloutputsmask) = utils.demask(inputs, no=self.net.no)
		
		assert type(targets) == np.ndarray # This should not be masked
		self.fulltargets = targets
		
		# By default, without batches, the full data will get used:
		self.fullbatch()
		
		# Setting up the cost function
		self.errfctname = errfctname
		
		# We initialize some counters for the optimization:
		self.optit = 0 # The iteration counter
		self.optcall = 0 # The cost function call counter
		self.opterr = np.inf # The current cost function value
		
		self.opterrs = [] # The cost function value at each call
		self.optiterrs = [] # The cost function value at each iteration
		self.optitparams = [] # A copy of the network parameters at each iteration
		self.optitcalls = [] # The cost function call counter at each iteration
		self.optittimes = [] # Time taken for iteration, in seconds
		
		self.verbose = verbose
		self.itersavepath = itersavepath
		
		# And let's test this out before we start, so that it fails fast in case of a problem:
		if self.itersavepath is not None:
			self.save(self.itersavepath)
			

	def __str__(self):
		return "Training using {self.errfctname} on {ncases} cases with {nrea} realizations".format(self=self, ncases=self.fullinputs.shape[2], nrea=self.fullinputs.shape[0])
	

	def save(self, filepath):
		"""
		Saves self into a pkl file
		"""
		utils.writepickle(self, filepath)		



	def fullbatch(self):
		"""
		Sets the full training sample as batch training data.
		"""
		self.inputs = self.fullinputs
		self.outputsmask = self.fulloutputsmask
		self.targets = self.fulltargets	
		


	def random_minibatch(self, size=10):
		"""
		Selects a random minibatch of the full training set
		"""
		
		ncases = self.fullinputs.shape[2]
		if size > ncases:
			raise RuntimeError("Cannot select {size} among {ncases}".format(**locals()))
		
		
		logger.info("Randomly seleting new minibatch of {size} cases...".format(**locals()))
		caseindexes = np.arange(ncases)
		np.random.shuffle(caseindexes)
		caseindexes = caseindexes[0:size]
			
		self.inputs = self.fullinputs[:,:,caseindexes]
		self.targets = self.fulltargets[:,caseindexes]
		
		if self.fulloutputsmask is not None:
			self.outputsmask = self.fulloutputsmask[:,caseindexes]
		
		
		
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
		self.optittimes.append(datetime.now())
		self.optiterrs.append(self.opterr)
		self.optitcalls.append(self.optcall)
		self.optitparams.append(args[0])
		
		if len(self.optittimes) > 1: # If it's not the first time callback is called:
			secondstaken = (self.optittimes[-1] - self.optittimes[-2]).total_seconds()
			callstaken = (self.optitcalls[-1] - self.optitcalls[-2])
			logger.info("Iteration {self.optit:4d}, {self.errfctname} = {self.opterr:.8e}, took {time:.4f} s for {calls} calls ({avg:.4f} s per call)".format(self=self, time=secondstaken, calls=callstaken, avg=float(secondstaken)/float(callstaken)))
		else:
			logger.info("Iteration {self.optit:4d}, {self.errfctname} = {self.opterr:.8e}".format(self=self))
	
		if self.itersavepath != None:
			self.save(self.itersavepath)
			
			
		# And now we take care of getting a new batch
		#self.randombatch()
		
		
		
		
	def cost(self, p):
		"""
		The as-fast-as-possible function to compute the error based on parameters p.
		This gets called repeatedly by the optimizers.
		"""
	
		errfct = eval("err.{0}".format(self.errfctname))
	
		self.params[:] = p # Updates the network parameters
		outputs = self.net.run(self.inputs) # This is not a masked array!
		if self.outputsmask is None:
			err = errfct(outputs, self.targets)
		else:
			err = errfct(np.ma.array(outputs, mask=self.outputsmask), self.targets)
			
		self.opterr = err
		self.optcall += 1
		self.opterrs.append(err)
		
		if self.verbose:
			logger.debug("Iteration {self.optit:4d}, call number {self.optcall:8d}: cost = {self.opterr:.8e}".format(self=self))
			logger.debug("\n" + self.net.report())
			
		return err


	def testcost(self):
		"""
		Calls the cost function and logs some info.
		"""
		
		logger.info("Testing cost function call...")
		starttime = datetime.now()
		err = self.cost(self.params)
		endtime = datetime.now()
		took = (endtime - starttime).total_seconds()
		
		logger.info("Done in {took:.4f} seconds. Current state: {self.errfctname} = {self.opterr:.8e}".format(self=self, took=took))
		


	def bfgs(self, maxiter=100, gtol=1e-8):
		
		self.testcost()
		logger.info("Starting BFGS for {0} iterations (maximum)...".format(maxiter))
		
		optres = scipy.optimize.fmin_bfgs(
			self.cost, self.params,
			fprime=None,
			maxiter=maxiter, gtol=gtol,
			full_output=True, disp=True, retall=True, callback=self.callback)
		
		if len(optres) == 8:
			(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs) = optres
			self.cost(xopt) # Is it important to do this, to set the optimal parameters? It seems not.
			logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
		else:
			logger.warning("Optimization output is fishy")
	

	
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
