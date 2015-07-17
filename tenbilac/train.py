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
	Holds together everthing related to the process of training a Tenbilac: the training data and the network.
	"""

	
	def __init__(self, net, dat, errfctname="msrb", itersavepath=None, verbose=False, name=None):
		"""

		Sets up
		- housekeeping lists
		- error function
				
		"""

		self.dat = dat
		self.net = net
		
		# Let's check compatibility between those two!
		assert net.ni == self.dat.getni()
		assert net.no == self.dat.getno()
		
		
		self.name = name
		self.params = self.net.get_params_ref(schema=2) # Fast connection to the network parameters
			
			
		# Setting up the cost function
		self.errfctname = errfctname
		self.errfct = eval("err.{0}".format(self.errfctname))
		
		# We initialize some counters for the optimization:
		self.optit = 0 # The iteration counter
		self.optcall = 0 # The cost function call counter
		self.optitcall = 0 # Idem, but gets reset at each new iteration
		self.opterr = np.inf # The current cost function value on the training set
		
		# And some lists describing the optimization:
		self.opterrs = [] # The cost function value on the training set at each (!) call
		
		self.optitparams = [] # A copy of the network parameters at each iteration
		self.optiterrs_train = [] # The cost function value on the training set at each iteration
		self.optiterrs_val = [] # The cost function value on the validation set at each iteration

		self.optitcalls = [] # The cost function call counter at each iteration
		self.optittimes = [] # Time taken for iteration, in seconds
		
		self.verbose = verbose
		self.itersavepath = itersavepath
		
		logger.info("Done with setup of {self}".format(self=self))
		
		# And let's test this out before we start, so that it fails fast in case of a problem:
		if self.itersavepath is not None:
			self.save(self.itersavepath)
	
	
	def set_dat(self, dat):
		"""
		Allows to add or replace training data (e.g. when reading a self.save()...)
		"""
		self.dat = dat
	

	def __str__(self):
		
		autotxt = "{self.errfctname}({self.net}, {self.dat})".format(self=self)				
		return autotxt
	
	

	def save(self, filepath):
		"""
		Saves the training progress into a pkl file
		As the training data is so massive, we do not save it!
		"""
		
		tmptraindata = self.dat
		self.sat = None
		utils.writepickle(self, filepath)		
		self.dat = tmptraindata


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
		logger.info("Cumulated training time: {0:.2f} s".format(np.sum(self.optittimes)))
		

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
		self.optiterrs_train.append(self.opterr)
		self.optitcalls.append(self.optcall)
		self.optitparams.append(args[0])
		
		# Now we evaluate the cost on the validation set:
		valerr = self.valcost()
		self.optiterrs_val.append(valerr)
		
		valerrratio = valerr / self.opterr
		
		mscallcase = 1000.0 * float(secondstaken) / (float(callstaken) * self.dat.getntrain()) # Time per call and training case
		
		logger.info("Iter. {self.optit:4d}, {self.errfctname} train = {self.opterr:.6e}, val = {valerr:.6e} ({valerrratio:4.1f}), {time:.4f} s for {calls} calls ({mscallcase:.4f} ms/cc)".format(
			self=self, time=secondstaken, valerr=valerr, valerrratio=valerrratio, calls=callstaken, mscallcase=mscallcase))
		
		if self.itersavepath != None:
			self.save(self.itersavepath)
		
		# We reset the iteration counters:
		self.iterationstarttime = now
		self.optitcall = 0 
			
		# And now we take care of getting a new batch
		#self.randombatch()
		
		
		
	def cost(self, p):
		"""
		The "as-fast-as-possible" function to compute the training error based on parameters p.
		This gets called repeatedly by the optimizers.
		"""
	
		self.params[:] = p # Updates the network parameters
		outputs = self.net.run(self.dat.traininputs) # This is not a masked array!
		if self.dat.trainoutputsmask is None:
			err = self.errfct(outputs, self.dat.traintargets)
		else:
			err = self.errfct(np.ma.array(outputs, mask=self.dat.trainoutputsmask), self.dat.traintargets)
			
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
		
		logger.info("Testing cost function calls...")
		starttime = datetime.now()
		err = self.currentcost()
		endtime = datetime.now()
		took = (endtime - starttime).total_seconds()		
		logger.info("On the training set:   {took:.4f} seconds, {self.errfctname} = {self.opterr:.8e}".format(self=self, took=took))
		starttime = datetime.now()
		err = self.valcost()
		endtime = datetime.now()
		took = (endtime - starttime).total_seconds()		
		logger.info("On the validation set: {took:.4f} seconds, {self.errfctname} = {err:.8e}".format(self=self, took=took, err=err))
		
	
		
	def valcost(self):
		"""
		Evaluates the cost function on the validation set.
		"""
		outputs = self.net.run(self.dat.valinputs) # This is not a masked array!
		if self.dat.valoutputsmask is None:
			err = self.errfct(outputs, self.dat.valtargets)
		else:
			err = errfct(np.ma.array(outputs, mask=self.dat.valoutputsmask), self.dat.valtargets)
		return err
		
	
	
	def minibatch_bfgs(self, mbsize=100, mbloops=10, **kwargs):
		
		for loopi in range(mbloops):
			if mbloops > 1:
				logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi, mbloops=mbloops))
			self.dat.random_minibatch(mbsize=mbsize)
			self.bfgs(**kwargs)
			
	

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
