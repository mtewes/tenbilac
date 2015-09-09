"""
Training a network happens here
"""

import numpy as np
import scipy.optimize
from datetime import datetime
import os

import logging
logger = logging.getLogger(__name__)

from . import layer
from . import utils
from . import err
from . import act
from . import plot



class Training:
	"""
	Holds together everthing related to the process of training a Net: the training data and the network.
	"""

	
	def __init__(self, net, dat, errfctname="msrb", itersavepath=None, autoplotdirpath=".", autoplot=False, verbose=False, name=None):
		"""

		Sets up
		- housekeeping lists
		- error function
				
		"""
		
		self.name = name
	
		self.set_net(net)
		self.set_dat(dat)
		
		# Let's check compatibility between those two!
		assert net.ni == self.dat.getni()
		assert net.no == self.dat.getno()
				
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
		self.optbatchchangeits = [] # Iteration counter (in fact indices) when the batche gets changed
		
		self.verbose = verbose
		self.itersavepath = itersavepath
		
		self.autoplotdirpath = autoplotdirpath
		self.autoplot = autoplot
		
		logger.info("Done with setup of {self}".format(self=self))
		
		# And let's test this out before we start, so that it fails fast in case of a problem.
		# But we only do so if the file does not yet exist, to avoid overwriting an existing training.
		if self.itersavepath is not None:
			if not os.path.exists(self.itersavepath):
				self.save(self.itersavepath)
	
	
	def set_dat(self, dat):
		"""
		Allows to add or replace training data (e.g. when reading a self.save()...)
		"""
		self.dat = dat
		self.set_datstr()
		

	def set_datstr(self):
		"""
		We do this so that we can still display this string on plots if dat has been removed.
		"""
		self.datstr = str(self.dat)
	

	def set_net(self, net):
		"""
		Replaces the network object
		"""
		self.net = net
		self.params = self.net.get_params_ref(schema=2) # Fast connection to the network parameters
		
	

	def __str__(self):
		"""
		A short spaceless automatic description
		"""	
		if self.dat is not None: # If available, we use the live dat.
			datstr = str(self.dat)
		else: # Otherwise, we use datstr as backup solution.
			datstr = getattr(self, "datstr", "Ouch!") # To ensure that it works also if datstr is missing (backwards compatibility).
		autotxt = "{self.errfctname}({self.net}, {datstr})".format(self=self, datstr=datstr)				
		return autotxt
	
	def title(self):
		"""
		Returns the name and string, typically nicer for plots.
		"""
		
		
		if self.name is not None:
			return "Training '{name}': {auto} ({it} it, {tmin:.1f} min)".format(name=self.name, auto=str(self), it=self.optit, tmin=np.sum(self.optittimes)/60.0)
		else:
			return "{auto} ({it} it, {tmin:.1f} min)".format(auto=str(self), it=self.optit, tmin=np.sum(self.optittimes)/60.0)
	
	
	def takeover(self, othertrain):
		"""
		This copies the net and all the progress counters and logs from another train object into the current one.
		Useful if you want to carry on a training with a new train object, typically with different settings and
		maybe on different data.
		"""
		
		logger.info("Setting up training '{}' to take over the work from '{}'...".format(self.name, othertrain.name))
		
		if not self.net.nparams() == othertrain.net.nparams():
			raise RuntimeError("Other network is not compatible, this is fishy!")
		
		self.set_net(othertrain.net)
		
		# Copying the counter positions:
		self.optit = othertrain.optit
		self.optcall = othertrain.optcall
		self.optitcall = 0 # Gets reset at each new iteration
		self.opterr = othertrain.opterr
		
		# And some lists describing the optimization:
		self.opterrs = othertrain.opterrs[:]
		
		self.optitparams = othertrain.optitparams[:]
		self.optiterrs_train = othertrain.optiterrs_train[:]
		self.optiterrs_val = othertrain.optiterrs_val[:]

		self.optitcalls = othertrain.optitcalls[:]
		self.optittimes = othertrain.optittimes[:]
		try:
			self.optbatchchangeits = othertrain.optbatchchangeits[:]
		except AttributeError:
			self.optbatchchangeits = []
		

		logger.info("Done with the takeover")
		
	

	def save(self, filepath, keepdata=False):
		"""
		Saves the training progress into a pkl file
		As the training data is so massive, by default we do not save it!
		Note that this might be done at each iteration!
		"""

		if keepdata is True:
			logger.info("Writing training to disk and keeping the data...")
			utils.writepickle(self, filepath)	
		else:
			tmptraindata = self.dat
			self.set_datstr()
			self.dat = None
			utils.writepickle(self, filepath)		
			self.dat = tmptraindata



	def makeplots(self, suffix="_optitXXXXX", dirpath=None):
		"""
		Saves a bunch of default checkplots into the specified directory.
		Can typically be called at the end of training, or after iterations.
		
		"""
		
		if dirpath is None:
			dirpath = self.autoplotdirpath
		
		if suffix == "_optitXXXXX":
			suffix = "_optit{0:05d}".format(self.optit)
		
		logger.info("Making and writing plots, with suffix '{}'...".format(suffix))
		plot.paramscurve(self, os.path.join(dirpath, "paramscurve"+suffix+".png"))
		plot.outdistribs(self, os.path.join(dirpath, "outdistribs"+suffix+".png"))
		plot.errorinputs(self, os.path.join(dirpath, "errorinputs"+suffix+".png"))
		logger.info("Done with plots")


	def start(self):
		"""
		Called a the beginning of a training 
		"""
		self.testcost()
		self.iterationstarttime = datetime.now()
		self.optitcall = 0
		
	
	def end(self):
		"""
		Called at the end of a training (each minibatch) depening on the algo.
		"""
		self.optitcall = 0
		logger.info("Cumulated training time: {0:.2f} s".format(np.sum(self.optittimes)))
		if self.autoplot:
			self.makeplots()
		

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
		self.iterationstarttime = datetime.now()
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
			err = self.errfct(np.ma.array(outputs, mask=self.dat.valoutputsmask), self.dat.valtargets)
		return err
		
	
	
	def minibatch_bfgs(self, mbsize=100, mbloops=10, **kwargs):
		
		for loopi in range(mbloops):
			if mbloops > 1:
				logger.info("Starting minibatch loop {loopi} of {mbloops}...".format(loopi=loopi+1, mbloops=mbloops))
			self.dat.random_minibatch(mbsize=mbsize)
			self.optbatchchangeits.append(self.optit) # We record this minibatch change
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
