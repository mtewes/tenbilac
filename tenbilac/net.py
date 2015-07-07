"""
This is Tenbilac!
"""

import numpy as np
import scipy.optimize

import logging
logger = logging.getLogger(__name__)

from . import act
from . import layer
from . import utils

class Tenbilac():
	"""
	Object representing a network made out of one or several hidden layers.
	"""
	
	def __init__(self, ni, nhs, no=1, onlyid=False):
		"""
		:param ni: Number of input features
		:param nhs: Numbers of neurons in hidden layers
		:type nhs: tuple
		:param no: Number of ouput neurons
		:param onlyid: Set this to true if you want identity activation functions on all layers
			(useful for debugging).
		"""
	
		self.ni = ni
		self.nhs = nhs
		self.no = no
		self.arch = np.array([self.ni]+self.nhs+[self.no])
		
		
		self.layers = [] # We build a list containing only the hidden layers and the output layer
		for (i, nh) in enumerate(self.nhs + [self.no]):
				self.layers.append(layer.Layer(ni=self.arch[i], nn=nh, actfct=act.Tanh(), name=str(i)))
		# For the output layer, set id activation function:
		self.layers[-1].actfct = act.Id()
		
		if onlyid: # Then all layers get the Id activation function:
			for l in self.layers:
				l.actfct = act.Id()
		
		
		# We initialize some counters for the optimization:
		self.optit = 0 # The iteration counter
		self.optcall = 0 # The cost function call counter
		self.opterr = np.inf # The current cost function value
		
		self.opterrs = [] # The cost function value at each call
		self.optiterrs = [] # The cost function value at each iteration
		self.optitcalls = [] # The cost function call counter at each iteration
		
		
		logger.info("Built " + str(self))

	
	def __str__(self):
		"""
		A short string describing the network
		"""
		return "Tenbilac with architecture {self.arch} and {nparams} params".format(self=self, nparams=self.nparams())

	
	def report(self):
		"""
		Returns a text about the network parameters, useful for debugging.
		"""
		txt = ["="*120, str(self)]
		for l in self.layers:
			txt.append(l.report())
		txt.append("="*120)
		return "\n".join(txt)

	
	def save(self, filepath):
		"""
		Saves self into a pkl file
		"""
		utils.writepickle(self, filepath)		
	
	
	def nparams(self):
		"""
		Returns the number of parameters of the network
		"""
		return sum([l.nparams() for l in self.layers])
		
	
	def get_params_ref(self, schema=2):
		"""
		Get a single 1D numpy array containing references to all network weights and biases.
		Note that each time you call this, you loose the "connection" to the ref from any previous calls.
		
		:param schema: different ways to arrange the weights and biases in the output array.
		
		"""
		
		ref = np.empty(self.nparams())
		ind = 0
		
		if schema == 1: # First layer first, weights and biases.
		
			for l in self.layers:
				ref[ind:ind+(l.nn*l.ni)] = l.weights.flatten() # makes a copy
				ref[ind+(l.nn*l.ni):ind+l.nparams()] = l.biases.flatten() # makes a copy
				l.weights = ref[ind:ind+(l.nn*l.ni)].reshape(l.nn, l.ni) # a view
				l.biases = ref[ind+(l.nn*l.ni):ind+l.nparams()] # a view
				ind += l.nparams()
		
		elif schema == 2: # Starting at the end, biases before weights
		
			for l in self.layers[::-1]:
			
				ref[ind:ind+l.nn] = l.biases.flatten() # makes a copy
				ref[ind+l.nn:ind+l.nparams()] = l.weights.flatten() # makes a copy
				l.biases = ref[ind:ind+l.nn] # a view
				l.weights = ref[ind+l.nn:ind+l.nparams()].reshape(l.nn, l.ni) # a view
				ind += l.nparams()
			
		else:
			raise ValueError("Unknown schema")
			
		
		# Note that such tricks do not work, as indexing by indices creates copies:
		#indices = np.arange(self.nparams())
		#np.random.shuffle(indices)
		#return ref[indices]

		assert ref.size == self.nparams()
		return ref



	def addnoise(self, **kwargs):
		"""
		Adds random noise to all parameters.
		"""
		
		logger.debug("Adding noise to network parameters...")
		
		for l in self.layers:
			l.addnoise(**kwargs)
			
			
	def setidentity(self):
		"""
		Adjusts the network parameters so to approximatively get an identity relation
		between the ith output and the ith input (for each i in the outputs).
		
		This should be a good starting position for "calibration" tasks. Example: first
		input feature is observed galaxy ellipticity g11, and first output is true g1.
		"""

		logger.info("Setting identity weights...")
		
		for l in self.layers:
			l.zero() # Sets everything to zero
			if l.nn < self.no or self.ni < self.no:
				raise RuntimeError("Network is too small for setting identity!")
			
		for io in range(self.no):
			for l in self.layers:
				l.weights[io, io] = 1.0 # Now we set selected weights to 1.0 (leaving biases at 0.0)
			
			

	def run(self, inputs):
		"""
		Propagates input through the network. This works for 1D, 2D, and 3D inputs, see layer.run().
		Note that this forward-running does not care about the fact that some of the inputs might be masked!
		"""
		
		output = inputs
		for l in self.layers:
			output = l.run(output)
		return output
	
			
	
	def optcallback(self, *args):
		"""
		Function called by the optimizer after each "iteration".
		Print out some info about the training progress,
		saves status of the counters,
		and optionally writes the network itself to disk.
		"""
		#print args
		self.optit += 1
		self.optiterrs.append(self.opterr)
		self.optitcalls.append(self.optcall)
		logger.info("Training iteration {self.optit:4d}, cost = {self.opterr:.8e}".format(self=self))
		if self.tmpitersavefilepath != None:
			self.save(self.tmpitersavefilepath)
		
	
	
	def train(self, inputs, targets, errfct, maxiter=100, itersavefilepath=None, verbose=True):
		"""
		First attempt of black-box training to minimize the given errfct
		
		
		:param itersavefilepath: Path to save the network at each optimization callback.
		:type itersavefilepath: string
		
		"""
			
		logger.info("Starting training with input = {intype} of shape {inshape} and targets = {tartype} of shape {tarshape}".format(
			intype=str(type(inputs)), inshape=str(inputs.shape), tartype=str(type(targets)), tarshape=str(targets.shape)))
	
		if inputs.ndim != 3 and targets.ndim != 2:
			raise ValueError("Sorry, for training I only accept 3D input and 2D targets.")
		
		assert type(targets) == np.ndarray # This should not be masked
		
		# We will "run" the network without paying attention to the masks.
		# Instead, we now manually generate a mask for the ouputs, so that the errorfunction can disregard the masked realizations.
		# Indeed all this masking stays the same for given training data, no need to compute this at every iteration...
		
		if isinstance(inputs, np.ma.MaskedArray):
			
			# First the consequence of the masked inputs: If any feature of a realization is maksed, the full realization should
			# be disregarded.
			assert inputs.mask.ndim == 3
			outputsmask = np.any(inputs.mask, axis=1) # This is 2D (rea, gal)
			# Let's also compute a mask for galaxies, just to see how many are affected:
			galmask = np.any(outputsmask, axis=0) # This is 1D (gal)
			galmaskall = np.all(outputsmask, axis=0) # This is 1D (gal)
			
			txt = []
			
			txt.append("In total {0} realizations ({1:.2%}) will be disregarded due to {2} masked features.".format(np.sum(outputsmask), float(np.sum(outputsmask))/float(outputsmask.size), np.sum(inputs.mask)))
			txt.append("This affects {0} ({1:.2%}) of the {2} training cases,".format(np.sum(galmask), float(np.sum(galmask))/float(galmask.size), galmask.size))
			txt.append("and {0} ({1:.2%}) of the training cases have no useable realizations at all.".format(np.sum(galmaskall), float(np.sum(galmaskall))/float(galmaskall.size)))
			
			logger.info(" ".join(txt))
			
			# Now we inflate this outputsmask to make it 3D (rea, label, gal)
			# Values are the same for all labels, but this is required for easy use in the error functions.
			outputsmask = np.swapaxes(np.tile(outputsmask, (self.no, 1, 1)), 0, 1)
			
			inputs = inputs.filled(fill_value=0.0) # We no longer need this to be a masked array.
			
		else:
			outputsmask = None
		
		assert type(inputs) == np.ndarray
		
		# Preparing stuff used by the callback function to save progress:
		if itersavefilepath != None:
			self.tmpitersavefilepath = itersavefilepath
			# And let's test this out before we start, so that it fails fast in case of a problem:
			self.save(self.tmpitersavefilepath)
		else:
			self.tmpitersavefilepath = None
		
	
		params = self.get_params_ref(schema=2)
		
		
		def cost(p):
			params[:] = p
			outputs = self.run(inputs) # This is not a masked array!
			if outputsmask == None:
				err = errfct(outputs, targets)
			else:
				err = errfct(np.ma.array(outputs, mask=outputsmask), targets)
			self.opterr = err
			self.optcall += 1
			self.opterrs.append(err)
			
			if verbose:
				logger.debug("Iteration {self.optit:4d}, call number {self.optcall:8d}: cost = {self.opterr:.8e}".format(self=self))
				logger.debug("\n" + self.report())
			return err
		
		
		optres = scipy.optimize.fmin_bfgs(
			cost, params,
			fprime=None,
			maxiter=maxiter, gtol=1e-08,
			full_output=True, disp=True, retall=True, callback=self.optcallback)
		
		"""
		optres = scipy.optimize.fmin_powell(
			cost, params,
			maxiter=maxiter, ftol=1e-06,
			full_output=True, disp=True, retall=True, callback=self.optcallback)
		"""
		"""
		optres = scipy.optimize.fmin(
			cost, params,
			xtol=0.0001, ftol=0.0001, maxiter=maxiter, maxfun=None,
			full_output=True, disp=True, retall=True, callback=self.optcallback)
		"""
		"""
		optres = scipy.optimize.minimize(
			cost, params, method="Anneal",
			jac=None, hess=None, hessp=None, bounds=None, constraints=(),
			tol=None, callback=self.optcallback, options={"maxiter":maxiter, "disp":True})
		"""
		
		"""
		optres = scipy.optimize.basinhopping(
			cost, params, 
			niter=maxiter, T=0.001, stepsize=1.0, minimizer_kwargs=None, take_step=None, accept_test=None,
			callback=self.optcallback, interval=50, disp=True, niter_success=None)
		"""
		
		
		#print optres
		if len(optres) == 8:
			(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs) = optres
			
			finalerror = cost(xopt) # Is it important to do this, to set the optimal parameters?
			
			logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
			
		else:
			logger.warning("Optimization output is fishy")
	
		
	
	
	
