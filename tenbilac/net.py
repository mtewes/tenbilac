"""

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
	This is Tenbilac
	"""
	
	def __init__(self, ni, nhs, nrea=1, onlyid=False):
		"""
		:param ni: Number of input features
		:param nhs: Numbers of neurons in hidden layers
		:param nrea: Number of realizations
		"""
	
		self.ni = ni
		self.nhs = nhs
		self.nrea = nrea
		self.arch = np.array([self.ni]+self.nhs+[1])
		
		#self.nweights = self.arch + 1 # for each node, the bias is just one more weight

		"""
		self.weights = []
		for i in range(1, len(self.arch)):
			self.weights.append(np.zeros((self.arch[i], self.arch[i-1]+1)))
		
		for (i, w) in enumerate(self.weights):
			logger.info("Weights {i} shape: {shape}".format(i=i, shape=w.shape))	
		"""
		
		
		self.layers = [] # We build a list containing only the hidden layers and the output layer
		for (i, nh) in enumerate(self.nhs + [1]):
				self.layers.append(layer.Layer(ni=self.arch[i], nn=nh, actfct=act.Sig(), name=str(i)))
		# For the output layer, set id activation function:
		self.layers[-1].actfct = act.Id()
		
		if onlyid: # Then all layers get the Id activation function:
			for l in self.layers:
				l.actfct = act.Id()
		
		logger.info("Built " + str(self))

	
	def __str__(self):
		"""
		A short description of the network
		"""
		return "Tenbilac with architecture {self.arch} and {nparams} params for {self.nrea} realizations".format(self=self, nparams=self.nparams())

	
	def report(self):
		"""
		Returns a text about the network parameters, useful for debugging.
		"""
		txt = ["="*80, str(self)]
		for l in self.layers:
			txt.append(l.report())
		txt.append("="*80)
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
		
	
	def get_params_ref(self):
		"""
		Get a single 1D numpy array containing references to all network weights and biases.
		
		We use the fact that slicing an array returns a view of it.
		"""
		
		ref = np.empty(self.nparams())
		ind = 0
		for l in self.layers:
		
			ref[ind:ind+(l.nn*l.ni)] = l.weights.flatten() # makes a copy
			ref[ind+(l.nn*l.ni):ind+l.nparams()] = l.biases.flatten() # makes a copy
			l.weights = ref[ind:ind+(l.nn*l.ni)].reshape(l.nn, l.ni) # a view
			l.biases = ref[ind+(l.nn*l.ni):ind+l.nparams()] # a view
			
			ind += l.nparams()
			
		assert ref.size == self.nparams()
		return ref


	def run(self, input):
		"""
		Propagates 2D input through the network. First index is feature, second is galaxy
		"""
		
		output = input
		for l in self.layers:
			output = l.run(output)
		return output
		
	
	def error_mse(self, input, targets):
		"""
		The conventional error function
		"""
		output = self.run(input)
		return np.mean(np.square(output - targets))
		
		
	def error_calib(self, input, targets, splitinds):
		"""
		Error on bias
		"""
		output = self.run(input)
		
		#print output.shape
		
		means = np.array([np.mean(case, axis=1) for case in np.split(output, splitinds, axis=1)]).transpose()
		
		ret = np.mean(np.square(means - targets))
			
		print ret
		return ret
		
	
	def train(self, input, targets, splitinds):
		"""
		
		"""
			
		logger.info("Starting training with input {0} and targets {1}".format(str(input.shape), str(targets.shape)))
	
		#assert len(targets.shape) == 2
		#assert len(input.shape) == 2
		#assert input.shape[0] == targets.shape
		
		for l in self.layers:
			l.addnoise()


		params = self.get_params_ref()
		
		def f(p):
			params[:] = p
			return self.error_calib(input, targets, splitinds)
			#return self.error_mse(input, targets)


		optres = scipy.optimize.fmin_bfgs(
			f, params,
			fprime=None,
			maxiter=100,
			full_output=True, disp=True, retall=True, callback=None)
	
		
		#print optres
		if len(optres) == 8:
			(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs) = optres
			#finalerror = f(xopt)
			logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
			
		else:
			logger.warning("Optimization output is fishy")
	
			
	def predict(self, input):
		"""
		Input is a single case, 1D:
			- features
		"""
		
		
	def masspredict(self, input):
		"""
		Input is an array with several cases, 2D:
			- case
			- feature
		"""

	def propagate(self, train_input):
		"""
		Input has 3 dimensions
		- case
		- feature
		- realization
		"""
		
	