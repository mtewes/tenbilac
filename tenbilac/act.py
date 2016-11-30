"""
Activation functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

def sig(x):
	return 1.0 / (1.0 + np.exp(-x))	# The actual sigmoid
	
			
def sige(x):
	return 2.0 / (1.0 + np.exp(-x)) - 1.0	# Making it even
		

def tanh(x):
	return np.tanh(x)
	
	
def iden(x):
	return x

def relu6(x):
	return np.amin([np.amax([x, np.zeros_like(x)], axis=0), 6 * np.ones_like(x)], axis=0)

def relu(x):
	return np.amax([x, np.zeros_like(x)], axis=0)


if __name__ == "__main__":
	
	import matplotlib.pyplot as plt
	
	x = np.linspace(-5, 5, 1000)
	acts = [sig, sige, tanh, iden, relu]
	
	for act in acts:
		plt.plot(x, act(x), label=act.__name__)
		
	plt.xlabel(r"$x$")
	plt.ylabel(r"$f(x)$")
	plt.title("Activation functions")
	plt.ylim(-1.2, 1.2)
	plt.legend()
	plt.grid()
	plt.show()
	


#class Sig:
#	"""
#	Sigmoid	
#	"""
#
#	def __call__(self, x):
#		#return 1.0 / (1.0 + np.exp(-x))	# The actual sigmoid
#		return 2.0 / (1.0 + np.exp(-x)) - 1.0	# Making it even
#		
#		
#
#class Tanh:
#	"""
#	Tanh	
#	"""
#
#	def __call__(self, x):
#		return np.tanh(x)
#	
#	
#class Id:
#	"""
#	Plain identity
#	"""
#	
#	def __call__(self, x):
#		return x
#
#
#if __name__ == "__main__":
#	
#	import matplotlib.pyplot as plt
#	
#	x = np.linspace(-5, 5, 100)
#	acts = [Sig, Tanh, Id]
#	
#	for act in acts:
#		f = act()
#		plt.plot(x, f(x), label=act.__name__)
#		
#	plt.xlabel(r"$x$")
#	plt.ylabel(r"$f(x)$")
#	plt.title("Activation functions")
#	plt.ylim(-1.2, 1.2)
#	plt.legend()
#	plt.grid()
#	plt.show()
	
	
