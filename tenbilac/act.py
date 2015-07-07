"""
Activation functions
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)


class Sig:
	"""
	Sigmoid	
	"""

	def __call__(self, x):
		#return 1.0 / (1.0 + np.exp(-x))	# The actual sigmoid
		return 2.0 / (1.0 + np.exp(-x)) - 1.0	# Making it even
		
		

class Tanh:
	"""
	Tanh	
	"""

	def __call__(self, x):
		return np.tanh(x)
	
	
class Id:
	"""
	Plain identity
	"""
	
	def __call__(self, x):
		return x


if __name__ == "__main__":
	
	import matplotlib.pyplot as plt
	
	x = np.linspace(-5, 5, 100)
	acts = [Sig, Tanh, Id]
	
	for act in acts:
		f = act()
		plt.plot(x, f(x), label=act.__name__)
		
	plt.xlabel(r"$x$")
	plt.ylabel(r"$f(x)$")
	plt.title("Activation functions")
	plt.ylim(-1.2, 1.2)
	plt.legend()
	plt.grid()
	plt.show()
	
	
	
