"""
The entry point to Tenbilac: functions defined here take care of running committees
(i.e., ensembles of networks) and communicating with config files. Hence the name com,
for communication, committees, common.

Its mission is to replace the complicated tenbilacwrapper of MegaLUT.
"""

from configparser import ConfigParser

import logging
logger = logging.getLogger(__name__)

from . import train
from . import utils



class Tenbilac():
	
	def __init__(self, configpath):
		"""Constructor, does not take a ton of arguments, just a path to a config file.
		"""
		
		self.configpath = configpath
		self.config = ConfigParser()
		self._readconfig(configpath)
		
		
	
	def _readconfig(self, configpath):
		"""
		"""
		logger.info("Reading in config from {}".format(configpath))
		self.config.read(configpath)
	

# 	def _writeconfig(self, configpath):
# 		"""
# 		"""
# 		logger.info("Writing config")


# 	def setup(self, inputs=None, targets=None):
# 		"""
# 		
# 		"""
# 		logger.info("Setting up Tenbilac {}".format(self.config.get("setup", "name")))
# 
# 
# 		#mwlist = eval(self.config.get("setup", "mwlist"))
# 		#print mwlist
		
		

	def train(self, inputs=None, targets=None):
		"""
		Make and save normers if they don't exist
		Norm data with new or existing normers
		Prepares training objects. If some exist, takesover their states
		Runs all thoses trainings with multiprocessing
		Analyses and compares the results obtained by the different members
		"""
		
		#self._preptrainargs(inputs, targets)
		self._makenorm(inputs, targets)
		#(inputs, targets) = self._norm(inputs, targets)
		

	def predict(self, inputs):
		"""
		Checks the workdir and uses each network found or specified in config to predict some output
		"""



	def _checktrainargs(self, inputs, targets):
		"""
		"""
		assert inputs.ndim == 3
		assert targets.ndim == 2
		
	

	def _makenorm(self, inputs, targets):
		"""
		"""
				# We normalize the inputs and labels, and save the Normers for later denormalizing.
		logger.info("{0}: normalizing training inputs...".format((str(self))))
		self.input_normer = tenbilac.data.Normer(inputs, type=self.params.normtype)
		norminputs = self.input_normer(inputs)
		
		if self.params.normtargets:
			logger.info("{0}: normalizing training targets...".format((str(self))))
			self.target_normer = tenbilac.data.Normer(targets, type=self.params.normtype)
			normtargets = self.target_normer(targets)
		else:
			logger.info("{0}: I do NOT normalize targets...".format((str(self))))
			self.target_normer = None
			normtargets = targets	



	def _norm(self, inputs, targets):
		"""Takes care of the norming
		"""


	def _summary(self):
		
		"""Analyses how the trainings of a committee went. Logs info and calls a checkplot.
		"""


