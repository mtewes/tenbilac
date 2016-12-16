"""
The entry point to Tenbilac: functions defined here take care of running committees
(i.e., ensembles of networks) and communicating with config files. Hence the name "com",
for communication, committees, common.

Its mission is to replace the messy tenbilacwrapper of MegaLUT.

Tenbilac is a class, but is NOT designed to be "kept" (in a pickle) between setup, training and predicting.
All the info is in the config files, there are NO secret instance attributes worth of keeping.

"""

from configparser import SafeConfigParser
import os

import logging
logger = logging.getLogger(__name__)

from . import train
from . import utils
from . import data
from . import net
from . import multnet
from . import train


class Tenbilac():
	
	def __init__(self, configpath):
		"""Constructor, does not take a ton of arguments, just a path to a config file.
		"""
		
		self.configpath = configpath
		self.config = SafeConfigParser(allow_no_value=True)
		logger.info("Reading in config from {}".format(configpath))
		self.config.read(configpath)
		
		# For easy access, we point to a few configuration items:
		self.name = self.config.get("setup", "name")
		self.workdir = self.config.get("setup", "workdir")
	
	def __str__(self):
		return "Tenbilac '{self.name}'".format(self=self)
	
# 	def _readconfig(self, configpath):
# 		"""
# 		"""
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
		
		

	def train(self, inputs, targets, inputnames=None, targetnames=None):
		"""
		Make and save normers if they don't exist
		Norm data with new or existing normers
		Prepares training objects. If some exist, takesover their states
		Runs all thoses trainings with multiprocessing
		Analyses and compares the results obtained by the different members
		"""
		
		#self._preptrainargs(inputs, targets)
		#self._makenorm(inputs, targets)
		#(inputs, targets) = self._norm(inputs, targets)
		
		# For this wrapper, we only allow 3D inputs and 2D targets.
		if (inputs.ndim) != 3 or (targets.ndim) != 2:
			raise ValueError("This wrapper only accepts 3D inputs and 2D targets, you have {} and {}".format(inputs.shape, targets.shape))
		
		# Creating the workdir
		if not os.path.isdir(self.workdir):
			os.makedirs(self.workdir)
		
		# Creating the normers and norming, if desired
		if self.config.getboolean("norm", "oninputs"):
			logger.info("{}: normalizing training inputs...".format((str(self))))
			self.input_normer = data.Normer(inputs, type=self.config.get("norm", "inputtype"))
			inputs = self.input_normer(inputs)
		else:
			logger.info("{}: inputs do NOT get normed.".format((str(self))))
		
		if self.config.getboolean("norm", "ontargets"):
			logger.info("{}: normalizing training targets...".format((str(self))))
			self.target_normer = data.Normer(targets, type=self.config.get("norm", "targettype"))
			targets = self.target_normer(targets)
		else:
			logger.info("{}: targets do NOT get normed.".format((str(self))))
		
		
		# And grouping them into a Traindata object:
		self.traindata = data.Traindata(
			inputs, targets, auxinputs=None,
			valfrac=self.config.getfloat("train", "valfrac"),
			shuffle=self.config.getboolean("train", "shuffle")
			)
		
		# Setting up the network- and training-objects according to the config
		nmembers = self.config.getint("net", "nmembers")
		self.committee = [] # Will be a list of Training objects.
		
		ni = inputs.shape[1]
		no = targets.shape[0]
		nettype = self.config.get("net", "type")
		
		logger.info("{}: Building a committee of {}s with {} members...".format(str(self), nettype, nmembers))
		
		for i in range(nmembers):
		
			# We first create the network
			if nettype == "Net":
				newnet = net.Net(
					ni=ni,
					nhs=list(eval(self.config.get("net", "nhs"))),
					no=no,
					actfctname=self.config.get("net", "actfctname"),
					oactfctname=self.config.get("net", "oactfctname"),
					multactfctname=self.config.get("net", "multactfctname"),
					inames=inputnames,
					onames=targetnames,
					name='{}-{}'.format(self.name, i)
					)
			elif nettype == "MultNet":
				newnet = multnet.MultNet(
					ni=ni,
					nhs=list(eval(self.config.get("net", "nhs"))),
					mwlist=list(eval(self.config.get("net", "mwlist"))),
					no=no,
					actfctname=self.config.get("net", "actfctname"),
					oactfctname=self.config.get("net", "oactfctname"),
					multactfctname=self.config.get("net", "multactfctname"),
					inames=inputnames,
					onames=targetnames,
					name='{}-{}-{}'.format(nettype, self.name, i)
					)
			else:
				raise RuntimeError("Don't know network type '{}'".format(nettype))
			
			# A directory where the training can store its stuff
			newnetdir = os.path.join(self.workdir, "{}_{:03d}".format(self.name, i))
			newtrainingpath = os.path.join(newnetdir, "Training.pkl")
			newplotdirpath = os.path.join(newnetdir, "plots")
			
			
			# Now we create the Training object, with the new network and the traindata
			newtrain = train.Training(	
				newnet,
				self.traindata,
				errfctname=self.config.get("train", "errfctname"),
				regulweight=self.config.get("train", "regulweight"),
				regulfctname=self.config.get("train", "regulfctname"),
				itersavepath=newtrainingpath,
				saveeachit=self.config.getboolean("train", "saveeachit"),
				autoplotdirpath=newplotdirpath,
				autoplot=self.config.getboolean("train", "autoplot"),
				trackbiases=self.config.getboolean("train", "trackbiases"),
				verbose=self.config.getboolean("train", "verbose"),
				name='Train-{}-{}'.format(self.name, i)
				)
			
			
			
			self.committee.append(newtrain)
		assert len(self.committee) == nmembers
		
		
		

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
		


	def _norm(self, inputs, targets):
		"""Takes care of the norming
		"""


	def _summary(self):
		
		"""Analyses how the trainings of a committee went. Logs info and calls a checkplot.
		"""


