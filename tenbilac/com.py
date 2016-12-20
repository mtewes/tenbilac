"""
The entry point to Tenbilac: functions defined here take care of running committees
(i.e., ensembles of networks) and communicating with config files. Hence the name "com",
for communication, committees, common.

Its mission is to replace the messy tenbilacwrapper of MegaLUT.

Tenbilac is a class, but is NOT designed to be "kept" (in a pickle) between setup, training and predicting.
All the info is in the config files, there are NO secret instance attributes worth of keeping.

"""

from ConfigParser import SafeConfigParser
import multiprocessing
import os
import glob
import datetime
import numpy as np

import logging
logger = logging.getLogger(__name__)

from . import train
from . import utils
from . import data
from . import net
from . import multnet
from . import train
from . import opt
from . import parmap
from . import plot


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
			

	def train(self, inputs, targets, inputnames=None, targetnames=None):
		"""
		Make and save normers
		Norm data with new normers
		Prepares training objects. TODO: If some exist, takesover their states
		Runs all thoses trainings with multiprocessing
		Analyses and compares the results obtained by the different members
		
		
		About multiprocessing: 
		One could split this task as early as possible and do all the construction of the Training and Net objects already in a pool.
		However, this preparation is very fast, and so for ease of debugging we keep this in normal loop.
		Also, we avoid duplicating the input data, which might be MASSIVE. We also avoid writing this input data to disk, and having
		the independent processes reading it again.
		
		"""
				
		# For this wrapper, we only allow 3D inputs and 2D targets.
		if (inputs.ndim) != 3 or (targets.ndim) != 2:
			raise ValueError("This wrapper only accepts 3D inputs and 2D targets, you have {} and {}".format(inputs.shape, targets.shape))
				
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
				netobj = net.Net(
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
				netobj = multnet.MultNet(
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
			trainobjdir = os.path.join(self.workdir, "{}_{:03d}".format(self.name, i))
			trainobjpath = os.path.join(trainobjdir, "Training.pkl")
			plotdirpath = os.path.join(trainobjdir, "plots")
			
			
			# Now we create the Training object, with the new network and the traindata
			
			if self.config.getboolean("train", "useregul"):
				raise RuntimeError("Regul wrapper not implemented, code has to be cleaned first.")
			
			trainobj = train.Training(	
				netobj,
				self.traindata,
				errfctname=self.config.get("train", "errfctname"),
				regulweight=None,
				regulfctname=None,
				itersavepath=trainobjpath,
				saveeachit=self.config.getboolean("train", "saveeachit"),
				autoplotdirpath=plotdirpath,
				autoplot=self.config.getboolean("train", "autoplot"),
				trackbiases=self.config.getboolean("train", "trackbiases"),
				verbose=self.config.getboolean("train", "verbose"),
				name='Train-{}-{}'.format(self.name, i)
				)
			
			# We keep the directories at hand with this object
			trainobj.dirstomake=[trainobjdir, plotdirpath]
			
			# Somewhere here we would maybe take over an existing training, if desired
			# If we don't take over, we have to start from identity and add noise
			
			if self.config.get("net", "startidentity"):
				trainobj.net.setidentity(
					onlyn=self.config.getint("net", "onlynidentity")
					)
				if nettype == "MultNet": # We have to call multini again!
					trainobj.net.multini()
					
			trainobj.net.addnoise(
				wscale=self.config.getfloat("net", "ininoisewscale"),
				bscale=self.config.getfloat("net", "ininoisebscale"),
				multwscale=self.config.getfloat("net", "ininoisemultwscale"),
				multbscale=self.config.getfloat("net", "ininoisemultbscale")
				)
			
			# We add this new Training object to the committee
			self.committee.append(trainobj)
			
		assert len(self.committee) == nmembers
		
		
		# Creating the directories
		if not os.path.isdir(self.workdir):
			os.makedirs(self.workdir)
		for trainobj in self.committee:
			for dirpath in trainobj.dirstomake:
				if not os.path.isdir(dirpath):
					os.makedirs(dirpath)

		# Saving the normers, now that we have the directories
		if self.config.getboolean("norm", "oninputs"):
			utils.writepickle(self.input_normer, os.path.join(self.workdir, "input_normer.pkl"))
		if self.config.getboolean("norm", "ontargets"):
			utils.writepickle(self.target_normer, os.path.join(self.workdir, "target_normer.pkl"))
	
		# Preparing the training configuration. So far, we train the committee with identical params.
		# In future, we could do this differently.
		
		# Get the name of the algorithm to use:
		algo = self.config.get("train", "algo")
		# Now we get the associated section as a dict, and "map" eval() on this dict to get ints as ints, floats as floats, bools as bools...
		trainkwargdict = {k: eval(v) for (k, v) in self.config.items("algo_" + algo)}
		
		
		# We add some further training parameters to this dict:
		trainkwargdict["algo"] = algo
		trainkwargdict["mbsize"] = None
		trainkwargdict["mbfrac"] = self.config.getfloat("train", "mbfrac")
		trainkwargdict["mbloops"] = self.config.getint("train", "mbloops")
		
		
		# And attach this dict to the committee members, to keep the map() very simple.
		for trainobj in self.committee:
			trainobj._trainkwargdict = trainkwargdict
		
		
		# We are ready to start the training (optimization)
		
		ncpu = self.config.getint("train", "ncpu")
		
		if ncpu == 0:
			try:
				ncpu = multiprocessing.cpu_count()
			except:
				logger.warning("multiprocessing.cpu_count() is not implemented!")
				ncpu = 1
			
		logger.info("{}: Starting the training on {} CPUs".format(str(self), ncpu))
		
		if ncpu == 1:
			# The single-processing version (not using multiprocessing to keep it easier to debug):
			logger.debug("Not using multiprocessing")
			map(_trainworker, self.committee)

		else:
			# multiprocessing map: # Does not work
			#pool = multiprocessing.Pool(processes=ncpu)
			#pool.map(_trainworker, self.committee)
			#pool.close()
			#pool.join()
			
			parmap.parmap(_trainworker, self.committee, ncpu)
		
		logger.info("{}: done with the training".format(str(self)))
		# We close with a summary of the results
		self.summary()
	
	
	def _readmembers(self):
		"""
		A method that finds available committee members by itself, exploring the file system, and reads them in.
		"""	
		
		trainpaths = sorted(glob.glob(os.path.join(self.workdir, "*/Training.pkl")))
		logger.info("Found {} committee members to read in...".format(len(trainpaths)))
		self.committee = [utils.readpickle(trainpath) for trainpath in trainpaths]
		
		
	
	def summary(self):
		"""
		Summarizes the training performance of committee members.
		"""
		self._readmembers()
		
		# First we just write some log info, to demonstrate the idea:
		for trainobj in self.committee:
			nit = trainobj.optit
			assert nit == len(trainobj.optiterrs_train)
			assert nit == len(trainobj.optiterrs_val)
			trainerr = trainobj.optiterrs_train[-1]
			valerr = trainobj.optiterrs_val[-1]
			valerrratio = valerr / trainerr
			logger.info("{:>20}: {:5d} iterations, train = {:.6e}, val = {:.6e} ({:4.1f})".format(trainobj.name, nit, trainerr, valerr, valerrratio))
	
		# And we create a plot
		if self.config.getboolean("train", "autoplot"):
			plotsdirpath = os.path.join(self.workdir, "plots")
			if not os.path.exists(plotsdirpath):
				os.makedirs(plotsdirpath)
			
			#plot.summaryerrevo(self.committee)
			plot.summaryerrevo(self.committee, filepath=os.path.join(plotsdirpath, "summaryerrevo.png"))
	

	def predict(self, inputs):
		"""
		Checks the workdir and uses each network found or specified in config to predict some output.
		Does note care about the "Tenbilac" object used for the training!
		"""
		
		self._readmembers()
		
		
		# Here comes code to select members
		
		
		# We norm the inputs
		if self.config.getboolean("norm", "oninputs"):
			logger.info("{}: normalizing inputs...".format(str(self)))
			input_normer = utils.readpickle(os.path.join(self.workdir, "input_normer.pkl"))
			inputs = input_normer(inputs)
			
		else:
			logger.info("{}: inputs do NOT get normed.".format(str(self)))
		
		# Run the predictions by all the committee members
		logger.info("{}: Making predictions with {} members...".format(str(self), len(self.committee)))
		predslist = [trainobj.net.predict(inputs) for trainobj in self.committee]
		
		# We might have to denorm the predictions:
		if self.config.getboolean("norm", "ontargets"):
			logger.info("{}: denormalizing predictions...".format(str(self)))
			target_normer = utils.readpickle(os.path.join(self.workdir, "target_normer.pkl"))
			predslist = [target_normer.denorm(preds) for preds in predslist]
		else:
			logger.info("{}: predictions do NOT get denormed.".format(str(self)))
		
		# And average the results
		logger.info("{}: Building predictions array...".format(str(self)))
		predsarray = np.array(predslist)
		assert predsarray.shape[0] == len(self.committee)
		
		
		
		
		combine = self.config.get("predict", "combine")
		if combine == "mean":
			retarray = np.mean(predsarray, axis=0)
		elif combine == "median":
			retarray = np.median(predsarray, axis=0)
		else:
			raise ValueError("Unknown combine")
		
		logger.info("{}: Done with averaging.".format(str(self)))
		return retarray
		


def _trainworker(trainobj):
	"""
	Function that is mapped to a list of Training objects to perform the actual training on several CPUs.
	"""
	starttime = datetime.datetime.now()
	p = multiprocessing.current_process()
	logger.info("{} is starting to train with PID {} and kwargs {}".format(p.name, p.pid, trainobj._trainkwargdict))
	trainobj.opt(**trainobj._trainkwargdict)
	endtime = datetime.datetime.now()
	logger.info("{} is done, it took {}".format(p.name, str(endtime - starttime)))
	
	

