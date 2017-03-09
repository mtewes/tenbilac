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
import shutil
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
	
	
	def __init__(self, configpath, configlist=None):
		"""Constructor, does not take a ton of arguments, just a path to a config file.
		Alternatively, the configpath can also just point to a directory (typically an existing tenbilac workdir),
		and the alphabetically *last* .cfg file in this directory will be used, AND this configpath will also be used as workdir,
		no matter what the config file says.
		
		:param configlist: An optional list of settings that will have precedence over what is
			written in the configfile. The structure is a list of tuples containing 3 strings (Section, Option, Value), such as
			[("setup", "workdir", "bla")]
		"""
		
		self.configpath = configpath
		self.config = SafeConfigParser(allow_no_value=True)
		
		
		if os.path.isfile(configpath): # Then we just read it
			logger.info("Reading in config from {}...".format(configpath))
			self.config.read(configpath)
		
		elif os.path.isdir(configpath): # We read the alphabetically last config file in this directory
			cfgfilepaths = sorted(glob.glob(os.path.join(configpath, "*.cfg")))
			if len(cfgfilepaths) == 0:
				raise RuntimeError("No config file found in {}!".format(self.configpath))
			
			cfgfilepath = cfgfilepaths[-1]
			logger.info("Found {} config files in dir, reading in {}...".format(len(cfgfilepaths), cfgfilepath))
			self.config.read(cfgfilepath)
			self.workdir = configpath
		
		else:
			raise RuntimeError("File or dir {} does not exist!".format(self.configpath))
				
		if configlist:
			logger.info("Using additional options: {}".format(configlist))
			for param in configlist:
				self.config.set(param[0], param[1], param[2])
				
		# If the name is not specified in the config, we use:
		# - the filename of the config file if configpath is a file
		# - the directory name, if configpath is a directory
		self.name = self.config.get("setup", "name") 
		if self.name is None or len(self.name.strip()) == 0: # if the ":" is missing as well, confirparser reads None
			# Then we use the filename
			self.name = os.path.splitext(os.path.basename(configpath))[0]

		# The workdir to be used
		if os.path.isdir(configpath):
			logger.info("Setting workdir to configpath")
			self.workdir = configpath
		else:
			self.workdir = self.config.get("setup", "workdir")
		
		# For easy access, we point to a few configuration items:
		self.inputnormerpath = os.path.join(self.workdir, "input_normer.pkl")
		self.targetnormerpath = os.path.join(self.workdir, "target_normer.pkl")
			
		logger.info("Constructed Tenbilac '{self.name}' with workdir '{self.workdir}'.".format(self=self))
		
		# Now we take care of the logging, in a brutal way. The methods will redicrec all tenbilac log to a file.
		# The code below only sets the logging up. It has to be activated and deactivated inside the methods.
		# Did not yet find the optimal way of doing this (decorator as class method ?)
		if self.config.getboolean("setup", "logtofile"):
			self.logger = logging.getLogger("tenbilac")
			logpath = os.path.join(self.workdir, "log.txt")
			logger.info("Tenbilac is set to log to {}".format(logpath))
			self.logfilehandler = logging.FileHandler(logpath, delay=True)
			self.logfilehandler.setLevel(logging.DEBUG)
			self.logfilehandler.setFormatter(logging.Formatter("PID %(process)d: %(levelname)s: %(name)s(%(funcName)s): %(message)s"))
			
			# Even if the file is only openend at the first message (delay=True in FileHandler above),
			# we do have to make sure the directories exist, so that logging can be done at any time.
			if not os.path.exists(self.workdir):
				os.makedirs(self.workdir)

	def _activatefilelog(self):
		"""
		Activate the logging of all tenbilac to a file.
		Note that only the method self.trian uses this so far.
		"""
		if self.config.getboolean("setup", "logtofile"):
			self.logger.addHandler(self.logfilehandler)
			self.logger.propagate = False
	
	def _deactivatefilelog(self):
		"""
		The problem: if a child method would call this, the logging-to-file would stop also for the parent calling the child.
		"""
		if self.config.getboolean("setup", "logtofile"):
			self.logger.removeHandler(self.logfilehandler)
			self.logger.propagate = True

	def __str__(self):
		return "Tenbilac '{self.name}'".format(self=self)
			

	def train(self, inputs, targets, inputnames=None, targetnames=None, auxinputs=None):
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
		
		# Can be used to save files at the level of this wrapper.
		startdt = datetime.datetime.now()
		
		self._activatefilelog()
		
		# For this wrapper, we only allow 3D inputs and 2D targets.
		if (inputs.ndim) != 3 or (targets.ndim) != 2:
			raise ValueError("This wrapper only accepts 3D inputs and 2D targets, you have {} and {}".format(inputs.shape, targets.shape))
		
		# Just for logging
		if os.path.exists(self.workdir):
			logger.info("The workdir already exists.")
		else:
			logger.info("The workdir does not exist, it will be created.")
		
		
		# Creating the normers and norming, if desired
		writeinputnormer = False # Should the new normers be written to disk, later?
		writetargetnormer = False
		if self.config.getboolean("norm", "oninputs"):
			logger.info("{}: normalizing training inputs...".format((str(self))))
			if self.config.getboolean("norm", "takeover") and os.path.exists(self.inputnormerpath):
				self.input_normer = utils.readpickle(self.inputnormerpath)
			else: # We make a new one:
				writeinputnormer = True
				self.input_normer = data.Normer(inputs, type=self.config.get("norm", "inputtype"))
			# And we norm the data:
			inputs = self.input_normer(inputs)
		else:
			logger.info("{}: inputs do NOT get normed.".format((str(self))))
		
		if self.config.getboolean("norm", "ontargets"):
			logger.info("{}: normalizing training targets...".format((str(self))))
			if self.config.getboolean("norm", "takeover") and os.path.exists(self.targetnormerpath):
				self.target_normer = utils.readpickle(self.targetnormerpath)
			else:
				writetargetnormer = True
				self.target_normer = data.Normer(targets, type=self.config.get("norm", "targettype"))
			targets = self.target_normer(targets)
		else:
			logger.info("{}: targets do NOT get normed.".format((str(self))))
		
		
		# And grouping them into a Traindata object:
		self.traindata = data.Traindata(
			inputs, targets, auxinputs=auxinputs,
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
					name='{}_{}'.format(self.name, i)
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
					name='{}_{}'.format(self.name, i)
					)
			else:
				raise RuntimeError("Don't know network type '{}'".format(nettype))
			
			
			# A directory where the training can store its stuff
			trainobjdir = os.path.join(self.workdir, "member_{}".format(i))
			trainobjpath = os.path.join(trainobjdir, "Training.pkl")
			plotdirpath = os.path.join(trainobjdir, "plots")
			
			# Let's see if an existing training is available in these directories:
			oldtrainobj = None
			if self.config.getboolean("train", "takeover") and os.path.exists(trainobjpath):
				# Then we read the existing training (before the new training has any chance to write files...)
				# (The acutal takeover will happen later).
				logger.info("Reading in existing training... ")
				oldtrainobj = utils.readpickle(trainobjpath)
			
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
				logpath=None, # Does not work well with multi-cpu, unfortunately.
				trackbiases=self.config.getboolean("train", "trackbiases"),
				verbose=self.config.getboolean("train", "verbose"),
				name='{}_{}'.format(self.name, i)
				)
			
			# We keep the directories at hand with this object
			trainobj.dirstomake=[trainobjdir, plotdirpath]
			
			# If desired, we take over a previous training.
			# If we don't take over, we have to start from identity and add noise
			
			if oldtrainobj is None:
			
				if self.config.getboolean("net", "startidentity"):
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
			
			else: # We do take over
				logger.info("Taking over an existing training...")
				trainobj.takeover(oldtrainobj)

				
			
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

		# Writing the config into the workdir, with a timestamp in the filename
		if self.config.getboolean("setup", "copyconfig"):
			configcopyname = self.name + "_" + datetimestr(startdt) + ".cfg"
			configcopypath = os.path.join(self.workdir, configcopyname)
			with open(configcopypath + "_running", 'wb') as configfile: # For now, we add this "_running". Will be removed when done.
				self.config.write(configfile)
			# No, we don't copy the file as (1) it might already have changed and (2) some options might have been passed as configlist.
			#shutil.copy(self.configpath, configcopypath + "_running")


		# Saving the normers, now that we have the directories
		if writeinputnormer:
			utils.writepickle(self.input_normer, self.inputnormerpath)
		if writetargetnormer:
			utils.writepickle(self.target_normer, self.targetnormerpath)
	
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
		
		if self.config.getboolean("setup", "copyconfig"):
			# We remove the "_running"
			if os.path.exists(configcopypath + "_running"):
				os.rename(configcopypath + "_running", configcopypath) # We take care reusign the copy, and not copying again what might already have changed...
		
		if self.config.getboolean("setup", "minimize"):
			self.minimize(dt=startdt)
					
		# We close with a summary of the results
		self.summary()
	
		self._deactivatefilelog()

	def _readmembers(self):
		"""
		A method that finds available committee members by itself, exploring the file system, and reads them in.
		"""	
		
		trainpaths = sorted(glob.glob(os.path.join(self.workdir, "member_*/Training.pkl")))
		logger.info("Found {} committee members to read in...".format(len(trainpaths)))
		self.committee = [utils.readpickle(trainpath) for trainpath in trainpaths]
		return trainpaths # Potentially useful
	
	
	def summary(self):
		"""
		Summarizes the training performance of committee members.
		"""
		
		self._readmembers()
		
		# First we just write some log info, to demonstrate the idea:
		trainlistsummary(self.committee)
		
		# And we create a plot
		if self.config.getboolean("train", "autoplot"):
			plotsdirpath = os.path.join(self.workdir, "plots")
			if not os.path.exists(plotsdirpath):
				os.makedirs(plotsdirpath)
			
			plot.summaryerrevo(self.committee, filepath=os.path.join(plotsdirpath, "summaryerrevo.png"))
		

	
	def minimize(self, destdir=None, dt=None):
		"""
		Prepares a self-sufficient "copy" of the current workdir containing all the information needed for predictions, but
		without all the (potentially large) logs, plots, etc.
		
		:params destdir: path to where to save the results
		:params dt: if destdir is not given, datetime object to use to name the destdir
		
		"""
		
		if dt is None:
			dt = datetime.datetime.now()
		if destdir is None:
			destdir = os.path.join(self.workdir, "mini_" + self.name + "_" + datetimestr(dt))
		
		logger.info("Minimizing into '{}'".format(destdir))
		memberfilepaths = self._readmembers() # Also fills self.committee!
		
		if os.path.exists(destdir):
			logger.warning("Destdir for minimization already exists, I will likely overwrite stuff!")
		else:
			os.makedirs(destdir)
		
		for (origfilepath, trainobj) in zip(memberfilepaths, self.committee):
			memberdestdirname = os.path.split(os.path.split(origfilepath)[0])[-1] # this is e.g. "member_010"
			memberdestdirpath = os.path.join(destdir, memberdestdirname)
			if not os.path.exists(memberdestdirpath):
				os.makedirs(memberdestdirpath)
			memberdestfilepath = os.path.join(memberdestdirpath, "Training.pkl")
			
			# And we make a tiny new training object, containing just the net
			net = trainobj.net
			net.resetcache()
			tinytrainobj = train.Training(net, None, name=trainobj.name)
			
			# To be able to select the best members later, we copy only the required stuff:
			tinytrainobj.optiterrs_val.append(trainobj.optiterrs_val[-1])
			tinytrainobj.optiterrs_train.append(trainobj.optiterrs_train[-1])
			tinytrainobj.optit = trainobj.optit
			
			# And save it
			tinytrainobj.save(memberdestfilepath)
			
		# We also copy the normers
		if os.path.exists(self.inputnormerpath):
			logger.info("Copying input normer...")
			shutil.copy(self.inputnormerpath, os.path.join(destdir, "input_normer.pkl"))
		if os.path.exists(self.targetnormerpath):
			logger.info("Copying target normer...")
			shutil.copy(self.targetnormerpath, os.path.join(destdir, "target_normer.pkl"))
		
		# And we write the config.
		configcopypath = os.path.join(destdir, self.name + ".cfg")
		with open(configcopypath, 'wb') as configfile:
			self.config.write(configfile)
		
		logger.info("Done with minimizing.")


	def predict(self, inputs):
		"""
		Checks the workdir and uses each network found or specified in config to predict some output.
		Does note care about the "Tenbilac" object used for the training!
		"""
		
		self._readmembers()
		
		
		if self.config.getboolean("predict", "selbest"):
			
			
			if self.config.get("predict", "selkind") == "bestn":
				thr = self.config.getint("predict", "thr")
				logger.info("Selecting {} best members among {}.".format(thr, len(self.committee)))
			elif self.config.get("predict", "selkind") == "sigmaclip":
				thr = self.config.getfloat("predict", "thr")
				logger.info("Sigma clipping to select the members. Selecting under {}sigma among {}.".format(thr, len(self.committee)))
			else:
				raise ValueError("Unknown member selection procedure")
			
			# We build a function to identify the best members
			bestkey = self.config.get("predict", "bestkey")
			if bestkey == "valerr":
				key = lambda trainobj: trainobj.optiterrs_val[-1]
			elif bestkey == "trainerr":
				key = lambda trainobj: trainobj.optiterrs_train[-1]
			elif bestkey == "nit":
				key = lambda trainobj: -1.0 * trainobj.optit # times minus one get the order right
			elif bestkey == "random":
				key = np.arange(len(self.committee))
				np.random.shuffle(key)
				for m, rid in zip(self.committee, key):
					m._rid = rid
				key = lambda trainobj: trainobj._rid
			else:
				raise ValueError("Unknown bestkey")
			logger.info("All potential committee members:")
			trainlistsummary(self.committee)
			
			if self.config.get("predict", "selkind") == "bestn":
				self.committee = sorted(self.committee, key=key)[0:thr]
			elif self.config.get("predict", "selkind") == "sigmaclip":
				keys = [key(mem) for mem in self.committee]
				_, ikeys = utils.sigma_clip_plus(keys, thr, get_indices=True)
				self.committee = [self.committee[i] for i in ikeys]
				logger.info("Sigma clipping to select the members. Selected {} members under {}sigma.".format(len(self.committee), thr))
			
		
			logger.info("Retained committee members:")
			trainlistsummary(self.committee)
				
		else:	
			logger.info("Keeping all members")
		
		
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
		predsarray = np.ma.array(predslist)
		logger.info("Prediction shape is {}".format(predsarray.shape))
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
		


def trainlistsummary(trainlist):
	"""
	Logs a summary table describing a list of Training objects
	"""
	for trainobj in trainlist:
		nit = trainobj.optit
		#assert nit == len(trainobj.optiterrs_train)   # No, we don't need this, and it doesn't hold for minimized Trainings
		#assert nit == len(trainobj.optiterrs_val)
		trainerr = trainobj.optiterrs_train[-1]
		valerr = trainobj.optiterrs_val[-1]
		valerrratio = valerr / trainerr
		logger.info("{:>20}: {:5d} iterations, train = {:.6e}, val = {:.6e} ({:4.1f})".format(trainobj.name, nit, trainerr, valerr, valerrratio))



def datetimestr(dt):
	"""
	Returns a string that can be used in filenames
	"""
	nomicrodt = dt.replace(microsecond=0) # Makes format simpler
	return nomicrodt.isoformat().replace(":", "-")
	
	

def _trainworker(trainobj):
	"""
	Function that is mapped to a list of Training objects to perform the actual training on several CPUs.
	"""
	starttime = datetime.datetime.now()
	p = multiprocessing.current_process()
	logger.info("{} is starting to train with PID {} and kwargs {}".format(p.name, p.pid, trainobj._trainkwargdict))
	np.random.seed() # So that every worker uses different random numbers when selecting minibatches etc
	# Of course we DONT feed any static number to the seed, as we want it to be different every time!
	trainobj.opt(**trainobj._trainkwargdict)
	endtime = datetime.datetime.now()
	logger.info("{} is done, it took {}".format(p.name, str(endtime - starttime)))
	



	
