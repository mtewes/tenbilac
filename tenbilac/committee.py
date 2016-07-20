"""
This is Tenbilac!
This file provides a class holder to define a ensemble (a committee) of Nets and a decorator class
that allows to train the committee on the same or different data. Committees are a stand-alone plugin.
Tenbilac Networks are not aware of this extension.
"""

import numpy as np
from multiprocessing import Process, Pipe, Queue
from itertools import izip  

import logging
logger = logging.getLogger(__name__)

from . import train
from . import utils


def spawn(f): 
	"""
	Helper function for `parmap` which prepares the worker `f` to be spawned.
	""" 
	def fun(pipe,x):  
		pipe.send(f(x))  
		pipe.close()  
	return fun  

def parmap(f, X, ncpu):  
	"""
	This is an alternative to multiprocessing.Pool to avoid the limitations of the package (pickling stuff...)
	
	.. note:: see http://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
	
	.. note:: It is very possible that multiprocessing.Pool is fixed in python3 
	"""
	pipe=[Pipe() for x in X]  
	processes=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]  
	numProcesses = len(processes)  
	processNum = 0  
	outputList = []  
	while processNum < numProcesses:  
		endProcessNum = min(processNum+ncpu, numProcesses)  
		for proc in processes[processNum:endProcessNum]:  
			proc.start()  
		# It is crucial to call recv() before join() to avoid deadlocks !
		for proc,c in pipe[processNum:endProcessNum]:  
			outputList.append(proc.recv())  
		for proc in processes[processNum:endProcessNum]:  
			proc.join()  

		processNum = endProcessNum  
	return outputList	

class Committee():
	"""
	Holder class for committee members
	"""
	
	def __init__(self, members, name=None):
		"""
		:param members: Tenbilac instances
		:type members: tuple of net or wnet
		
		:param name: if None, will be set automatically
		:type name: string
		"""
		if name is None:
			self.name = 'Committee'
		else:
			self.name = name
		self.members = members
		
		logger.info("Built " + str(self))
		
		
	def __str__(self):
		"""
		A short string describing the network
		"""
		autotxt = "Committee class with {:d} members:\n".format(len(self.members))
		for ii, memi in enumerate(self.members):
			autotxt = "{autotxt}\t{ii}: {memi}\n".format(ii=ii+1, autotxt=autotxt, memi=memi)
		
		return "'{name}'\n{spacer}\n {autotxt}".format(name=self.name, spacer=(len(self.name)+2)*"=", autotxt=autotxt)
	
	def call(self, method, **kwargs):
		"""
		Method that runs a given `method` on all `trainings` instances.
		
		:param method: The name of the method
		:type method: string
		
		All other kwargs are passed to the method
		"""
		
		return np.asarray([eval("memi.%s(**kwargs)" % (method)) for memi in self.members])


class CommTraining():
	"""
	This is an implicit decorator of the training class to handle the training of committees
	"""
	
	def __init__(self, committee, multiple_trainings=False, **kwargs):

		if multiple_trainings:
			# If multiple_trainings then it means that all args are a list of same len:
			lenkw = None
			for kw in kwargs:
				if len(committee.members) != len(kwargs[kw]): 
					raise ValueError("All kwargs must be of the same size as the committee! {name}: {lenk}/{lenc}".format(name=kw,\
						 lenk=len(kwargs[kw]), lenc=len(committee.members)))

		self.committee = committee
		rslt = []
		for ii, memi in enumerate(self.committee.members):
			
			if multiple_trainings:
				ukwargs = {}
				for kw in kwargs:
					ukwargs[kw] = kwargs[kw][ii]
			else:
				ukwargs = kwargs
			logger.info("Setting up Training committee {i}/{f}".format(i=(ii+1), f=len(self.committee.members)))
			output = train.Training(memi, **ukwargs)
			rslt.append(output)
			
		self.trainings = rslt
		
	def call(self, method, call_ncpu=1, attr=None, **kwargs):
		"""
		Method that runs a given `method` on all `trainings` instances.
		
		:param method: The name of the method
		:type method: string
		
		:param call_ncpu: number of cpu to use
		
		:param attr: calls a method on an attribute of all `trainings` instances. If `None` simply
			ignores this argument
			
		All other kwargs are passed to the method
		
		.. warning:: There is a known issue there. You should not run on multiple core except when training
			otherwise the callback function in training is not called.
		"""

		params = [[memi, method, attr, kwargs] for memi in self.trainings]
		
		if call_ncpu == 1:
			rslt = map(_worker, params)
		else:
			# We must use this, because of the limitations of multiprocessing.Pool
			rslt = parmap(_worker, params, call_ncpu)
			
		outputs = []
		
		# This is necessary for the multiple cpu case 
		for ii, (rs, out) in enumerate(rslt):
			self.trainings[ii] = rs 
			self.committee.members[ii] = rs.net
			outputs.append(out)
			
		return outputs 
	
	def save(self, filepath, keepdata=False):
		"""
		Saves the training committee into a pkl file
		As the training data is so massive, by default we do not save it!
		Note that this might be done at each iteration!
		
		..note:: This is a similar method as in `train.py`
		"""

		if keepdata is True:
			logger.info("Writing training committee to disk and keeping the data...")
			utils.writepickle(self, filepath)	
		else:
			tmptraindata = [dat for dat in self.trainings]
			self.call('set_dat', dat=None)
				
			utils.writepickle(self, filepath)	
			
			for ii in range(len(self.trainings)):
				self.trainings[ii].dat = tmptraindata[ii]

def _worker(params):
	training, method, attr, kwargs = params

	if attr is None:
		cmd_line = "training.%s(**kwargs)" % (method)
	else:
		cmd_line = "training.%s.%s(**kwargs)" % (attr, method)
	output = eval(cmd_line)

	return training, output
