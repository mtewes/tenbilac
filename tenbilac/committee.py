"""
This is Tenbilac!
This is a net (Wnet in the future) ensemble class.
"""

import numpy as np
from multiprocessing import Process, Pipe, Queue
from itertools import izip  

import logging
logger = logging.getLogger(__name__)

from . import train


def spawn(f):  
	def fun(pipe,x):  
		pipe.send(f(x))  
		pipe.close()  
	return fun  

def parmap(f, X, ncpu):  
	"""
	This is an alternative to multiprocessing.Pool to avoid the limitations of the package (pickling stuff...)
	
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
			# If multiple_trainings then it means that all args are a list of len = 3:
			lenkw = None
			for kw in kwargs:
				if len(committee.members) != len(kwargs[kw]): 
					raise ValueError("All kwargs must be of the same size as the committee!")

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
			
def _worker(params):
	training, method, attr, kwargs = params

	if attr is None:
		cmd_line = "training.%s(**kwargs)" % (method)
	else:
		cmd_line = "training.%s.%s(**kwargs)" % (attr, method)
	output = eval(cmd_line)

	return training, output
