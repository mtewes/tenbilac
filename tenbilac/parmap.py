"""
http://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class
"""

import multiprocessing
from multiprocessing import Process, Pipe, Queue
from itertools import izip  


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


if __name__ == '__main__':
	print(parmap(lambda i: i * 2, [1, 2, 3, 4, 6, 7, 8], 3))


