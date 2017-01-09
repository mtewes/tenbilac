"""
Custom and non-custom optimization algorithms to be used for training.
Functions take a train.Training object as first argument, perform their job, and return nothing.
They are almost methods of train.Training, but outsourced in this separate module.

"""

import numpy as np
import scipy.optimize


import logging
logger = logging.getLogger(__name__)



def bfgs(training, maxiter=100, gtol=1e-8, **kwargs):
	"""Calling scipy BFGS
	"""

	optres = scipy.optimize.fmin_bfgs(
			training.cost, training.params[training.paramslice],
			fprime=None,
			maxiter=maxiter, gtol=gtol,
			full_output=True, disp=True, retall=False, callback=training.callback, **kwargs)
		
	if len(optres) == 7:
		(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag) = optres
		training.cost(xopt) # To set the optimal parameters (even if this might not be needed, we don't want to depend on scipy internals...)
		logger.info("Done with optimization, {0} func_calls and {1} grad_calls".format(func_calls, grad_calls))
	else:
		logger.warning("Optimization output is fishy")


def multnetbfgs(training, nepochs=10, maxiter_sum=200, maxiter_mult=200, gtol=1e-8, **kwargs):
	"""A special version for MultNets, in development
	"""
	
	# We first start by optimizing only the sum layers, leaving the mult-layer as it is:
	if "maxiter" in kwargs:
		logger.warning("maxiter IN MULTNETBFGS IS FISHY. FOR NOW JUST REMOVING IT. IT WILL SOON BE RAISING AN ERROR")
		del kwargs["maxiter"]

	for epoch in range(nepochs):
		logger.info("Epoch {}/{} starting".format(epoch, nepochs))
		
		if maxiter_sum > 0:
			training.set_paramslice(mode="sum")
			bfgs(training, maxiter=maxiter_sum, gtol=gtol, **kwargs)
			training.end()
			logger.info("Optimisation sum layers done.")
		else:
			logger.info("Not optimising sum layers.")
	
		if maxiter_mult > 0:
			training.set_paramslice(mode="mult")
			bfgs(training, maxiter=maxiter_mult, gtol=gtol, **kwargs)
			training.end()
			logger.info("Optimisation mult layers done.")
		else:
			logger.info("Not optimising mult layers.")


def brute(training, maxiter=100, gtol=1e-6, **kwargs):
	"""Custom brute-force like optimization, for tests.
	"""
	logger.info("Starting brute optimization with options {}".format(kwargs))
	
	nparams = len(training.params)
	previouscost = None
	
	for iiter in range(maxiter):
		
		iniparams = training.params.copy()
		
		# We prepare a list for the params to be tested.
		paramslist = []
		costlist = []
		
		for iparam in range(nparams):
			for v in [-1., -0.51, -0.1, 0.1, 0.51, 1.0]:
				testparams = iniparams.copy()
				testparams[iparam] = v
				paramslist.append(testparams)
				cost = training.cost(testparams)
				costlist.append(cost)
				#print testparams, cost
			#for v in [-0.1, 0.1]:
			#	testparams = iniparams.copy()
			#	testparams[iparam] += v
			#	paramslist.append(testparams)
			#	cost = training.cost(testparams)
			#	costlist.append(cost)
			#	#print testparams, cost
				
		assert len(paramslist) == len(costlist)
		bestindex = np.argmin(costlist)
		bestparams = paramslist[bestindex].copy()
		bestcost = costlist[bestindex]
		
		training.cost(bestparams)
		training.callback(bestparams)	


		if previouscost == None:
			previouscost = bestcost
			continue # with next iteration
		
		if np.fabs((previouscost - bestcost) / bestcost) < gtol :
			#if np.allclose(bestparams, previousparams): # no, does not work because of zeros that make different params all have the same minimal cost.
			# Then nothing improved, we stop
			logger.info("Stopping this")
			break

		else:
			previouscost = bestcost

