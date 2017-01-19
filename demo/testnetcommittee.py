
import numpy as np
import tenbilac
import pylab as plt

import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(10)

net1 = tenbilac.net.Net(ni=1, nhs=[3])
net2 = tenbilac.net.Net(ni=1, nhs=[3])
net3 = tenbilac.net.Net(ni=1, nhs=[3])

members = [net1, net2, net3]

comm = tenbilac.committee.Committee(members)

n = 200 # Size of training set
noise_scale = 0.05
ntest = 100 # Number of points to draw the regression line

def f(x):
	return np.sin(x)/(2. + x)

params = np.linspace(0.0, 10.0, n).reshape((1, n))
obs = f(params) + noise_scale*np.random.randn(n).reshape((1, n))

# Tenbilac only accept 3D input arrays
params = np.array([params])

# Normer object for the inputs and targets
obs_normer = tenbilac.data.Normer(obs)
params_normer = tenbilac.data.Normer(params)

# Do the normalisation
normobs = obs_normer(obs)
normparams = params_normer(params)

# Create some points where we want to see the value of the fitted obs, given the data.
testparams = np.linspace(-1.0, 11, ntest).reshape((1, ntest))
normtestparams = params_normer(testparams)

# Prepare a traindata object 
dat = tenbilac.data.Traindata(inputs=normparams, targets=normobs, valfrac=0.2)

# Preparing the training: 
#commtraining = comm.Training(dat=dat, errfctname="mse")
ctraining = tenbilac.committee.CommTraining(comm, dat=dat, errfctname="mse")

# The method `call` allows to call a function on each of the membre of the class
# In the simplereg.py example this would be training.net.addnoise()
# For some reason, we cannot call this with more than 1 cpu,
# otherwise, the callback fct in training is not called!!
ctraining.call(attr='net', method='addnoise', wscale=0.3, bscale=0.3)

# In the simplereg.py example this would be training.minibatch_bfgs()
ctraining.call(method='opt', call_ncpu=3, maxiter=500, algo="bfgs")#, mbsize=50, )

# Predicting the testparams, notice the exact same call method.
normtestpreds = comm.call(method='run', inputs=normtestparams)
testpreds = obs_normer.denorm(normtestpreds)

# We go back from a 3D array to 2D for plottting
params = params[0]

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.plot(params.T, obs.T, "b.", label="obs")
for ii in range(len(comm.members)): 
	ax.plot(testparams.T, testpreds[ii].T, label='%i: %s' % (ii, comm.members[ii]))

ax.plot(testparams.T, np.mean(testpreds.T, axis=2), lw=2, label='mean fit')
ax.plot(testparams.T, f(testparams.T), "k--", lw=2, label="truth")
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y = f(x) + \mathrm{noise}$", fontsize=18)
ax.set_ylim([-0.5, 0.5])
ax.legend(loc='best')

plt.tight_layout()

plt.show()	

print 'test completed.'
