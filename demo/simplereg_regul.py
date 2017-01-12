"""
This demo tests the network by asking for a plain regular regression (not inverse),
in 1D.

"""

import numpy as np
import tenbilac
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

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

# Generate the network
net = tenbilac.net.Net(1, [3])
training = tenbilac.train.Training(net, dat, errfctname="mse", regulweight=0.01, regulfctname="l2")

# Adding some (gaussian) noise to the weights and bias to initialise
training.net.addnoise(wscale=0.3, bscale=0.3)
# We train this normal (non-inverse) regression with params as inputs, and observations as output:
training.opt(algo="bfgs", mbsize=50, maxiter=500)

# Predicting the testparams
normtestpreds = net.run(normtestparams)
testpreds = obs_normer.denorm(normtestpreds)

print training

# We go back from a 3D array to 2D for plottting
params = params[0]

fig = plt.figure(figsize=(6, 4))

ax = fig.add_subplot(1, 1, 1)
ax.plot(params.T, obs.T, "b.", label="obs")
ax.plot(testparams.T, testpreds.T, "r-", label="fit")
ax.plot(testparams.T, f(testparams.T), "k--", lw=2, label="truth")
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y = f(x) + \mathrm{noise}$", fontsize=18)
ax.set_ylim([-0.5, 0.5])
ax.legend(loc='best')

plt.tight_layout()

tenbilac.plot.netviz(net)
plt.show()	

