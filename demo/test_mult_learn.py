"""
Training z = x * y

But single realziation regression, not inverse regression.

"""

import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


nrea = 1
ncas = 1000

# We prepare the data:

# Range of x and y
xs = np.random.uniform(0.1, 1, ncas)
ys = np.random.uniform(0.1, 1, ncas)

zs = xs * ys 

inputs = np.array([xs, ys])
inputs = np.tile(inputs, (nrea, 1, 1))

#inputs += 0.1*np.random.randn(inputs.size).reshape(inputs.shape)

# This is 3D (rea, features=2, case)
#print inputs
#print inputs.shape


targets = np.array([zs])

# This is 2D (feature=1, case)
#print targets
#print targets.shape


#inputnormer = tenbilac.data.Normer(inputs, type="01")
#inputs = inputnormer(inputs) + 0.01

dat = tenbilac.data.Traindata(inputs=inputs, targets=targets)

net = tenbilac.net.Net(ni=2, nhs=[-1], no=1, actfctname="iden", oactfctname="iden", inames=["x", "y"], onames=["z"])
#net = tenbilac.net.Net(ni=2, nhs=[5, 5], inames=["x", "y"], onames=["z"])

#print net.report()
net.addnoise()

training = tenbilac.train.Training(net, dat, errfctname="mse", autoplot=False, autoplotdirpath="plots")

training.bfgs(maxiter=1000, gtol=1e-20)

print net.report()

outs = net.predict(inputs)


assert nrea==1

zerrs = (outs[0] - targets)[0]

import matplotlib.pyplot as plt

plt.scatter(xs, ys, c=zerrs, s=80, marker="o", edgecolors="face")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()


