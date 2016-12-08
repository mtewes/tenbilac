"""
Similar to test_mult_learn, but this time as inverse regression.

Training z = x * y

x and y are noisy inputs.
"""

import numpy as np
import tenbilac
import os

import logging
logging.basicConfig(level=logging.INFO)


nrea = 5
ncas = 500

# We prepare the data:

# Range of x and y
xs = np.random.uniform(-3, 6, ncas)
ys = np.random.uniform(-10, -5, ncas)

zs = xs * ys 

inputs = np.array([xs, ys])
inputs = np.tile(inputs, (nrea, 1, 1))
# We add noise:
inputs += 0.001*np.random.randn(inputs.size).reshape(inputs.shape)

# This is 3D (rea, features=2, case)
#print inputs
print inputs.shape

targets = np.array([zs])

# This is 2D (feature=1, case)
#print targets
print targets.shape


# Now the normer:

inputnormer = tenbilac.data.Normer(inputs, type="sa1")
inputs = inputnormer(inputs)
targetnormer = tenbilac.data.Normer(targets, type="sa1")
targets = targetnormer(targets)

print inputnormer
print targetnormer


dat = tenbilac.data.Traindata(inputs=inputs, targets=targets)

net = tenbilac.net.Net(ni=2, nhs=[-1], no=1, actfctname="iden", oactfctname="iden",  multactfctname="iden", inames=["x", "y"], onames=["z"])

# The exact solution:
net.setidentity()
#net.layers[0].weights[0,1] = 1.0
#net.layers[1].weights[0] = ((inputnormer.b[0]*inputnormer.b[1])/targetnormer.b[0])

net.addnoise(multwscale=0.5, wscale=0.1, bscale=0.1)


print net.report()


training = tenbilac.train.Training(net, dat, errfctname="msb", autoplot=True, trackbiases=True, autoplotdirpath=".")


training.opt(algo="brute", mbsize=None, mbfrac=1.0, mbloops=1, maxiter=15, gtol=1e-8)
training.opt(algo="bfgs", mbsize=None, mbfrac=1.0, mbloops=1, maxiter=15, gtol=1e-8)

print net.report()


outs = targetnormer.denorm(net.predict(inputs))
#outs = net.predict(inputs)
# Shape is (rea, neuron, case)

assert zs.size == ncas
residues = outs[:,0,:] - zs # We have only one neuron
# Shape is (rea, case)

# Stats for each case, over the realizations:
meanres = np.mean(residues, axis=0)
stdres = np.std(residues, axis=0)
assert meanres.size == ncas


#training.save("test.pkl", keepdata=True)
#exit()
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(xs, ys, c=meanres, s=80, marker="o", edgecolors="face")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Mean residue")

plt.subplot(1, 2, 2)
plt.scatter(xs, ys, c=stdres, s=80, marker="o", edgecolors="face")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.title("Std residue")

plt.tight_layout()


plt.show()

