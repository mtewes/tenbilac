"""
Training z = x * y with a single product unit, testing this mult neuron.

Single realziation, no noise in the data, not inverse regression.

"""

import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


nrea = 5
ncas = 100

# We prepare the data:

# Range of x and y
xs = np.random.uniform(-5, 5, ncas)
ys = np.random.uniform(-10, 30, ncas)

zs = xs * ys 

inputs = np.array([xs, ys])
inputs = np.tile(inputs, (nrea, 1, 1))
# We add noise:
inputs += 0.001*np.random.randn(inputs.size).reshape(inputs.shape)

# This is 3D (rea, features=2, case)
#print inputs
#print inputs.shape


targets = np.array([zs])

# This is 2D (feature=1, case)
#print targets
#print targets.shape


inputnormer = tenbilac.data.Normer(inputs, type="sa1")
inputs = inputnormer(inputs)
targetnormer = tenbilac.data.Normer(targets, type="sa1")
targets = targetnormer(targets)


dat = tenbilac.data.Traindata(inputs=inputs, targets=targets)

net = tenbilac.multnet.MultNet(ni=2, mwlist=[(1, 1)], nhs=[3], no=1, actfctname="iden", oactfctname="iden", multactfctname="iden", inames=["x", "y"], onames=["z"])

training = tenbilac.train.Training(net, dat, errfctname="msb", autoplot=False, autoplotdirpath=".")


net.setidentity()
net.multini()

# We can even add noise afterwards, as long as we keep the mult-scales at zero:
net.addnoise(multwscale=0.0, multbscale=0.0, wscale=0.1, bscale=0.1)

tenbilac.plot.netviz(training, title="Ready!")
#print net.report()

training.set_paramslice(mode="sum")
training.opt(algo="bfgs", mbsize=None, mbfrac=1.0, mbloops=1, maxiter=20, gtol=1e-8)

print net.report()

training.set_paramslice(mode="mult")

training.opt(algo="bfgs", mbsize=None, mbfrac=1.0, mbloops=1, maxiter=20, gtol=1e-8)

print net.report()
tenbilac.plot.netviz(training, title="Done!")


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

