
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


(obs_normer, params_normer, normobs, normparams, normuniparams, normuniobs, normtestobs) = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.wnet.WNet(1, [7])
net.setini()
net.addnoise(wscale=0.01, bscale=0.01)

traindata = tenbilac.data.Traindata(normobs, normparams, valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, traindata, errfctname="msbw")

#train.minibatch_bfgs(mbsize=250, mbloops=1, maxiter=200)
train.minibatch_bfgs(mbsize=50, mbloops=5, maxiter=20)


#train.save("train_msbw.pkl", keepdata=True)
#train = tenbilac.utils.readpickle("train_msbw.pkl")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(3, 1, 1)

tenbilac.plot.errevo(ax, train, showtimes=True)

ax = plt.subplot(3, 1, 2)
tenbilac.plot.paramsevo(ax, train, wnetpart="o")

ax = plt.subplot(3, 1, 3)
tenbilac.plot.paramsevo(ax, train, wnetpart="w")


plt.tight_layout()
	
plt.show()
