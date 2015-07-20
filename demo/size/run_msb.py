
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


(n, nrea, noise_scale, params, obs, obs_normer, params_normer, normobs, normparams, uninrea, uniparams, uniobs, ntest, testobs, normtestobs, normuniparams, normuniobs) = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.net.Tenbilac(1, [5])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)


data = tenbilac.data.Traindata(normobs, normparams, valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, data, errfctname="msrb")

train.minibatch_bfgs(mbsize=50, mbloops=5, maxiter=30)

train.save("train_msrb.pkl", keepdata=True)

#tenbilac.plot.paramscurve(train)

#exit()

"""
print train.currentcost()

train.bfgs(maxiter=maxiter)

print train.currentcost()
train.random_minibatch(size=minibatch_size)
print train.currentcost()


tenbilac.plot.paramscurve(train)

exit()


train.bfgs(maxiter=maxiter)


train.random_minibatch(size=minibatch_size)
train.bfgs(maxiter=maxiter)
train.random_minibatch(size=minibatch_size)
train.bfgs(maxiter=maxiter)

train.fullbatch()
train.bfgs(maxiter=5)

"""
tenbilac.plot.paramscurve(train)


#train.save("train_msrb.pkl")

