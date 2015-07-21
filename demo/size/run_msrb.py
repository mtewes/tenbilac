
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


(obs_normer, params_normer, normobs, normparams, normuniparams, normuniobs, normtestobs) = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.net.Tenbilac(1, [7])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)

traindata = tenbilac.data.Traindata(normobs, normparams, valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, traindata, errfctname="msrb")

train.minibatch_bfgs(mbsize=100, mbloops=5, maxiter=50)

tenbilac.plot.paramscurve(train, filepath="train_msrb.png")

train.save("train_msrb.pkl", keepdata=True)


#tenbilac.plot.paramscurve(train)

