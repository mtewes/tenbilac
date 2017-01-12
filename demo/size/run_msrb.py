
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


(obs_normer, params_normer, normobs, normparams, normuniparams, normuniobs, normtestobs) = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.net.Net(1, [7])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)

traindata = tenbilac.data.Traindata(normobs, normparams, valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, traindata, errfctname="msrb")

train.opt(algo="bfgs", mbsize=250, mbloops=1, maxiter=250)

tenbilac.plot.sumevo(train, filepath="train_msrb.png")

train.save("train_msrb.pkl", keepdata=True)


#tenbilac.plot.paramscurve(train)

