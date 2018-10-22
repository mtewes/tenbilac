
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


data = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.net.Net(1, [5])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)

traindata = tenbilac.data.Traindata(data["normed_train_obs"], data["normed_train_params"], valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, traindata, errfctname="msb")

train.opt(algo="bfgs", mbfrac=1, mbloops=1, maxiter=200)

tenbilac.plot.sumevo(train, filepath="train_msb.png")

train.save("train_msb.pkl", keepdata=True)


