import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.DEBUG)


data = tenbilac.utils.readpickle("data.pkl")


net = tenbilac.wnet.WNet(2, [5], 1, name="My WNet", inames=["obs_size", "obs_flux"], onames=["pre_size"])
net.setini()

traindata = tenbilac.data.Traindata(data["norminp"], data["tar"], valfrac=0.5, shuffle=True)

train = tenbilac.train.Training(net, traindata, errfctname="msbw")

#train.set_paramslice(mode="o")
#train.minibatch_bfgs(mbsize=100, mbloops=3, maxiter=30)
#train.save("train_msbw_no_w.pkl", keepdata=True)
#train.net.netw.addnoise(wscale=0.1, bscale=0.1)

train.set_paramslice(mode=None)
train.net.netw.addnoise(wscale=0.1, bscale=0.1)
train.net.neto.addnoise(wscale=0.1, bscale=0.1)
train.minibatch_bfgs(mbsize=250, mbloops=1, maxiter=100)


train.save("train_msbw.pkl", keepdata=True)


