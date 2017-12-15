
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


data = tenbilac.utils.readpickle("data.pkl")

# We average the training observations over the realizations:
avg_normed_train_obs = np.mean(data["normed_train_obs"], axis=0).reshape(1, 1, data["normed_train_obs"].shape[2])


traindata = tenbilac.data.Traindata(avg_normed_train_obs, data["normed_train_params"], valfrac=0.5, shuffle=True)

net = tenbilac.net.Net(1, [7])
net.setidentity()
net.addnoise(wscale=0.1, bscale=0.1)

train = tenbilac.train.Training(net, traindata, errfctname="mse")

train.opt(algo="bfgs", mbfrac=1, mbloops=1, maxiter=200)

tenbilac.plot.sumevo(train, filepath="train_avg.png")

train.save("train_avg.pkl", keepdata=True)



