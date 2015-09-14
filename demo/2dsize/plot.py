import numpy as np
import tenbilac
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)


data = tenbilac.utils.readpickle("data.pkl")
train = tenbilac.utils.readpickle("train_msbw.pkl")
#train = tenbilac.utils.readpickle("train_msbw_no_w.pkl")

# Making predictions
pred = train.net.run( data["norminp"])


pre_sizes = pred[:,0,:]
pre_weights = 10**pred[:,1,:]


wavg_pre_sizes = np.mean(pre_sizes * pre_weights, axis=0)
avg_pre_sizes = np.mean(pre_sizes, axis=0)


fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 2, 1)


ax.scatter(data["tar"], data["inp"][0, 0, :], c=data["inp"][0, 1, :])
ax.set_xlabel("tru_size")
ax.set_ylabel("obs_size")


ax = plt.subplot(2, 2, 2)

ax.scatter(data["inp"][0, 1, :], pre_weights[0,:], c=data["inp"][0, 0, :])
ax.set_xlabel("obs_flux")
ax.set_ylabel("pre_weight")

ax = plt.subplot(2, 2, 3)
ax.scatter(data["tar"], wavg_pre_sizes - data["tar"])
ax.set_xlabel("tru_size")
ax.set_ylabel("wavg_pre_size - tru_size")


ax = plt.subplot(2, 2, 4)
ax.scatter(data["tar"], avg_pre_sizes - data["tar"])
ax.set_xlabel("tru_size")
ax.set_ylabel("wavg_pre_size - tru_size")


plt.tight_layout()
	
plt.show()


