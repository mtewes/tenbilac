
import numpy as np
import tenbilac
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)


(n, nrea, noise_scale, params, obs, obs_normer, params_normer, normobs, normparams, uninrea, uniparams, uniobs, ntest, testobs, normtestobs, normuniparams, normuniobs) = tenbilac.utils.readpickle("data.pkl")

net_msb = tenbilac.utils.readpickle("net_msb.pkl")
testpreds_msb = params_normer.denorm(net_msb.run(normtestobs))
preds_msb = params_normer.denorm(net_msb.run(normuniobs))
firstreapreds_msb = params_normer.denorm(net_msb.run(normuniobs[0]))
biases_msb = np.mean(preds_msb, axis=0) - uniparams

net_mse = tenbilac.utils.readpickle("net_mse.pkl")
testpreds_mse = params_normer.denorm(net_mse.run(normtestobs))
preds_mse = params_normer.denorm(net_mse.run(normuniobs))
firstreapreds_mse = params_normer.denorm(net_mse.run(normuniobs[0]))
biases_mse = np.mean(preds_mse, axis=0) - uniparams

net_avg = tenbilac.utils.readpickle("net_avg.pkl")
testpreds_avg = params_normer.denorm(net_avg.run(normtestobs))
preds_avg = params_normer.denorm(net_avg.run(normuniobs))
firstreapreds_avg = params_normer.denorm(net_avg.run(normuniobs[0]))
biases_avg = np.mean(preds_avg, axis=0) - uniparams




fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(1, 3, 1)
ax.plot(params.T, obs[0].T, marker=".", color="gray", ls="None")
ax.plot(testpreds_mse.T, testobs.T, "r-", label="Standard MSE", lw=2)
ax.plot(testpreds_avg.T, testobs.T, "b-", label=r"Learning on $< d >$", lw=2)
ax.plot(testpreds_msb.T, testobs.T, "g-", label="Tenbilac MSB", lw=2)
ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)

ax.legend(loc=2)

ax = fig.add_subplot(1, 3, 2)
ax.plot(uniparams.T, biases_mse.T, marker=".", color="red", ls="None")
ax.plot(uniparams.T, biases_avg.T, marker=".", color="blue", ls="None")
ax.plot(uniparams.T, biases_msb.T, marker=".", color="green", ls="None")
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)



ax = fig.add_subplot(1, 3, 3)
ax.plot(firstreapreds_mse.T, biases_mse.T, marker=".", color="red", ls="None")
ax.plot(firstreapreds_avg.T, biases_avg.T, marker=".", color="blue", ls="None")
ax.plot(firstreapreds_msb.T, biases_msb.T, marker=".", color="green", ls="None")
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)




plt.tight_layout()
plt.show()	

