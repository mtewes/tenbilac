
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


trutheta = np.linspace(0.0, 3.0, 100)
trud = np.sqrt(4.0 + np.square(trutheta))


fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(1, 3, 1)
ax.plot(params.T, obs[0].T, marker=".", color="gray", ls="None", ms=2)
ax.plot(testpreds_mse.T, testobs.T, "r-", label="Standard MSE", lw=1.5)
ax.plot(testpreds_avg.T, testobs.T, "b-", label=r"Learning on $< d >$", lw=1.5)
ax.plot(testpreds_msb.T, testobs.T, "g-", label="Tenbilac MSRB", lw=1.5)
ax.plot(trutheta, trud, "r-", color="black", dashes=(5, 5), lw=1.5, label=r"$d = \sqrt{2^2 + \theta^2}$(truth)")
ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)
ax.set_xlim(-1.2, 2.4)
ax.set_ylim(1.6, 3.1)
ax.legend(loc=2)

ax = fig.add_subplot(1, 3, 2)
ax.plot(uniparams.T, biases_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(uniparams.T, biases_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(uniparams.T, biases_msb.T, marker=".", color="green", ls="None", ms=2)
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.4, 0.4)


ax = fig.add_subplot(1, 3, 3)
ax.plot(firstreapreds_mse.T, biases_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(firstreapreds_avg.T, biases_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(firstreapreds_msb.T, biases_msb.T, marker=".", color="green", ls="None", ms=2)
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)
ax.set_xlim(-1.7, 2.5)
ax.set_ylim(-0.4, 0.4)




plt.tight_layout()
plt.show()	

