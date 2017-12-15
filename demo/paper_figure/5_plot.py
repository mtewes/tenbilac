
import numpy as np
import tenbilac
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



d = tenbilac.utils.readpickle("data.pkl")

params_normer = d["params_normer"]
obs_normer = d["obs_normer"]
test_obs = d["test_obs"]
val_params = d["val_params"]


# One one plot, we show the actual training data points.
# We modify those arrays to get everything into flat 1D arrays
mod_train_params = np.concatenate(d["train_params"].shape[0]*[d["train_params"].T]).T[0]
mod_train_obs = d["train_obs"].ravel()

# Showing all of them looks ugly, just a few:
showtrainindices = np.arange(mod_train_params.size)
np.random.shuffle(showtrainindices)
showtrainindices = showtrainindices[:10000]
mod_train_params = mod_train_params[showtrainindices]
mod_train_obs = mod_train_obs[showtrainindices]


train_msb = tenbilac.utils.readpickle("train_msb.pkl")
test_preds_msb = params_normer.denorm(train_msb.net.run(d["normed_test_obs"]))
val_preds_msb = params_normer.denorm(train_msb.net.run(d["normed_val_obs"]))
#firstreapreds_msb = params_normer.denorm(train_msb.net.run(normuniobs[0]))
val_biases_msb = np.mean(val_preds_msb, axis=0) - d["val_params"]


train_mse = tenbilac.utils.readpickle("train_mse.pkl")
test_preds_mse = params_normer.denorm(train_mse.net.run(d["normed_test_obs"]))
val_preds_mse = params_normer.denorm(train_mse.net.run(d["normed_val_obs"]))
#firstreapreds_mse = params_normer.denorm(train_mse.net.run(normuniobs[0]))
val_biases_mse = np.mean(val_preds_mse, axis=0) - d["val_params"]


train_avg = tenbilac.utils.readpickle("train_avg.pkl")
test_preds_avg = params_normer.denorm(train_avg.net.run(d["normed_test_obs"]))
val_preds_avg = params_normer.denorm(train_avg.net.run(d["normed_val_obs"]))
#firstreapreds_avg = params_normer.denorm(train_avg.net.run(normuniobs[0]))
val_biases_avg = np.mean(val_preds_avg, axis=0) - d["val_params"]



trutheta = np.linspace(0.0, 2.2, 100)
trud = np.sqrt(1.0 + np.square(trutheta))

fig = plt.figure(figsize=(9, 4.5))

ax = fig.add_subplot(1, 2, 1)
ax.plot(mod_train_params, mod_train_obs, marker=".", color="gray", ls="None", ms=2)
ax.plot(test_preds_mse.T, test_obs.T, "r-", label="Trained with MSE", lw=1.5)
ax.plot(test_preds_avg.T, test_obs.T, "b-", label=r"Trained on $\langle d \rangle$", lw=1.5)
ax.plot(test_preds_msb.T, test_obs.T, "g-", label="Trained with MSB", lw=1.5)
ax.plot(trutheta, trud, "r-", color="black", dashes=(5, 5), lw=1.5, label=r"$d = \sqrt{1 + \theta^2}$ (truth)")
ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)
ax.set_xlim(-1.2, 2.4)
ax.set_ylim(0.5, 3.0)
ax.legend(loc=2)


ax = fig.add_subplot(1, 2, 2)
ax.plot(val_params.T, val_biases_mse.T, marker=".", color="red", ls="None", ms=2)
ax.plot(val_params.T, val_biases_avg.T, marker=".", color="blue", ls="None", ms=2)
ax.plot(val_params.T, val_biases_msb.T, marker=".", color="green", ls="None", ms=2)
ax.axhline(0.0, color="black", lw=2)
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$\langle \hat{\theta} - \theta \rangle$", fontsize=18)
ax.set_xlim(0.0, 2.0)
ax.set_ylim(-0.3, 0.3)



plt.tight_layout()
#plt.show()
plt.savefig("MSB_demo.pdf")

