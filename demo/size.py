
import numpy as np
import tenbilac
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

n = 1000
nrea = 100 
noise_scale = 0.1

params = np.random.triangular(0.1, 0.3, 2.0, size=n).reshape((1, n))
obs = np.array([np.sqrt(4.0 + params**2) + noise_scale*np.random.randn(n).reshape((1, n)) for rea in range(nrea)])

#print obs.shape



obs_normer = tenbilac.utils.Normer(obs)
params_normer = tenbilac.utils.Normer(params)

normobs = obs_normer(obs)
normparams = params_normer(params)


#Idea : train this so that it fits well only the large theta, disregard the others to see if wave ?

ntest = 100
testobs = np.linspace(1.6, 3, ntest).reshape((1, ntest))
normtestobs = obs_normer(testobs)



net_msb = tenbilac.net.Tenbilac(1, [10])
net_msb.addnoise()
net_msb.train(normobs, normparams, tenbilac.err.msb, maxiter=200)
net_msb.save("net_msb.pkl")


net_mse = tenbilac.net.Tenbilac(1, [10])
net_mse.addnoise()
net_mse.train(normobs, normparams, tenbilac.err.mse, maxiter=200)
net_mse.save("net_mse.pkl")




#net = tenbilac.utils.readpickle("test.pkl")
#exit()



testpreds_msb = params_normer.denorm(net_msb.run(normtestobs))
preds_msb = params_normer.denorm(net_msb.run(normobs))
biases_msb = np.mean(preds_msb, axis=0) - params

testpreds_mse = params_normer.denorm(net_mse.run(normtestobs))
preds_mse = params_normer.denorm(net_mse.run(normobs))
biases_mse = np.mean(preds_mse, axis=0) - params




fig = plt.figure(figsize=(10, 4))

	
ax = fig.add_subplot(1, 2, 1)
ax.plot(params.T, obs[0].T, marker=".", color="gray", ls="None")
ax.plot(testpreds_mse.T, testobs.T, "r-", label="Standard MSE", lw=2)
ax.plot(testpreds_msb.T, testobs.T, "g-", label="Tenbilac MSB", lw=2)
ax.set_xlabel(r"$\theta$ $\mathrm{and}$ $\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)

ax.legend(loc=2)

ax = fig.add_subplot(1, 2, 2)
ax.plot(params.T, biases_mse.T, marker=".", color="red", ls="None")
ax.plot(params.T, biases_msb.T, marker=".", color="green", ls="None")
ax.axhline(0.0, color="gray")
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)


"""
ax = fig.add_subplot(1, 3, 3)
ax.plot(firstpreds.T, biases.T, "b,")
ax.axhline(0.0)
ax.set_xlabel(r"$\hat{\theta}$", fontsize=18)
ax.set_ylabel(r"$< \hat{\theta} - \theta >$", fontsize=18)
"""




plt.tight_layout()
plt.show()	

