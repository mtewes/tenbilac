
import numpy as np
import tenbilac
import matplotlib.pyplot as plt


import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

n = 200
nrea = 1000 
noise_scale = 0.1

params = np.random.triangular(0.1, 0.3, 2.0, size=n).reshape((1, n))
obsreas = [np.sqrt(4.0 + params**2) + noise_scale*np.random.randn(n).reshape((1, n)) for rea in range(nrea)]

obs = np.dstack(obsreas).reshape(1, nrea*n)

splitinds = np.arange(nrea, nrea*n, nrea)


ntest = 100
testobs = np.linspace(1.5, 3, ntest).reshape((1, ntest))

"""
params_normer = tenbilac.utils.Normer(params)
obs_normer = tenbilac.utils.Normer(obs)
normparams = params_normer(params)
normobs = obs_normer(obs)
"""


net = tenbilac.net.Tenbilac(1, [10])
#net = tenbilac.utils.readpickle("test.pkl")

#for l in net.layers:
#l.addnoise(wscale=1.0)


net.error_calib(obs, params, splitinds)


#print net.error(obs, params)
net.train(obs, params, splitinds)
#print net.error(obs, params)




#net.save("test.pkl")

#exit()



#net = tenbilac.utils.readpickle("test.pkl")

testpreds = net.run(testobs)


preds = net.run(obs)
meanpreds = np.array([np.mean(case, axis=1) for case in np.split(preds, splitinds, axis=1)]).transpose()
biases = meanpreds - params


#firstpreds = np.array([case[:,0:1][0] for case in np.split(preds, splitinds, axis=1)]).transpose()

#print firstpreds.shape	

#exit()

fig = plt.figure(figsize=(15, 4))

	
ax = fig.add_subplot(1, 3, 1)
ax.plot(params.T, obsreas[0].T, "b,")
ax.plot(testpreds.T, testobs.T, "r-")
ax.set_xlabel(r"$\theta$", fontsize=18)
ax.set_ylabel(r"$d$", fontsize=18)


ax = fig.add_subplot(1, 3, 2)
ax.plot(params.T, biases.T, "b,")
ax.axhline(0.0)
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

