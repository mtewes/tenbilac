"""
Demo to make "the figure"
"""

import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)

def noise(n):
	return np.random.randn(n)
	#return 0.1*(np.exp(0.4*np.random.randn(n)) - 0.7)
	
def f(x):
	return np.sqrt(1.0 + x**2)


# The data that will be used for training (to make it faster, reduce number of cases...)
train_ncase = 1000 # Number of cases (i.e., number of parameters "theta")
train_nrea = 1000 # How many realizations per case (i.e., "observations" per parameter).
noise_scale = 0.1
# The explanatory variables:
train_params = np.linspace(0.25, 2.0, train_ncase).reshape((1, train_ncase))
# And the corresponding observations of the dependt variable:
train_obs = np.array([f(train_params) + noise_scale*noise(train_ncase).reshape((1, train_ncase)) for rea in range(train_nrea)])


# To study bias afterwards, we make a "validation" set.
val_ncase = 1000
val_nrea = 1000
val_params = np.linspace(0.25, 2.0, val_ncase).reshape((1, val_ncase))
val_obs = np.array([f(val_params) + noise_scale*noise(val_ncase).reshape((1, val_ncase)) for rea in range(val_nrea)])


# To plot the inverse regression as continuous lines, it's good to have samples uniform in the dependent variable:
test_obs = np.linspace(0.5, 3, 100).reshape((1, 100))


# Now we norm all these. We build the Normers by using the larger validation data (but training would work as well...):
obs_normer = tenbilac.data.Normer(val_obs)
params_normer = tenbilac.data.Normer(val_params)

# And apply them:
normed_train_params = params_normer(train_params)
normed_train_obs = obs_normer(train_obs)
normed_val_params = params_normer(val_params)
normed_val_obs = obs_normer(val_obs)
normed_test_obs = obs_normer(test_obs)

# And save a pkl file


pkldata = {
	"obs_normer":obs_normer,
	"params_normer":params_normer,
	"train_params":train_params,
	"train_obs":train_obs,
	"normed_train_params":normed_train_params,
	"normed_train_obs":normed_train_obs,
	"val_params":val_params,
	"val_obs":val_obs,
	"normed_val_params":normed_val_params,
	"normed_val_obs":normed_val_obs,
	"test_obs":test_obs,
	"normed_test_obs":normed_test_obs,
}

tenbilac.utils.writepickle(pkldata, "data.pkl")

