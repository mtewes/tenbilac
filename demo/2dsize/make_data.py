
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)

#np.random.seed(0)



# The data that will be used for training:
ncas = 500 # Number of cases
nrea = 100 # How many realizations per case


tru_sizes = np.random.uniform(0.1, 2.0, size=ncas) # We have ncas different true sizes.

realist = []
for tru_size in tru_sizes: # Each case has nrea realizations, with different fluxes:
	
	obs_fluxes = np.random.uniform(10, 100, size=nrea)
	size_noise_scales = 1.0/np.sqrt(obs_fluxes) # The higher the flux, the lower the noise on the size
	size_noises = np.random.randn(nrea) * size_noise_scales
	obs_sizes = np.sqrt(4.0 + tru_size**2) + size_noises

	reas = np.vstack((obs_sizes, obs_fluxes)).transpose()
	assert reas.shape == (nrea, 2) # We have two features: obs_sizes and obs_fluxes
	
	realist.append(reas)

inp = np.dstack(realist)
assert inp.shape == (nrea, 2, ncas) # The input is 3D: (rea, feature, case).


# The targets are the tru_sizes, one node.

tar = tru_sizes.reshape((1, ncas))
assert tar.shape == (1, ncas)


# The outputs will be predicted sizes and corresponding weights, for each realization.


# We normalize the inputs

inp_normer = tenbilac.data.Normer(inp) 

norminp = inp_normer(inp)


# And save a pkl file

pkldata = {
	"inp_normer":inp_normer,
	"norminp":norminp,
	"inp":inp,
	"tar":tar,
	}

tenbilac.utils.writepickle(pkldata, "data.pkl")
