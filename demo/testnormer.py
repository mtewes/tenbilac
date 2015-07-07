
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


ni = 5
ngal = 4
nrea = 3
no = 1
noise_scale = 1.0

# 3D case:
inputs = np.random.randn(ni * ngal * nrea).reshape((nrea, ni, ngal))*1000.0 + 15.0

# 2D case:
#input = np.random.randn(ni * ngal).reshape((ni, ngal))*1000.0 + 15.0


print inputs
print inputs.shape

normer = tenbilac.utils.Normer(inputs, type="01")

normed = normer(inputs)

print normed

print np.min(normed), np.max(normed)

denormed = normer.denorm(normed)

if not np.allclose(denormed, inputs):
	raise RuntimeError("Comparision test failed !")
else:
	print "Test is good!"

