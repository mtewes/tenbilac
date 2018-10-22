
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)



net = tenbilac.net.Net(ni=2, nhs=[3, 3], onlyid=True)

params = net.get_params_ref()

print params

net.addnoise()

print params


net.setidentity()

print params

print net.report()

ni = 2
ngal = 500
nrea = 10
no = 1


#inputs = np.random.randn(ni * ngal).reshape((ni, ngal))
inputs = np.ones(ni * ngal * nrea).reshape((nrea, ni, ngal))


#inputs = np.array([[1, 1, -1], [2, 2, -2]])
#inputs = np.array([1, 2])


print "inputs shape", inputs.shape

params[0] = 10
params[9] = 20
params[21] = 20

print net.report()

outputs = net.run(inputs)
print "ouputs shape ", outputs.shape



targets = np.ones(no * ngal).reshape((no, ngal))

# Prepare a traindata object 
dat = tenbilac.data.Traindata(inputs=inputs, targets=targets)

# Generate the network
training = tenbilac.train.Training(net, dat, errfctname="msb")

# We train this normal (non-inverse) regression with params as inputs, and observations as output:
training.opt(algo="bfgs", maxiter=3)



print net.report()
