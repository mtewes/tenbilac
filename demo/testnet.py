
import numpy as np
import calibnet


import logging
logging.basicConfig(level=logging.INFO)



net = calibnet.net.Calibnet(ni=2, nhs=[3, 3], nrea=1, onlyid=True)

#for l in net.layers:
#	l.addnoise()

#print net.nparams()

params = net.get_params_ref()

#print params

#layer.addnoise()

#layer.report()


#layer.weights[1, 1] = 1
#layer.biases[2] = 10
#layer.report()

ni = 2
ngal = 5


#input = np.random.randn(ni * ngal).reshape((ni, ngal))
input = np.ones(ni * ngal).reshape((ni, ngal))

#input = np.array([[1, 1, -1], [2, 2, -2]])
#input = np.array([1, 2])


print "input = ", input

params[0] = 10
params[9] = 20
params[21] = 20

print net.report()

output = net.run(input)
print "ouput = ", output




