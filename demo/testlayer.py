
import numpy as np
import calibnet


import logging
logging.basicConfig(level=logging.INFO)



layer = calibnet.layer.Layer(ni=2, nn=3, actfct=calibnet.act.Id())

#layer.addnoise()

#layer.report()


layer.weights[1, 1] = 1
layer.biases[2] = 10

layer.report()


input = np.array([[1, 1, 1, -1], [2, 2, 2, -2]])
#input = np.array([1, 2])


print "input = ", input

output = layer.run(input)

print "ouput = ", output

