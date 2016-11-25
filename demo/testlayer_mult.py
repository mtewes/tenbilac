
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


# 2 inputs and 3 neurons

layer = tenbilac.layer.Layer(ni=2, nn=3, actfct=tenbilac.act.iden, mode="mult")

#layer.addnoise()
#print layer.report()

#layer.zero()
#print layer.report()

# First neuron: simple product
layer.weights[0, 0] = 1
layer.weights[0, 1] = 1

# Second neuron: product of squres
layer.weights[1, 0] = 2.0
layer.weights[1, 1] = 2.0

# Third one is left to play 
layer.weights[2, 0] = 1.0
layer.weights[2, 1] = -1.0

layer.biases[2] = 42

print layer.report()

print "1D case, just a single case or realization"

inp = np.array([1.0, 2.0])

outp = layer.run(inp)

print "input = ", inp

print "ouput = ", outp


print "2D input: (feature=2, case=4identical)"

inp = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

outp = layer.run(inp)

print "input = ", inp

print "ouput = ", outp, "shape", outp.shape



print "3D input: (realization=5, feature=2, case=4)"

inp = np.array([
[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
])

outp = layer.run(inp)

print "input = ", inp

print "ouput = ", outp, "shape",  outp.shape

