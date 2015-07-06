
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt


"""
# Test of dot on masked arrays:

a = np.ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, False]])
b = np.ma.array([[1, 2], [3, 4]], mask=[[False, False], [False, False]])

print "a = "
print a

print "b = "
print b

print "dot = "
print np.ma.dot(a, b, strict=True) # This works, but only for 2D arrays !
# See code of np.ma.dot, it's quite explicit.

"""


"""
ni = 3
ngal = 4
nrea = 1
no = 1

net = tenbilac.net.Tenbilac(ni=ni, nhs=[3, 3], onlyid=True)

net.setidentity()
net.addnoise()
print net.report()

inputs = np.random.randn(nrea* ni * ngal).reshape((nrea, ni, ngal))
r_mask = 1.0
inputs = np.ma.masked_outside(inputs, -r_mask, r_mask)

print "Input:"
print inputs.shape
print inputs


#output = net.run(inputs)

output = net.layers[0].run(inputs)
#output = net.layers[1].run(output)
#output = net.layers[2].run(output)


print "Output:"
print output.shape
print output

"""
