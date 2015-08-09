
import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

"""
# Test with 5 reas, 2 nodes, 3 cases.

a = np.random.randn(5*2*3).reshape(5, 2, 3)
mask = np.zeros(5*2*3).reshape(5, 2, 3)
a = np.ma.array(a, mask=mask)

# First node and case has one crazy value:
a[0,0,0] = 1.0e8

print "mean"
print np.mean(a, axis=0)

a.mask[0,0,0] = True

print np.mean(a, axis=0)

# What happens if all realizations are masked :

a.mask[:,0,0] = True

print np.mean(a, axis=0)

print "std"
print np.std(a, axis=0)
a.mask[:,0,0] = False
print np.std(a, axis=0)

# And if only one rea is available:

a.mask[1:,0,0] = True
print np.std(a, axis=0)

# OK it gives 0.0. All as exected.
"""

# Testing tile: Good, works as expected.
a = np.ma.array([1, 2, 3, 4], mask=[True, False, False, False])

print a
print np.tile(a, 3)





"""
# HUGE WARNING: NP.RAVEL() SILENTLY IGNORES THE MASK !!!
a = np.ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, False]])

print a

print np.ravel(a)

print a.flatten()

"""


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
