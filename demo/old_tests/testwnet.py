
import numpy as np
import tenbilac


import logging
logging.basicConfig(level=logging.INFO)


ni = 1

wnet = tenbilac.wnet.WNet(ni=ni, nhs=[3, 3], name="test", netokwargs={"onlyid":True})

#print wnet.nparams()
#print wnet.report()


ref = wnet.get_params_ref()

wnet.addnoise()

#print ref
#print wnet.report()
wnet.setini()
print wnet.report()



ncas = 3
nrea = 5
no = 1

inputs = np.random.randn(nrea * ni * ncas).reshape((nrea, ni, ncas))


#print inputs

outputs = wnet.run(inputs)


#print outputs

targets = np.ones(no * ncas).reshape((no, ncas))

# There is no by default weights, so let's initialise the weights to ones:
weights = np.ones_like(outputs)

print tenbilac.err.msbw(outputs, targets, auxinputs=weights)

"""
print "ouputs shape ", outputs.shape



targets = np.ones(no * ngal).reshape((no, ngal))


net.train(inputs, targets, tenbilac.err.msb, maxiter=3)



print net.report()
"""
