
import matplotlib.pyplot as plt
import numpy as np
import tenbilac



inputs = np.linspace(-1, 1, 1000)

ws = [-2, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
ws = [-2, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

ws = np.linspace(-1, 1, 5)
ws = [-0.1, 0.1, 1.0, 2.0]

for w in ws:
	#plt.plot(inputs, tenbilac.act.iden(np.power(inputs, 2.0*tenbilac.act.sig(w))), label="w="+str(w))
	#plt.plot(inputs, np.real(np.power(inputs+0j,  2.0*tenbilac.act.sig(10.0*w))), label="w="+str(w))
	plt.plot(inputs, np.sign(inputs)*np.power(np.fabs(inputs),  w), label="w="+str(w))

	#plt.plot(inputs, np.real(np.power(inputs + 0j,  w)), label="w="+str(w))


plt.xlabel("input")
plt.ylabel("output = input ** w")
plt.legend()
plt.show()


