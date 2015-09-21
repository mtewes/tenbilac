import numpy as np
import tenbilac
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)


data = tenbilac.utils.readpickle("data.pkl")


train = tenbilac.utils.readpickle("train_msbwnet.pkl")

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(3, 1, 1)

tenbilac.plot.errevo(ax, train, showtimes=True)

ax = plt.subplot(3, 1, 2)
tenbilac.plot.paramsevo(ax, train, wnetpart="o")

ax = plt.subplot(3, 1, 3)
tenbilac.plot.paramsevo(ax, train, wnetpart="w")


plt.tight_layout()
	
plt.show()


