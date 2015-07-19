import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


train = tenbilac.utils.readpickle("train_msrb.pkl")
#tenbilac.plot.errorcurve(train)
tenbilac.plot.paramscurve(train)

"""
net = tenbilac.utils.readpickle("net_avg.pkl")
tenbilac.plot.errorcurve(net)

net = tenbilac.utils.readpickle("net_mse.pkl")
tenbilac.plot.errorcurve(net)
"""
