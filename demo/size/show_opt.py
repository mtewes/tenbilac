import numpy as np
import tenbilac

import logging
logging.basicConfig(level=logging.INFO)


net = tenbilac.utils.readpickle("net_msb.pkl")
#tenbilac.plot.errorcurve(net)
tenbilac.plot.paramscurve(net)

"""
net = tenbilac.utils.readpickle("net_avg.pkl")
tenbilac.plot.errorcurve(net)

net = tenbilac.utils.readpickle("net_mse.pkl")
tenbilac.plot.errorcurve(net)
"""
