"""
Running the new interface
"""

import numpy as np
import tenbilac


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


(inputs, targets) = tenbilac.utils.readpickle("data.pkl")

# And go:

ten = tenbilac.com.Tenbilac("tenbilac.cfg")
ten.train(inputs, targets)


# Demo of alternative way to load an existing tenbilac object:
#ten = tenbilac.com.Tenbilac("/vol/fohlen11/fohlen11_1/mtewes/tenbilac_demo_workdir/mini_tenbilac_2017-01-31T21-17-52")

preds = ten.predict(inputs)
print preds
