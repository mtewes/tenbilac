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

#ten.train(inputs, targets)
ten.summary()

