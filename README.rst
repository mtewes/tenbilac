Tenbilac
========

.. warning:: For now this is undocumented work in progress.

Tenbilac is a python package implementing an (experiemental) feedforward neural network designed to solve an *inverse regression* problem (aka "calibration problem" of regression, see `wikipedia <https://en.wikipedia.org/wiki/Calibration_(statistics)>`_). It performs a supervised machine learning regression from a noisy feature space to a well known explanatory space, and is trained to minimize *bias* instead of error in this explanatory space.

Despite some research, I could not find an implementation of such a "thing", so here it is. But why code this from scratch instead of using an existing neural network library ? The training is sufficiently different from what is done with normal ANNs, as tenbilac has to "experience" many realizations of the training set in order to probe bias. To simplify bookkeeping, to get more freedom in the design of the error function, and also aim for a robust behavior (committees have to be done anyway...), tenbilac uses black-box optimizers instead of back-propagation (for now).

Note that tenbilac is far from deep learning! Indeed the regressions that this network should perform are relatively simple. Typical architecture would be 5 to 10 input features, 2 hidden layers of 10 nodes each, and one output. But one could well use calibnet to calibrate the outputs of a deep learning machine or an auto-encoder, if dimensionality is a concern.

