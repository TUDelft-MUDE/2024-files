import numpy as np
import matplotlib.pylab as plt

a = np.array([0.013, 0.556, 0.374,
0.308, 0.115, -0.36,
-0.31, -1.05, 0.127,
1.052, -0.10, 0.000,
0.000, -0.689])
np.allclose
assert (np.allclose(a, [0.013, 0.556, 0.374,
                              0.308, 0.115, -0.36,
                              -0.31, -1.05, 0.127,
                              1.052, -0.10, 0.000,
                              0.000, -0.689],
                              rtol=1e-2),
"fail")