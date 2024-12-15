
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, gumbel_r

loc = 28.167
scale = 13.097
print(gumbel_r.pdf(30, loc, scale))
print(gumbel_r.cdf(30, loc, scale))
print(gumbel_r.ppf(0.4192, loc, scale))

print(gumbel_r.ppf(1/773, loc, scale))

dir(gumbel_r)

print(1/773, 772/773)

test = norm('m'=0, 'v'=1)
