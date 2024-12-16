# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: mude-base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tips for Teachers: Week 1.7, Continuous Distributions
#
# Book chapters [here](https://mude.citg.tudelft.nl/2024/book/probability/Reminder_intro.html).
#
# [Location, shape and scale](https://mude.citg.tudelft.nl/2024/book/probability/Loc-scale.html)

# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, gumbel_r

# %%
loc = 28.167
scale = 13.097
print(gumbel_r.pdf(30, loc, scale))
print(gumbel_r.cdf(30, loc, scale))
print(gumbel_r.ppf(0.4192, loc, scale))

print(gumbel_r.ppf(1/773, loc, scale))

# %%
dir(gumbel_r)

# %%
print(1/773, 772/773)

# %% [markdown]
# Scipy.stats
#
# [Homepage](https://docs.scipy.org/doc/scipy/reference/stats.html)
#
# If you find yourself looking at documentation for something and there is no info, try the [`rv_continuous` page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous)
#
#

# %%
test = norm('m'=0, 'v'=1)
