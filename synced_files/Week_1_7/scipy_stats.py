# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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

# %% [markdown]
# **End of notebook.**
#
# <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
#   <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#   </div>
#   <div style="font-size: 75%; margin-top: 10px; text-align: right;">
#     &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. 
#     This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
