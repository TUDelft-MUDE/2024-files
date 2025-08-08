# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
# from vis import *
from tickets import *
from minutes import *
from models import *

# %%
all = Tickets()
all.add([[1, 60]])

day_min = np.zeros((len(all.tickets), 2))
for i in range(len(all.tickets)):
    day_min[i, :] = Minutes.get_day_min(all.tickets[i])
print(day_min.shape)

parameters = [30, 5, 720, 60]

transform = Minutes.get_transform(parameters)

day_t, min_t = transform(day_min[:,0], day_min[:,1])
day_min_radius = Minutes.radius(day_t, min_t)
print(type(day_min_radius))
print(day_min_radius.shape)

radius_array = np.zeros((60, 1440))
for i, r in enumerate(day_min_radius):
    d, m = Minutes.get_day_min(all.tickets[i])
    radius_array[d-1, m] = r

# %%
m = Models(model_id=1)
m.plot(radius_array, custom_label="Number of Std Devs from Mean",
       custom_title="Check radius for aribitrary parameters")

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
#     By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
#     &copy; 2024 TU Delft. 
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
#     <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
