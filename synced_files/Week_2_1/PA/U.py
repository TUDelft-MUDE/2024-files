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
length = 10
height = (length**2 - (length/2)**2)**0.5
x = [ 0 , length , length/2 , length/2+length ,   length  ,   2*length , (5/2)*length ,  3*length , (7/2)*length  ,  3*length , 4*length ]
y = [ 0 ,   0    , -height  ,     -height     , -2*height ,  -2*height , -height      , -2*height ,  -height      ,   0       , 0 ]
 

# %%
import matplotlib.pyplot as plt

plt.scatter(x, y)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
from city import *

# %%
coordinates = np.array([x,y]).T
print(coordinates)

# %%
print(my_plan.triangles)

# %%


my_plan = Plan(coordinates, length)
my_plan.define_triangles()
my_plan.get_all_sides();

# %%
my_plan.try_triangles()

# %%
my_plan.refine_mesh()
my_plan.plot_triangles()

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
