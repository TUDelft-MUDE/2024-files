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
# # Illustration `bivariate`
#
#

# %%
import bivariate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyvinecopulib as cop
import scipy.stats as st

# %%
X_1 = st.norm(loc=3, scale=1)
X_2 = st.norm(loc=5, scale=1)

n = 10000

X_1_samples = X_1.rvs(size=n)
X_2_samples = X_2.rvs(size=n)

X_combined_samples = np.array([X_1_samples, X_2_samples]).T

X_class_A = bivariate.class_copula.Region_of_interest(
                random_samples=X_combined_samples)

X_class_A.plot_emperical_contours(bandwidth=4)

def underwater(X1,X2):
    Z_now = 10.0
    function = (Z_now - X1 - X2 <= 0)
    return function
  
X_class_A.function =  underwater
X_class_A.inside_function()
X_class_A.plot_inside_function();

# %%
# define multivariate normal distribution

X = st.multivariate_normal(mean=[3, 5],
                           cov=[[1, 0.5],
                                [0.5, 1]])

n = 10000
X_samples = X.rvs(size=n)

X_class_A = bivariate.class_copula.Region_of_interest(
                random_samples=X_samples)

X_class_A.plot_emperical_contours(bandwidth=4)

def underwater(X1,X2):
    Z_now = 10.0
    function = (Z_now - X1 - X2 <= 0)
    return function
  
X_class_A.function =  underwater
X_class_A.inside_function()
X_class_A.plot_inside_function();


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
