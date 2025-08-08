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
import numpy as np
from scipy import stats
import pyvinecopulib as pv
import matplotlib.pyplot as plt
import matplotlib


# %%




def joint_exceedance(rho):
    x = [4, 6]
    X = stats.multivariate_normal(mean=[3, 5],
                              cov=[[1.00, rho],
                                   [rho, 2.60]])
    
    X1 = stats.norm(loc=3, scale=1.0)
    X2 = stats.norm(loc=5, scale=1.6)

    p1 = X1.cdf(x[0])
    p2 = X2.cdf(x[1])
    p = X.cdf(x)

    return 1 - p1 - p2 + p

rho = np.linspace(-0.99, 0.99, 100)
p = np.zeros(len(rho))
for i, r in enumerate(rho):
    p[i] = joint_exceedance(r)


plt.plot(rho, p)
plt.plot(0,
         (1-stats.norm(loc=3, scale=1.0).cdf(4))*(1-stats.norm(loc=5, scale=1.6).cdf(6)),
         'r', marker='o')
plt.xlabel(r'$\rho$')
plt.ylabel('Joint exceedance probability')
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.show()

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
