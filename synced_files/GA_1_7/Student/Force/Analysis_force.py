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
# # Group Assignment 1.7: Distribution Fitting
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px; height: auto; margin: 0" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px; height: auto; margin: 0" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 7, Friday Oct 18, 2024.*

# %% [markdown] id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945"
# ## Case 1: Wave impacts on a crest wall
#
# **What's the propagated uncertainty? *How large will the horizontal force be?***
#
# In this project, you have chosen to work on the uncertainty of wave periods and wave heights in the Alboran sea to estimate the impacts on a crest wall: a concrete element installed on top of mound breakwater. You have observations from buoys of the significant wave height ($H$) and the peak wave period ($T$) each hour for several years. As you know, $H$ and $T$ are hydrodynamic variables relevant to estimate wave impacts on the structure. The maximum horizontal force (exceeded by 0.1% of incoming waves) can be estimated using the following equation (USACE, 2002).
#
# $$
# F_h = \left( A_1 + A_2 \frac{H}{A_c} \right) \rho g C_h L_{0p}
# $$
#
# where $A_1=-0.016$ and $A_2=0.025$ are coefficients that depend on the geometry of the structure, $A_c=3m$ is the elevation of the frontal berm of the structure, $\rho$ is the density of water, $g$ is the gravity acceleration, $C_h=2m$ is the crown wall height, and $L_{0p}=\frac{gT^2}{2\pi}$ is the wave length in deep waves. Thus, the previous equation is reduced to
#
# $$
# F_h = 255.4 H T^2 -490.4 T^2
# $$
#
# **The goal of this project is:**
# 1. Choose a reasonable distribution function for $H$ and $T$.
# 2. Fit the chosen distributions to the observations of $H$ and $T$.
# 3. Assuming $H$ and $T$ are independent, propagate their distributions to obtain the distribution of $F_h$.
# 4. Analyze the distribution of $F_h$.

# %% [markdown] id="d33f1148-c72b-4c7e-bca7-45973b2570c5"
# ## Importing packages

# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})

# %% [markdown]
# ## 1. Explore the data

# %% [markdown]
# First step in the analysis is exploring the data, visually and through its statistics.

# %%
# Import
_, H, T = np.genfromtxt('dataset_HT.csv', delimiter=",", unpack=True, skip_header=True)

# plot time series
fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(H,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Wave height, H (m)')
ax[0].grid()

ax[1].plot(T,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water period, T (s)')
ax[1].grid()

# %%
# Statistics for H

print(stats.describe(H))

# %%
# Statistics for d

print(stats.describe(T))


# %% [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1:</b>
#
# Describe the data based on the previous statistics:
#     <li>Which variable presents a higher variability?</li>
#     <li>What does the skewness coefficient means? Which kind of distribution functions should we consider to fit them?</li>
# </p>
# </div>

# %% [markdown]
# ## 2. Empirical distribution functions

# %% [markdown]
# Now, we are going to compute and plot the empirical PDF and CDF for each variable. Note that you have the pseudo-code for the empirical CDF in the [reader](https://mude.citg.tudelft.nl/book/probability/empirical.html).

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2:</b>  
#  
# Define a function to compute the empirical CDF. Plot your empirical PDF and CDF.
# </p>
# </div>

# %%
def ecdf(YOUR_INPUTS):
    #your code
    return YOUR_OUTPUT


# %%
# Your plot here

# %% [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3:</b>   
#
# Based on the results of Task 1 and the empirical PDF and CDF, select <b>one</b> distribution to fit to each variable. For $H$, select between Exponential or Gaussian distribution, while for $T$ choose between Uniform or Gumbel.
# </p>
# </div>

# %% [markdown]
# ## 3. Fitting a distribution

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 4:</b>  
#  
# Fit the selected distributions to the observations using MLE.
# </p>
# </div>
#
# Hint: Use [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html) built in functions (watch out with the parameters definition!).

# %%
#Your code here

# %% [markdown]
# ## 4. Assessing goodness of fit

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 5:</b>  
#  
# Assess the goodness of fit of the selected distribution using:
#     <li> One graphical method: QQplot or Logscale. Choose one.</li>
#     <li> Kolmogorov-Smirnov test.</li>
# </p>
# </div>
#
# Hint: You have Kolmogorov-Smirnov test implemented in [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html).

# %%
#Your code here

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 6:</b>  
#  
# Interpret the results of the GOF techniques. How does the selected parametric distribution perform?
# </p>
# </div>

# %% [markdown]
# ## 5. Propagating the uncertainty

# %% [markdown]
# Using the fitted distributions, we are going to propagate the uncertainty from $H$ and $T$ to $F_h$ **assuming that $H$ and $T$ are independent**.
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 7:</b>   
#     
# 1. Draw 10,000 random samples from the fitted distribution functions for $H$ and $T$.
#     
# 2. Compute $F_h$ for each pair of samples.
#     
# 3. Compute $F_h$ for the observations.
#     
# 4. Plot the PDF and exceedance curve in logscale of $F_h$ computed using both the simulations and the observations.
# </p>
# </div>

# %%
# Here, the solution is shown for the Lognormal distribution

# Draw random samples
rs_H = #your code here
rs_T = #your code here

#Compute Fh
rs_Fh = #your code here

#repeat for observations
Fh = #your code here

#plot the PDF and the CDF


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 8:</b>   
#
# Interpret the figures above, answering the following questions:
# - Are there differences between the two computed distributions for $F_h$?
# - What are the advantages and disadvantages of using the simulations?
# </p>
# </div>

# %% [markdown]
# If you run the code in the cell below, you will obtain a scatter plot of both variables. Explore the relationship between both variables and answer the following questions:
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 9:</b>   
#     
# 1. Observe the plot below. What differences do you observe between the generated samples and the observations?
#     
# 2. Compute the correlation between $H$ and $T$ for the samples and for the observartions. Are there differences?
#     
# 3. What can you improve into the previous analysis? Do you have any ideas/suggestions on how to implement those suggestions?
# </p>
# </div>

# %%
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_H, rs_T, 40, 'k', label = 'Simulations')
axes.scatter(H, T, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

# %%
#Correlation coefficient calculation here

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
