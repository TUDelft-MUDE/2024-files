# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: multi_more
#     language: python
#     name: multi_more
# ---

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

# + [markdown] id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945"
# ## Case 3: Discharges on a structure
#
# **What's the propagated uncertainty? *How large will be the discharge?***
#
# In this project, you have chosen to work on the uncertainty of water depths ($h$) and water velocities ($u$) on top of a hydraulic structure to estimate the discharge. You have observations from physical experiments of waves impacting a breakwater during a wave storm scaled up to prototype scale. You can further read on the dataset [here](https://doi.org/10.1016/j.coastaleng.2024.104483). Remember that the discharge can be computed as
#
# $$
# q = u S
# $$
#
# where $S$ is the section the flow crosses. Thus, assuming a discharge width of 1m, we can simplify the previous equation as
#
# $$
# q = u h 
# $$
#
# **The goal of this project is:**
# 1. Choose a reasonable distribution function for $d$ and $h$.
# 2. Fit the chosen distributions to the observations of $d$ and $h$.
# 3. Assuming $d$ and $h$ are independent, propagate their distributions to obtain the distribution of $q$.
# 4. Analyze the distribution of $q$.

# + [markdown] id="d33f1148-c72b-4c7e-bca7-45973b2570c5"
# ## Importing packages

# + id="4fc6e87d-c66e-43df-a937-e969acc409f8"
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})
# -

# ## 1. Explore the data

# First step in the analysis is exploring the data, visually and through its statistics.

# +
# Import
h, u = np.genfromtxt('dataset_hu.csv', delimiter=",", unpack=True, skip_header=True)

# plot time series
fig, ax = plt.subplots(2, 1, figsize=(10, 7), layout = 'constrained')
ax[0].plot(h,'k')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Water depth, h (m)')
ax[0].grid()

ax[1].plot(u,'k')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Water velocity, u (m/s)')
ax[1].grid()

# +
# Statistics for h

print(stats.describe(h))

# +
# Statistics for u

print(stats.describe(u))


# + [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1:</b>  
#  
# Describe the data based on the previous statistics:
#     <li>Which variable presents a higher variability?</li>
#     <li>What does the skewness coefficient means? Which kind of distribution functions should we consider to fit them?</li>
# </p>
# </div>
# -

# ## 2. Empirical distribution functions

# Now, we are going to compute and plot the empirical PDF and CDF for each variable. Note that you have the pseudo-code for the empirical CDF in the [reader](https://mude.citg.tudelft.nl/book/probability/empirical.html).

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2:</b>   
#
# Define a function to compute the empirical CDF.
# </p>
# </div>

def ecdf(YOUR_INPUT:
    #Your code
    return YOUR_OUTPUT


# +
#Your plot

# + [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3:</b>   
#
# Based on the results of Task 1 and the empirical PDF and CDF, select <b>one</b> distribution to fit to each variable. For $h$, select between Uniform or Gaussian distribution, while for $u$ choose between Exponential or Gumbel.
# </p>
# </div>
# -

# ## 3. Fitting a distribution

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 4:</b>   
#
# Fit the selected distributions to the observations using MLE.
# </p>
# </div>
#
# Hint: Use [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html) built in functions (watch out with the parameters definition!).

# +
#Your code here
# -

# ## 4. Assessing goodness of fit

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

# +
#Your code here
# -

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 6:</b>   
#
# Interpret the results of the GOF techniques. How does the selected parametric distribution perform?
# </p>
# </div>

# ## 5. Propagating the uncertainty

# Using the fitted distributions, we are going to propagate the uncertainty from $h$ and $u$ to $q$ **assuming that $h$ and $u$ are independent**.
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 7:</b>   
#     
# 1. Draw 10,000 random samples from the fitted distribution functions for $h$ and $u$.
#     
# 2. Compute $q$ for each pair of samples.
#     
# 3. Compute $q$ for the observations.
#     
# 4. Plot the PDF and exceedance curve in logscale of $q$ computed using both the simulations and the observations.
# </p>
# </div>

# +
# Here, the solution is shown for the Lognormal distribution

# Draw random samples
rs_h = #Your code here
rs_u = #Your code here

#Compute Fh
rs_q = #Your code here

#repeat for observations
q = #Your code here

#plot the PDF and the CDF
# -

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 8:</b>  
#  
# Interpret the figures above, answering the following questions:
# - Are there differences between the two computed distributions for $q$?</li>
# - What are the advantages and disadvantages of using the simulations?</li>
# </p>
# </div>

# If you run the code in the cell below, you will obtain a scatter plot of both variables. Explore the relationship between both variables and answer the following questions:
#
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 9:</b>   
#     
# 1. Observe the plot below. What differences do you observe between the generated samples and the observations?
#     
# 2. Compute the correlation between $h$ and $u$ for the samples and for the observartions. Are there differences?
#     
# 3. What can you improve into the previous analysis? Do you have any ideas/suggestions on how to implement those suggestions?
# </p>
# </div>

fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_h, rs_u, 40, 'k', label = 'Simulations')
axes.scatter(h, u, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Wave height, H (m)')
axes.set_ylabel('Wave period, T (s)')
axes.legend()
axes.grid()

# +
#Correlation coefficient calculation here
# -

# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png"/>
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png"/>
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
