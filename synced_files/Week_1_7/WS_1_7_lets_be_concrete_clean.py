# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: mude-base
#     language: python
#     name: python3
# ---

# # WS 1.7: Modelling Uncertain Concrete Strength
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.7. Due: October 16, 2024.*

# + [markdown] id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945"
# Assessing the uncertainties in the compressive strength of the produced concrete is key for the safety of infrastructures and buildings. However, a lot of boundary conditions influence the final resistance of the concrete, such the cement content, the environmental temperature or the age of the concrete. Probabilistic tools can be applied to model this uncertainty. In this workshop, you will work with a dataset of observations of the compressive strength of concrete (you can read more about the dataset [here](https://www.kaggle.com/datasets/gauravduttakiit/compressive-strength-of-concrete)). 
#
# **The goal of this project is:**
# 1. Choose a reasonable distribution function for the concrete compressive strength analyzing the statistics of the observations.
# 2. Fit the chosen distributions by moments.
# 3. Assess the fit computing probabilities analytically.
# 4. Assess the fit using goodness of fit techniques and computer code.
#
# The project will be divided into 3 parts: 1) data analysis, 2) pen and paper stuff (math practice!), and 3) programming.

# + id="4fc6e87d-c66e-43df-a937-e969acc409f8"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})
# -

# ## Part 1: Explore the data

# First step in the analysis is exploring the data, visually and through its statistics.

# +
# Import
data = np.genfromtxt('dataset_concrete.csv', delimiter=",", skip_header=True)

# Clean
data = data[~np.isnan(data)]

# plot time series
plt.figure(figsize=(10, 6))
plt.plot(data,'ok')
plt.xlabel('# observation')
plt.ylabel('Concrete compressive strength [MPa]')
plt.grid()

weights = 5*np.ones(len(data))
plt.hist(data, orientation='horizontal', weights=weights, color='lightblue', rwidth=0.9)
# -

# In the figure above, you can see all the observations of concrete compressive strength. You can see that there is no clear pattern in the observations. Let's see how the statistics look like!

# +
# Statistics

df_describe = pd.DataFrame(data)
df_describe.describe()


# + [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1:</b>   
#     Using <b>ONLY</b> the statistics calculated in the previous lines:
#     <li>Choose an appropriate distribution to model the data between the following: (1) Gumbel, (2) Uniform, and (3) Gaussian. </li>
#     <li>Justiy your choice.</li>
# </p>
# </div>
# -

# _Your answer here._

# ## Part 2: Use pen and paper!

# Once you have selected the appropriate distribution, you are going to fit it by moments manually and check the fit by computing some probabilities analytically. Remember that you have all the information you need in the textbook. Do not use any computer code for this section, you have to do in with pen and paper. You can use the notebook as a calculator.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1:</b>   
# Fit the selected distribution by moments.
# </p>
# </div>

# _Your answer here._

# We can now check the fit by computing manually some probabilities from the fitted distribution and comparing them with the empirical ones.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2:</b>   
# Check the fit of the distribution:
#     <li>Use the values obtained from the statistical inspection: the min, 25%, 50%, 75% and max values. What non-exceedance probabilities correspond to those values?</li>
#     <li>Compute the values of the random variable corresponding to those probabilities using the fitted distribution.</li>
#     <li>Compare the obtained values with the empirical ones and assess the fit.</li>
# </p>
# You can summarize you answers in the following table (report your values with 3-4 significant digits max, as needed).
# </div>
#

# |   |Minimum value|P25%|P50%|P75%|Maximum value|
# |---|-------------|----|----|----|-------------|
# |Non-exceedance probability| |  |  |  |  |
# |Empirical quantiles| | | | | |
# |Predicted quantiles||||||

# ## Part 3: Let's do it with Python!

# Now, let's assess the performance using further goodness of fit metrics and see whether they are consistent with the previously done analysis.

# + [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.1:</b>   
# Prepare a function to compute the empirical cumulative distribution function.
# </p>
# </div>
# -

def ecdf(YOUR_CODE_HERE):
    YOUR_CODE_HERE # may be more than one line
    return YOUR_CODE_HERE


# + [markdown] id="bfadcf3f-4578-4809-acdb-625ab3a71f27"
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.2:</b>   
# Transform the fitted parameters for the selected distribution to loc-scale-shape.
# </p>
# </div>
#
# Hint: the distributions are in our online textbook, but it is also critical to make sure that the formulation in the book is identical to that of the Python package we are using. You can do this by finding the page of the relevant distribution in the [Scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) documentation.
# -

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.3:</b>   
# Assess the goodness of fit of the fitted distribution by:
#     <li> Comparing the empirical and fitted PDF.</li>
#     <li> Using the exceedance plot in log-scale.</li>
#     <li> Using the QQplot.</li>
#     <li> Interpret them. Do you reach a conclusion similar to that in the previous section?</li>
# </p>
# </div>
#
# Hint: Use [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html) built in functions (watch out with the parameters definition!).

# +
loc = YOUR_CODE_HERE
scale = YOUR_CODE_HERE

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.hist(YOUR_CODE_HERE,
          edgecolor='k', linewidth=0.2, color='cornflowerblue',
          label='Empirical PDF', density = True)
axes.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
          'k', linewidth=2, label='YOUR_DISTRIBUTION_NAME_HERE PDF')
axes.set_xlabel('Compressive strength [MPa]')
axes.set_title('PDF', fontsize=18)
axes.legend()

# +
fig, axes = plt.subplots(1, 1, figsize=(10, 5))

axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
          color='k', label='Empirical CDF')
axes.plot(YOUR_CODE_HERE, YOUR_CODE_HERE,
          color='cornflowerblue', label='YOUR_DISTRIBUTION_NAME_HERE CDF')
axes.set_xlabel('Compressive strength [MPa]')
axes.set_ylabel('${P[X > x]}$')
axes.set_title('Exceedance plot in log-scale', fontsize=18)
axes.set_yscale('log')
axes.legend()
axes.grid()

# +
fig, axes = plt.subplots(1, 1, figsize=(10, 5))

axes.plot([0, 120], [0, 120], 'k')
axes.scatter(YOUR_CODE_HERE, YOUR_CODE_HERE, 
             color='cornflowerblue', label='Gumbel')
axes.set_xlabel('Observed compressive strength [MPa]')
axes.set_ylabel('Estimated compressive strength [MPa]')
axes.set_title('QQplot', fontsize=18)
axes.set_xlim([0, 120])
axes.set_ylim([0, 120])
axes.set_xticks(np.arange(0, 121, 20))
axes.grid()
# -

# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
