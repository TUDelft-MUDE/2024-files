# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %% [markdown] id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff"
# # WS 1.8: The Thingamajig!
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.8. For: 23 October, 2024.*

# %% [markdown]
# ## Objective
#
# This workshop focuses fundamental skills for building, using and understanding multivariate distributions: in particular when our variables are no longer statistically independent.
#
# For our case study we will use a Thingamajig: an imaginary object for which we have limited information. One thing we _do_ know, however, is that it is very much influenced by two random variables, $X_1$ and $X_2$: high values for these variables can cause the Thingamajig to fail. We will use a multivariate probability distribution to compute the probability of interest under various _cases_ (we aren't sure which one is relevant, so we consider them all!). We will also use a comparison of distributions drawn from our multivariate probability model with the empirical distributions to validate the model.
#
# ### Multivariate Distribution
#
# In this assignment we will build a multivariate distribution, which is defined by a probability density function. From now on, we will call it _bivariate_, since there are only two random variables:
#
# $$
# f_{X_1,X_2}(x_1,x_2)
# $$
#
# This distribution is implemented in `scipy.stats.multivariate_normal`. The bivariate normal distribution is defined by 5 parameters: the parameters of the Gaussian distribution for $X_1$ and $X_2$, as well as the correlation coefficient between them, $\rho_{X_1,X_2}$. In this case we often refer to $X_1$ and $X_2$ as the marginal variables (univariate) and the bivariate distribution as the joint distribution. We will use the bivariate PDF to create contour plots of probability density, as well as the CDF to evaluate probabilities of different cases:
#
# $$
# F_{X_1,X_2}(x_1,x_2)
# $$
#
#
# ### Cases
#
# We will consider three different cases and see how the probability of interest is different for each, as well as how they are influenced by the dependence structure of the data. The cases are described here; although they vary slightly, they have something in common: _they are all integrals of the bivariate PDF over some domain of interest $\Omega$._
#
#
# #### Case 1: Union (OR)
#
# The union case is relevant if the Thingamajig fails when either or both random variable exceeds a specified value:
#
# $$
# P[X_1>x_1 \;\cup\; X_2>x_2]
# $$
#
# This is also called the "OR" probability because it considers either one variable _or_ the other _or_ both exceeding a specified value.
#
# #### Case 2: Intersection (AND)
#
# The intersection case is relevant if the Thingamajig fails when the specified interval for each random variable are exceeded together:
#
# $$
# P[X_1>x_1 \;\cap\; X_2>x_2]
# $$
#
# This is also called the "AND" probability because it considers _both_ variables exceeding a specified value.
#
# #### Case 3: Function of Random Variables 
#
# Often it is not possible to describe a region of interest $\Omega$ as a simple union or intersection probability. Instead, there are many combinations of $X_1$ and $X_2$ that define $\Omega$. If we can integrate the probability density function over this region we can evaluate the probability.
#
# Luckily, it turns out there is some extra information about the Thingamajig: a function that describes some aspect of its behavior that we are very very interested in:
#
# $$
# Z(X_{1},X_{2}) = 800 - X_{1}^2 - 20X_{2}
# $$
#
# where the condition in which we are interested occurs when $Z(X_{1},X_{2})<0$. Thus, the probability of interest is:
#
# $$
# P[X_1,X_2:\; Z<0]
# $$
#
# #### Evaluating Probabilities in Task 2
#
# Cases 1 and 2 can be evaluated with the bivariate cdf directly because the integral bounds are relatively simple (be aware that some arithmetic and thinking is required, it's not so simple as `multivariate.cdf()`).
#
# Case 3 is not easy to evaluate because it must be integrated over a complicated region. Instead, we will approximate the integral numerically using _Monte Carlo simulation_ (MCS). This is also how we will evaluate the distribution of the function of random variables in Task 3. Remember, there are four essential steps to MCS:
#
# 1. Define distributions for random variables (probability density over a domain)  
# 2. Generate random samples  
# 3. Do something with the samples (deterministic calculation)  
# 4. Evaluate the results: e.g., “empirical” PDF, CDF of samples, etc.
#
# _Note that as we now have a multivariate distribution we can no longer sample from the univariate distributions independently!_
#
# ### Validating the Bivariate Distribution in Task 3
#
# The final task of this assignment is to use the distribution of the function of random variables (univariate) to validate the bivariate distribution, by comparing the empirical distribution to our model. Once the sample is generated, it involves the same goodness of fit tools that we used last week.
#

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour

# %% [markdown]
# ## Part 1: Creating a Bivariate Distribution
#
# We need to represent our two dependent random variables with a bivariate distribution; a simple model is the bivariate Gaussian distribution, which is readily available via `scipy.stats.multivariate_normal`. To use it in this case study, we first need to check that the marginal distributions are each Gaussian, as well as compute the covariance and correlation coefficient.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1:</b>   
# Import the data in <code>data.csv</code>, then find the parameters of a normal distribution to fit to the data for each marginal. <em>Quickly</em> check the goodness of fit and state whether you think it is an appropriate distribution (we will keep using it anyway, regardless of the answer).
# <p>
# <em>Don't spend more than a few minutes on this, you should be able to quickly use some of your code from last week.</em>
# </p>
# </p>
# </div>

# %%
data = np.genfromtxt('data.csv', delimiter=";")
data.shape

# %%
YOUR_CODE_HERE # probably many lines


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.2:</b>   
# Write two functions to compute the covariance and correlation coefficient between the two random variables. Print the results.
#
# <b>The input arguments should be Numpy arrays and you should calculate each value without using a pre-existing Python package or method.</b>
# </p>
# </div>

# %%
def calculate_covariance(X1, X2):
    '''
    Covariance of two random variables X1 and X2 (numpy arrays).
    '''
    YOUR_CODE_HERE # may be more than one line
    return covariance

def pearson_correlation(X1, X2):
    YOUR_CODE_HERE # may be more than one line
    return correl_coeff


# %%
covariance = calculate_covariance(data_x1, data_x2)
print(f'The covariance of X1 and X2 is {covariance:.5f}')
correl_coeff = pearson_correlation(data_x1, data_x2)
print(f'The correlation coefficient of X1 and X2 is {correl_coeff:.5f}')

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3:</b>   
# Build the bivariate distribution using <code>scipy.stats.multivariate_normal</code> (as well as the mean vector and covariance matrix). To validate the result, create a plot that shows contours of the joint PDF, compared with the data (see note below). Comment on the quality of the fit in 2-3 sentences or bullet points.
# </p>
# </div>

# %% [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"> <p>Use the helper function <code>plot_contour</code> in <code>helper.py</code>; it was already imported above. Either look in the file to read it, or view the documentation in the notebook with <code>plot_contour?</code></p>
#
# <p><em>Hint: for this Task use the optional </em><code>data</code><em> argument!.</em></p></div>

# %%
# plot_contour? # uncomment and run to read docstring

# %%
mean_vector = YOUR_CODE_HERE
cov_matrix = YOUR_CODE_HERE
bivar_dist = YOUR_CODE_HERE

# %%
plot_contour(YOUR_CODE_HERE, [0, 30, 0, 30], data=YOUR_CODE_HERE);

# %% [markdown]
# ## Part 2: Using the Bivariate Distribution
#
# Now that we have the distribution, we will use it compute probabilities related to the three cases, presented above, as follows:
#
# 1. $P[X_1>20 \;\cup\; X_2>20]$
# 2. $P[X_1>20 \;\cap \ X_2>20]$
# 3. $P[X_1,X_2:\; Z<0]$

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2:</b>
#     
# For each of the three cases, do the following:
#
# - Compute the requested probability using the empirical distribution.
# - Compute the requested probability using the bivariate distribution.
# - Create a bivariate plot that includes PDF contours <em>and</em> the region of interest.
# - Repeat the calculations for additional cases of correlation coefficient (for example change $\rho$ to: +0.9, 0.0, then -0.9) to see how the answer changes (you can simply regenerate the plot, you don't need to make multiple versions). <em>You can save this sub-task for later if you are running out of time. It is more important to get through Task 3 during the in-class session.</em>
# - Write two or three sentences that summarize the results and explains the quantitative impact of correlation coefficient. Make a particular note about whether or not one case may or be affected more or less than the others.
#
# </p>
# </div>

# %% [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"> <p>Note that the optional arguments in the helper function <code>plot_contour</code> will be useful here--<b>also for the Project on Friday!</b>
#
# Here is an example code that shows you what it can do (the values are meaningless)
# </p></div>

# %%
region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 2.1 and 2.2:</b> create cells below to carry out the OR and AND calculations.
# </p>
# </div>

# %%
YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 2.3:</b> create cells below to carry out the Case 3 calculations.
#
# Note that in this case you need to make the plot to visualize the region over which we want to integrate. We need to define the boundary of the region of interest by solving the equation $Z(X_1,X_2)$ for $X_2$ when $Z=0$.
# </p>
# </div>

# %% [markdown]
# The equation can be defined as follows:
#
# $$
# \textrm{WRITE THE EQUATION HERE}
# $$
#
# which is then defined in Python and included in the `plot_contours` function as an array for the keyword argument `region`.

# %%
YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)

# %% [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"> <p>Note: the bivariate figures are an important concept for the exam, so if using the code is too difficult for you to use when studying on your own, try sketching it on paper.</p></div>

# %% [markdown]
# ## Part 3: Validate Bivariate with Monte Carlo Simulation
#
# Now that we have seen how the different cases give different values of probability, let's focus on the function of random variables. This is a more interesting case because we can use the samples of $Z$ to approximate the distribution $f_Z(z)$ and use the empirical distribution of $Z$ to help validate the bivariate model.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# $\textbf{Task 3:}$
#     
# Do the following:
#
# - Use Monte Carlo Simulation to create a sample of $Z(X_1,X_2)$ and compare this distribution to the empirical distribution.</li>
# - Write 2-3 sentences assessing the quality of the distribution from MCS, and whether the bivariate distribution is acceptable for this problem. Use qualitative and quantitative measures from last week to support your arguments.
#
# </p>
# <p>
#     <em>Note: it would be interesting to fit a parametric distribution to the MCS sample, but it is not required for this assignment.</em>
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 3.1:</b> Plot histograms of $Z$ based on the Monte Carlo samples, and based on the data. Note that you probably already computed the samples in Part 2.
# </p>
# </div>

# %%
plot_values = np.linspace(-100, 1000, 30)
fig, ax = plt.subplots(1)
ax.hist([YOUR_CODE_HERE, YOUR_CODE_HERE],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 3.2:</b> Define a function to compute the ecdf.
# </p>
# </div>

# %%
def ecdf(var):
    x = YOUR_CODE_HERE # sort the values from small to large
    n = YOUR_CODE_HERE # determine the number of datapoints
    y = YOUR_CODE_HERE
    return [y, x]


# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 3.3:</b> Create a semi-log plot of the non-exceedance probability.
# </p>
# </div>

# %%
fig, axes = plt.subplots(1, 1, figsize=(8, 5))

axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
          color='k', label='Data')
axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE,
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel('CDF, $\mathrm{P}[Z < z]$')
axes.set_title('Comparison: CDF (log scale expands lower tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
#
# <b>Task 3.4:</b> Create a semi-log plot of the exceedance probability.
# </p>
# </div>

# %%
fig, axes = plt.subplots(1, 1, figsize=(8, 5))

axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE, 
          color='k', label='Data')
axes.step(YOUR_CODE_HERE, YOUR_CODE_HERE,
          color='r', label='MCS Sample')
axes.set_xlabel('$Z(X_1,X_2)$')
axes.set_ylabel('Exceedance Probability, $\mathrm{P}[Z > z]$')
axes.set_title('Comparison: CDF (log scale expands upper tail)')
axes.set_yscale('log')
axes.legend()
axes.grid()

# %% [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"><p>In case you are wondering, the data for this exercise was computed with a Clayton Copula. A Copula is a useful way of modelling non-linear dependence. If you would like to learn more about this, you should consider the 2nd year cross-over module CEGM2005 Probabilistic Modelling of real-world phenomena through ObseRvations and Elicitation (MORE).</p></div>

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
