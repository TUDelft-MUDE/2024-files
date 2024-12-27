<userStyle>Normal</userStyle>

---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
---

# GA 1.8: Multivariate Distributions

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px; height: auto; margin: 0" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px; height: auto; margin: 0" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.8, Friday Oct 25, 2024.*


## Objective

This notebook is structured in an identical way to WS 1.8. In particular, we have 3 main tasks:
1. Build a multivariate distribution $F_{X_1,X_2}(x_1,x_2)$
2. Use the distribution by comparing empirical probabilities to values computed with $F_{X_1,X_2}(x_1,x_2)$ for the AND, OR and function of random variable cases
3. Validate the distribution using the distribution of the function of random variables, $Z(X_1,X_2)$.

### Multivariate Distribution (Task 1)

As with WS 1.8, we will build a multivariate distribution, but this time we will include non-Gaussian marginal distributions using a Gaussian Copula:

$$
F_{X_1,X_2}(x_1,x_2) = C[F_{X_1}(x_1)F_{X_2}(x_2)]
$$

In this case, the Copula requires one parameter: the Pearson correlation coefficient, $\rho$.

This distribution has been implemented in the class `Bivariate` within `helper.py`. You can define an instance of the class using:
```
my_bivariate_dist = Bivariate(marginal_1, marginal_2, rho)
```
where the arguments are the marginal distributions for your random variables and are instances of the class `rv_continuous` from `scipy.stats`. In fact, the `Bivariate` class has been created with similar methods to `multivariate_normal`; in other words, you can use the methods `pdf`, `cdf` and `rvs` in the same way as `multivariate_normal` in WS 1.8.

Note that the function `plot_contours` will also work, but due to the way `Bivariate` is implemented, it is slow to make the plots, so you may want to use small sample sizes.

#### Python Package: `pyvinecopulib`

A package `pyvinecopulib` is required for the Bivariate class. It is only available on PyPI, so it has to be installed using `pip`. Fortunately you should already have done this as part of PA 1.8, so all you need to do is remember to use your environment `mude-week-8`.

### Probability Calculations for 3 Cases (Task 2)

For each data set, we will use the 90th percentile of each random variable to evaluate the AND, OR and function of random variable cases. In other words:

$$
x_{1,90} = F_{X_1}^{-1}(0.90) \;\;\textrm{and}\;\; x_{2,90} = F_{X_2}^{-1}(0.90)
$$

Since there were three data sets to choose from last week, each with different functions and variables, this notebook uses the notation $Z(X_1,X_2)$ to represent the function of random variables for your particular case. As we did in WS 1.8, we would like to evaluate $Z$ for a threshold condition (in WS 1.8 it was $Z<0$ for the Thingamajig). For this assignment, consider the threshold case to be all combinations of $x_1$ and $x_2$ such that:

$$
Z>Z(x_{1,90},z_{2,90})
$$

### Propagating Uncertainty (Task 3)

This proceeds as in WS 1.8 as well, where you will use the samples generated in Task 2 to evaluate the distribution of $Z$ and compare to the empirical data set.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from helper import plot_contour
from helper import Bivariate
```

## Part 1: Creating a Bivariate Distribution

We need to represent our two dependent random variables with marginal distributions and use the correlation coefficient to model dependence, as found previously in GA 1.7.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1:</b>   

Import your data set from last week and define the marginal distributions as frozen scipy stats objects.

<em>The cell below illustrates how to create a frozen Gaussian distribution</em> 
</p>
</p>
</div>


<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
Remember there is a <a href="https://mude.citg.tudelft.nl/2024/book/probability/python.html" target="_blank">new page on scipy stats in the book</a> in case you need a reference.
</p>
</div>

```python
# YOUR_CODE_HERE # many lines
# parameters1 = st.norm.fit_loc_scale(data_x1)
# dist_x1 = st.norm(*parameters1)
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2:</b>   
Compute the covariance and correlation coefficient between the two random variables. Print the results.
</p>
</div>

```python
# YOUR_CODE_HERE # many lines
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3:</b>   
Build the bivariate distribution by instantiating the <code>Bivariate</code> class in <code>helper.py</code> (and described above). To validate the result, create a plot that shows contours of the joint PDF, compared with the data (see note below). Include the data in your plot and write a few comments on the quality of the fit for use in your Report.
</p>
</div>

<!-- #region id="0491cc69" -->
<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"> <p>Use the helper function <code>plot_contour</code> in <code>helper.py</code>; it works exactle the same as in WS 1.8.</em></p></div>
<!-- #endregion -->

```python
# plot_contour? # uncomment and run to read docstring
```

```python
bivar_dist = Bivariate(YOUR_CODE_HERE, YOUR_CODE_HERE, YOUR_CODE_HERE)

plot_contour(YOUR_CODE_HERE, YOUR_CODE_HERE, data=data)
```

## Part 2: Using the Bivariate Distribution

Now that we have the distribution, we will use it compute probabilities related to the three cases, presented above, as follows:

1. $P[X_1>x_{1,90} \;\cup\; X_2>x_{2,90}$
2. $P[X_1>x_{1,90} \;\cap\; X_2>x_{2,90}]$
3. $P[X_1,X_2:\; Z>Z(x_{1,90},x_{2,90})]$

Note that the "critical" Z value in this case is that where your Z function is evaluated at the 90th percentile values of each random variable.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2:</b>
    
For each of the three cases, do the following:

- Compute the requested probability using the empirical distribution.
- Compute the requested probability using the bivariate distribution.
- Create a bivariate plot that includes PDF contours <em>and</em> the region of interest.
- Repeat the calculations for additional cases of correlation coefficient (for example change $\rho$ to: +0.9, 0.0, then -0.9) to see how the answer changes (you can simply regenerate the plot, you don't need to make multiple versions). <em>You can save this sub-task for later if you are running out of time. It is more important to get through Task 3 during the in-class session.</em>
- Write two or three sentences that summarize the results and explains the quantitative impact of correlation coefficient. Make a particular note about whether or not one case may or be affected more or less than the others.

</p>
</div>

<!-- #region id="0491cc69" -->
<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"> <p>Note that the optional arguments in the helper function <code>plot_contour</code> will be useful here.

Here is an example code that shows you what it can do (the values are meaningless)
</p></div>
<!-- #endregion -->

```python
region_example = np.array([[0, 5, 12, 20, 28, 30],
                           [5, 20, 0, 18, 19, 12]])

plot_contour(bivar_dist, [0, 30, 0, 30],
             case=[23, 13],
             region=region_example);

```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 2.1 and 2.2:</b> create cells below to carry out the OR and AND calculations.
</p>
</div>

```python
# YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 2.3:</b> create cells below to carry out the Case 3 calculations.

Note that in this case you need to make the plot to visualize the region over which we want to integrate. We need to define the boundary of the region of interest by solving the equation $Z(X_1,X_2)$ for $X_2$ when $Z=Z(x_{1,90},x_{2,90})$.
</p>
</div>


The equation can be defined as follows:

$$
\textrm{WRITE THE EQUATION HERE}
$$

which is then defined in Python and included in the `plot_contours` function as an array for the keyword argument `region`.

```python
# YOUR_CODE_HERE
# DEFINITELY more than one line.
# probably several cells too ;)
```

## Part 3: Validate Bivariate with Monte Carlo Simulation

Now that we have seen how the different cases give different values of probability, let's focus on the function of random variables. This is a more interesting case because we can use the samples of $Z$ to approximate the distribution $f_Z(z)$ and use the empirical distribution of $Z$ to help validate the bivariate model.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

$\textbf{Task 3:}$
    
Do the following:

- Use Monte Carlo Simulation to create a sample of $Z(X_1,X_2)$ and compare this distribution to the empirical distribution.</li>
- Write 2-3 sentences assessing the quality of the distribution from MCS, and whether the bivariate distribution is acceptable for this problem. Use qualitative and quantitative measures from last week to support your arguments.
    
</p>
<p>
    <em>Note: it would be interesting to fit a parametric distribution to the MCS sample, but it is not required for this assignment.</em>
</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 3.1:</b> Plot histograms of $Z$ based on the Monte Carlo samples, and based on the data. Note that you probably already computed the samples in Part 2.
</p>
</div>

```python
plot_values = np.linspace(sample_Z.min(), sample_Z.max(), 30)
fig, ax = plt.subplots(1)
ax.hist([YOUR_CODE_HERE, YOUR_CODE_HERE],
         bins=plot_values,
         density=True,
         edgecolor='black');
ax.legend(['Data', 'MCS Sample'])
ax.set_xlabel('$Z(X_1,X_2)$')
ax.set_xlabel('Empirical Density [--]')
ax.set_title('Comparison of Distributions of $Z$');
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 3.2:</b> Define a function to compute the ecdf.
</p>
</div>

```python
def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints
    y = np.arange(1, n+1) / (n + 1)
    return [y, x]

```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 3.3:</b> Create a semi-log plot of the non-exceedance probability.
</p>
</div>

```python
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
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>

<b>Task 3.4:</b> Create a semi-log plot of the exceedance probability.
</p>
</div>

```python
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
```

<!-- #region id="0491cc69" -->
<div style="background-color:#facb8e; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px"><p>In case you are wondering, the data for this exercise was computed with a Clayton Copula. A Copula is a useful way of modelling non-linear dependence. If you would like to learn more about this, you should consider the 2nd year cross-over module CEGM2005 Probabilistic Modelling of real-world phenomena through ObseRvations and Elicitation (MORE).</p></div>
<!-- #endregion -->

**End of notebook.**
<h2 style="height: 60px">
</h2>
<h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    
</h3>
<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
