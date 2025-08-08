<userStyle>Normal</userStyle>

# Group Assignment 1.7: Distribution Fitting

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 7, Friday Oct 18, 2024.*


## Case 3: Discharges on a structure

**What's the propagated uncertainty? *How large will be the discharge?***

In this project, you have chosen to work on the uncertainty of water depths ($h$) and water velocities ($u$) on top of a hydraulic structure to estimate the discharge. You have observations from physical experiments of waves impacting a breakwater during a wave storm scaled up to prototype scale. You can further read on the dataset [here](https://doi.org/10.1016/j.coastaleng.2024.104483). Remember that the discharge can be computed as

$$
q = u S
$$

where $S$ is the section the flow crosses. Thus, assuming a discharge width of 1m, we can simplify the previous equation as

$$
q = u h 
$$

**The goal of this project is:**
1. Choose a reasonable distribution function for $u$ and $h$.
2. Fit the chosen distributions to the observations of $u$ and $h$.
3. Assuming $d$ and $h$ are independent, propagate their distributions to obtain the distribution of $q$.
4. Analyze the distribution of $q$.


## Importing packages

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})
```

## 1. Explore the data


First step in the analysis is exploring the data, visually and through its statistics.

```python
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
```

```python
# Statistics for h

print(stats.describe(h))
```

```python
# Statistics for u

print(stats.describe(u))
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1:</b>  
 
Describe the data based on the previous statistics:
    <li>Which variable presents a higher variability?</li>
    <li>What does the skewness coefficient means? Which kind of distribution functions should we consider to fit them?</li>
</p>
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <b>Solution:</b>

- $h$ presents a higher variance while $h$ and $u$ have a similar mean. Then, if we compute the coefficient of variation to standardize that variability, we obtain $CV(h)=0.130/1.211 = 0.107$ and $CV(u)= 0.092/1.464 = 0.063$. Thus, $h$ has higher variability than $u$.</li>
- Both $h$ and $u$ have a positive non-zero skewness, being the one for $u$ significantly higher. Thus, the data presents a right tail and mode < median < mean. An appropriate distribution for $h$ and $u$ would be one which: (1) it is bounded in 0 (no negative values of $h$ or $u$ are physically possible), and (2) has a positive tail. If we consider the distributions that you have been introduced to, Lognormal, Gumbel or Exponential would be a possibility. Also, Gaussian distribution might be a possibility for $h$ as the skewness is relatively low and might not be significant.</li>
</div>
</div>


## 2. Empirical distribution functions


Now, we are going to compute and plot the empirical PDF and CDF for each variable. Note that you have the pseudo-code for the empirical CDF in the [reader](https://mude.citg.tudelft.nl/book/probability/empirical.html).


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2:</b>   

Define a function to compute the empirical CDF. Plot the empirical PDF and CDF.
</p>
</div>

```python
def ecdf(var):
    x = np.sort(var) # sort the values from small to large
    n = x.size # determine the number of datapoints\
    y = np.arange(1, n+1) / (n+1)
    return [y, x]
```

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(h, edgecolor='k', linewidth=0.2, 
             color='cornflowerblue', label='Water depth, h (m)', density = True)
axes[0].set_xlabel('Random variable (X)')
axes[0].set_ylabel('density')
axes[0].hist(u, edgecolor='k', linewidth=0.2, alpha = 0.5, 
             color='grey', label='Water velocity, u (m/s)', density = True)
axes[0].set_title('PDF', fontsize=18)
axes[0].grid()
axes[0].legend()

axes[1].step(ecdf(h)[1], ecdf(h)[0], 
             color='cornflowerblue', label='Water depth, h (m)')
axes[1].set_xlabel('Random variable (X)')
axes[1].set_ylabel('${P[X \leq x]}$')
axes[1].step(ecdf(u)[1], ecdf(u)[0], 
             color='grey', label='Water velocity, u (m/s)')
axes[1].set_title('CDF', fontsize=18)
axes[1].legend()
axes[1].grid()
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3:</b>   

Based on the results of Task 1 and the empirical PDF and CDF, select <b>one</b> distribution to fit to each variable. For $h$, select between Uniform or Gaussian distribution, while for $u$ choose between Exponential or Gumbel.
</p>
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <b>Solution:</b>

$h$: Gaussian
$u$: Gumbel
</div>
</div>


## 3. Fitting a distribution


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 4:</b>   

Fit the selected distributions to the observations using MLE.
</p>
</div>

Hint: Use [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html) built in functions (watch out with the parameters definition!).

```python
params_h = stats.norm.fit(h)
params_u = stats.gumbel_r.fit(u)
```

## 4. Assessing goodness of fit


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 5:</b>   

Assess the goodness of fit of the selected distribution using:
    <li> One graphical method: QQplot or Logscale. Choose one.</li>
    <li> Kolmogorov-Smirnov test.</li>
</p>
</div>

Hint: You have Kolmogorov-Smirnov test implemented in [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html).

```python
#Graphical method

#Logscale

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].step(ecdf(h)[1], 1-ecdf(h)[0], 
             color='k', label='Water depth, h')
axes[0].plot(ecdf(h)[1], 1-stats.norm.cdf(ecdf(h)[1], *params_h),
             '--', color = 'grey', label='Gaussian')
axes[0].set_xlabel('Water depth, h (m)')
axes[0].set_ylabel('${P[X > x]}$')
axes[0].set_title('h', fontsize=18)
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(u)[1], 1-ecdf(u)[0], 
             color='k', label='Water velocity, u')
axes[1].plot(ecdf(u)[1], 1-stats.gumbel_r.cdf(ecdf(u)[1], *params_u),
             '--', color = 'grey', label='Gumbel')
axes[1].set_xlabel('Water velocity, u (m/s)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('u', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()
```

```python
# QQplot

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot([trunc(min(h)), ceil(max(h))], [trunc(min(h)), ceil(max(h))], 'k')
axes[0].scatter(ecdf(h)[1], stats.norm.ppf(ecdf(h)[0], *params_h), 
             color='grey', label='Gaussian')
axes[0].set_xlabel('Observed h (m)')
axes[0].set_ylabel('Estimated h (m)')
axes[0].set_title('h', fontsize=18)
axes[0].set_xlim([trunc(min(h)), ceil(max(h))])
axes[0].set_ylim([trunc(min(h)), ceil(max(h))])
axes[0].legend()
axes[0].grid()

axes[1].plot([trunc(min(u)), ceil(max(u))], [trunc(min(u)), ceil(max(u))], 'k')
axes[1].scatter(ecdf(u)[1], stats.gumbel_r.ppf(ecdf(u)[0], *params_u), 
             color='grey', label='Gumbel')
axes[1].set_xlabel('Observed u (m/s)')
axes[1].set_ylabel('Estimated u (m/s)')
axes[1].set_title('u', fontsize=18)
axes[1].set_xlim([trunc(min(u)), ceil(max(u))])
axes[1].set_ylim([trunc(min(u)), ceil(max(u))])
axes[1].legend()
axes[1].grid()
```

```python
#KStest

_, p_h = stats.kstest(h,stats.norm.cdf, args=params_h)
_, p_u = stats.kstest(u,stats.gumbel_r.cdf, args=params_u)

print('The p-value for the fitted Gaussian distribution to h is:', round(p_h, 3))
print('The p-value for the fitted Gumbel distribution to u is:', round(p_u, 3))
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 6:</b>   

Interpret the results of the GOF techniques. How does the selected parametric distribution perform?
</p>
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <b>Solution:</b>
    
- Logscale plot: This technique allows to visually assess the fitting of the parametric distribution to the tail of the empirical distribution. For both $h$ and $u$, Gaussian and Gumbel distributions performs well even in the tail of the distribution. For $h$, high values start to deviate from the Gaussian distribution, indicating that for lower non-exceedance probabilities it might not be a good fit. </li>
- QQplot: Similar conclusions to those for Logscale can be derived.</li>
- Kolmogorov-Smirnov test: remember that the test statistic measures the difference between two distributions. The p-value then represents the probability of observing a difference at least that large for a sample from the assumed distribution. If p-value is lower than the significance ($\alpha=0.05$, for instance), the null hypothesis is rejected. Considering here $\alpha=0.05$, we can accept that the variable $h$ comes from a Gaussian distribution and that $u$ comes from a Gumbel distribution.</li>
</div>
</div>


## 5. Propagating the uncertainty


Using the fitted distributions, we are going to propagate the uncertainty from $h$ and $u$ to $q$ **assuming that $h$ and $u$ are independent**.

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 7:</b>   
    
1. Draw 10,000 random samples from the fitted distribution functions for $h$ and $u$.
    
2. Compute $q$ for each pair of samples.
    
3. Compute $q$ for the observations.
    
4. Plot the PDF and exceedance curve in logscale of $q$ computed using both the simulations and the observations.
</p>
</div>

```python
# Here, the solution is shown for the Lognormal distribution

# Draw random samples
rs_h = stats.norm.rvs(*params_h, size = 10000)
rs_u = stats.gumbel_r.rvs(*params_u, size = 10000)

#Compute Fh
rs_q = rs_h * rs_u

#repeat for observations
q = h * u

#plot the PDF and the CDF
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
axes[0].hist(rs_q, edgecolor='k', linewidth=0.2, density = True, label = 'From simulations')
axes[0].hist(q, edgecolor='k', facecolor = 'orange', alpha = 0.5, linewidth=0.2, 
             density = True, label = 'From observations')
axes[0].set_xlabel('Discharge (m3/s)')
axes[0].set_ylabel('density')
axes[0].set_title('PDF', fontsize=18)
axes[0].legend()
axes[0].grid()

axes[1].step(ecdf(rs_q)[1], 1-ecdf(rs_q)[0], label = 'From simulations')
axes[1].step(ecdf(q)[1], 1-ecdf(q)[0], color = 'orange', label = 'From observations')
axes[1].set_xlabel('Discharge (m3/s)')
axes[1].set_ylabel('${P[X > x]}$')
axes[1].set_title('Exceedance plot', fontsize=18)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid()
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 8:</b>  
 
Interpret the figures above, answering the following questions:
- Are there differences between the two computed distributions for $q$?</li>
- What are the advantages and disadvantages of using the simulations?</li>
</p>
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <b>Solution:</b>

- In the PDF plot, we can see that the shape of the distribution is similar for $q$. In the CDF plot, we can see that there are significant differences in the tail of the distribution of $q$, being the values from the observations higher than those computed from the simulations. This is because the Gaussian distribution does not properly fit the tail of the distribution of $h$ and when inferring values with very low exceedance probabilities, that becomes more noticeable. </li>
- <b>Disadvantages:</b> we are assuming that $h$ and $u$ are independent (we will see how to address this issue next week). But is that true? Also, the results are conditioned to how good model is the selected parametric distribution. In this case, since the tail of the distribution of $h$ is not properly fitted, the obtained distribution for $q$ deviates from the one obtained from the observations. Also, some simulated values are negative and, thus, non-physical. That could be corrected using distributions bounded in 0. <li><b>Advantages:</b> I can draw all the samples I want allowing the computation of events I have not observed yet (extreme events).
</div>


If you run the code in the cell below, you will obtain a scatter plot of both variables. Explore the relationship between both variables and answer the following questions:

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 9:</b>   
    
1. Observe the plot below. What differences do you observe between the generated samples and the observations?
    
2. Compute the correlation between $h$ and $u$ for the samples and for the observartions. Are there differences?
    
3. What can you improve into the previous analysis? Do you have any ideas/suggestions on how to implement those suggestions?
</p>
</div>

```python
fig, axes = plt.subplots(1, 1, figsize=(7, 7))
axes.scatter(rs_h, rs_u, 40, 'k', label = 'Simulations')
axes.scatter(h, u, 40, 'r','x', label = 'Observations')
axes.set_xlabel('Water depth, h (m)')
axes.set_ylabel('Flow velocity, u (m/s)')
axes.legend()
axes.grid()
```

```python
#Correlation
correl = stats.pearsonr(h, u)
correl_rs = stats.pearsonr(rs_h, rs_u)
print('The correlation between the observations is:', correl[0])
print('The correlation between the simulations is:', correl_rs[0])
```

<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
    <b>Solution:</b>

- The observations are focussed in an area of the plot while the simulations are spreaded all around. this is because the observations are dependent to each other, there is a physical relationship between the water depth and the velocity of the flow, while the simualtions were assumed to be independent. </li>
- There is a correlation of 0.39 between the observed $h$ and $u$, indicating the physical dependence between the variables. On the contrary, no significant correlation between the generated samples is observed.</li>
- <b>Some suggestions:</b> Improve the fit in the tail of $h$. Account for the dependence between the two variables. </li>
</div>

<!-- #region -->
**End of notebook.**

<div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
  <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
  </div>
  <div style="font-size: 75%; margin-top: 10px; text-align: right;">
    By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
    &copy; 2024 TU Delft. 
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
    <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
