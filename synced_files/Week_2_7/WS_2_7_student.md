<userStyle>Normal</userStyle>

# Workshop 2.7: Extreme temperature

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Extreme Value Analysis, Week 2.7, Wednesday, Jan 8, 2024.*


In this session, you will work with the uncertainty of extreme temperatures in the airport of Barcelona to assess the extreme loads induced by temperature in a steel structure in the area. You have daily observations of the maximum temperature for several years. The dataset was retrieved from the Spanish Agency of Metheorology [AEMET](https://www.aemet.es/es/portada). Your goal is to design the structure for a _lifespan of 50 years_ with a _probability of failure of 0.1_ during the design life. 

**The goal of this project is:**
1. Compute the required design return period for the steel structure.
2. Perform monthly Block Maxima and fit the a distribution to the sampled observations.
3. Assess the Goodness of fit of the distribution.
4. Compute the return level plot.
5. Compute the design return level.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats 
from math import ceil, trunc

plt.rcParams.update({'font.size': 14})
```

## Task 1: Data Exploration

First step in the analysis is exploring the data, visually and through its statistics. We import the dataset and explore its columns.

```python
T = pd.read_csv('temp.csv', delimiter = ',', parse_dates = True).dropna().reset_index(drop=True)
T.columns=['Date', 'T'] #rename columns
T.head()
```

The dataset has two columns: the time stamp of the measurements and the cumulative daily precipitation. We set the first columns as a datetime as they are the dates of the measurements.

```python
T['Date'] = pd.to_datetime(T['Date'], format='mixed')
T['Date']
```

Once formatted, we can plot the timeseries and the histogram.

```python
fig, axes = plt.subplots(1,2, figsize=(12,5), layout='constrained')

axes[0].hist(T['T'], label = 'T', density = True, edgecolor = 'darkblue')
axes[0].set_xlabel('PDF')
axes[0].set_xlabel('Maximum daily temperature [degrees]')
axes[0].grid()
axes[0].legend()
axes[0].set_title('(a) Histogram')

axes[1].plot(T['Date'], T['T'],'k', label = 'P')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Maximum daily temperature [degrees]')
axes[1].grid()
axes[1].set_title('(b) Time series')
axes[1].legend()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.1:</b>   
Based on the above plots, briefly describe the data.
</p>
</div>


## Task 2: Sample Monthly Maxima


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.2:</b>   
Sample monthly maxima from the timeseries and plot them on the timeseries. Plot also the histograms of both the maxima and the observations.
</p>
</div>

```python
# Extract year and month from the Date column
T['Year'] = T['Date'].dt.year
T['Month'] = T['Date'].dt.month

# Group by Year and Month, then get the maximum observation
idx_max = #your code here
max_list = T.loc[idx_max]
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.3:</b>   
Look at the previous plots. Are the sampled maxima independent and identically distributed? Justify your answer. What are the implications for further analysis?
</p>
</div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.3:</b>   
Compute PDF and the empirical cumulative distribution function of the observations and the sampled monthly maxima. Plot the ECDF in log-scale. Where are the sampled monthly maxima located with respect with the CDF of all the observations?
</p>
</div>

```python
def ecdf(var):
    #your code here
    return [y, x]

#your plot here
```

## Task 3: Distribution Fitting

We did this a lot at the end of Q1---refer to your previous work as needed!


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3:</b>   
Fit a distribution to the monthly maxima. Print the values of the obtained parameters and interpret them:
<ol>
    <li>Do the location and scale parameters match the data?</li>
    <li>According to the shape parameter, what type of distribution is this?</li>
    <li>What type of tail does the distribution have (refer to EVA Chapter 7.2 of the book)?</li>
    <li>Does the distribution have an upper bound? If so, compute it!</li>
</ol>
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>Use <a href="https://docs.scipy.org/doc/scipy/reference/stats.html" target="_blank">scipy.stats</a> built in functions (watch out with the parameter definitions!), similar to Week 1.7 and use the DataFrame created in Task 2.
</p></div>

```python
params_T = #your code here
print(params_T)
```

## Task 4: Goodness of Fit


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4:</b>   
Assess the goodness of fit of the selected distribution using the exceedance probability plot in semi-log scale.
    
Consider the following questions:
<ol>
    <li>How well do the probabilities of the fitted distribution match the empirical distribution? Is there an over- or under-prediction?</li>
    <li>Is the tail type of this GEV distribution appropriate for the data?</li>

</ol>

</p>
</div>

```python
#your code here
```

## Task 5: Return Levels

It was previously indicated that the structure should be designed for a lifespan of 50 years with a probability of failure of 0.1 along the design life.


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5.1:</b>   

Compute the design return period using both the Binomial and Poisson model for extremes. Compare the obtained return.
</p>
</div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5.2:</b>   

Considering that you have sampled monthly maxima, compute and plot the return level plot: values of the random variable in the x-axis and return periods on the y-axis. Y-axis in logscale.
</p>
</div>

```python
RT_range = #range of values of return level
monthly_probs = #compute monthly probabilities
eval_nitrogen = #compute the values of the random variable for those probabilities

plt.figure(figsize=(10, 6))
plt.plot(eval_nitrogen, RT_range, 'k')
plt.xlabel('Temperature [mm]')
plt.ylabel('RT [years]')
plt.yscale('log') 
plt.grid()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5.3:</b>   
Compute the design value of temperature. Choose the return level you prefer within the Poisson and Binomial model.
</p>
</div>

```python
RT_design = #value of the return period
monthly_design = #compute the monthly probability
design_T = #compute the design value

print('The design value of temperature is:',
      f'{design_T:.3f}')
```

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
    &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. 
    This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
