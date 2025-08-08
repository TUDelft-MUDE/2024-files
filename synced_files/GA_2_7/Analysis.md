<userStyle>Normal</userStyle>

# Group Assignment 2.7: Extreme Value Analysis

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Extreme Value Analysis, Week 2.7, Friday, Jan 10, 2025.*


In this project, you will work on the uncertainty of precipitation in Turís, close to Valencia (Spain), where the past 29th October an extreme flood occurred. Turís was the location where the highest rainfall was recorded, being also the record of that station with 771mm in a day.

<img src="https://github.com/patmanas/figs/raw/main/flood_Valencia.avif" style="width:400px" />

You have cumulative daily observations since 1999. The dataset was retrieved from the Ministry of Agriculture [here](https://servicio.mapa.gob.es/websiar/SeleccionParametrosMap.aspx?dst=1).

**The goal of this project is:**
1. Perform Extreme Value Analysis using Yearly Maxima and GEV.
2. Perform Extreme Value Analysis using POT and GPD.
3. Compare the results from both methods and define the return period of the event of October 2024.

_Read the instructions for this project in `README.md` before beginning the analysis._

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats 
from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 14})
```

## Task 1: Import and Explore Data

The first step in the analysis is importing and exploring the data set.

The dataset has two columns: the first one provides the time stamp of the measurements and the last column shows the measured precipitation (mm). We set the first column as a datetime as they are the dates of the measurements.

```python
# Import
P = pd.read_csv('turis.csv', delimiter = ',', parse_dates = True)
P.columns=['Date', 'Prec'] #rename columns
P['Date'] = pd.to_datetime(P['Date'], format='mixed')
P = P.sort_values(by='Date')
P=P.reset_index(drop=True)

P.head()
```

To start the data exploration, we can compute some basic statistics.

```python
# values for markdown report
print(f"{P['Prec'].size:d}",
      f"{P['Prec'].min():.3f}",
      f"{P['Prec'].max():.3f}",
      f"{P['Prec'].mean():.3f}",
      f"{P['Prec'].std():.3f}",
      f"{P['Prec'].isna().sum():d}",
      f"{sum(P['Prec']==0):d}",
      sep=' | ')
```

We can also plot the timeseries. Notice how the event of October 2024 stands out.

```python
#Solution
plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
plt.title('Time series of precipitation');
```

## Task 2: Yearly Maxima & GEV

Let's start with the first EVA analysis!


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.1:</b>   
Sample the data set by finding the yearly maxima and add the sampled extremes to your time series plot as red dots.
</p>
</div>

```python
idx_max = #your code here
YM = P.loc[idx_max]
print('The shape of the sampled extremes is:', YM.shape)

plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(YM['Date'], YM['Prec'], 40, 'r')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.2:</b>   Compute the empirical cumulative distribution function of the sampled maxima. Fit the appropriate extreme value distribution function using <code>scipy.stats</code> and assess the fit via an exceedance plot with semi-log scale.
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p><b>Tip:</b> save the parameters of an instance of <code>rv_continuous</code> as a tuple to make it easier to use the methods of the distribution later.
<br><br>For example, define parameters like this:
<br><code>params = ... .fit( ... )</code>
<br>instead of like this:
<br><code>param1, param2, param3 = ... .fit( ... )</code>
<br>(see WS 2.7 solution for examples).
</p></div>

```python
#Function for the ECDF
def ecdf(var):
    #your code here
    return [y, x]

#Fit the parametric distribution to the extreme observations
params_YM = #your code here
print('GEV parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_YM))

#Compare both the empirical and parametric distributions in semilog scale
#your plot here
```

## Task 3: POT and GPD

Let's go for Peak Over Threshold (POT)---make sure you **read** this section carefully, as there are some important tips for implementing the method correctly!

As described in the book: _when performing EVA with POT, we model **excesses** over the threshold, not the actual value of the extreme event._ In principle, this can simply be described as follows:

$$
X = th + E
$$

where $X$ is the random variable of interest (rain, in this case), $th$ is the threshold and $E$ is the _excess_ (also a random variable). In other words, the distribution-fitting part of our EVA is performed on $E$. However, it is not practical to verify/validate or make inference with our distribution on the random variable $E$ for two key reasons:
1. It is more convenient to think in terms of $X$, and
2. We often need to calibrate the threshold value, which produces countless different sets of sampled extremes, distributions and random variables, $E$, which cannot be compared easily.

As such, we continue to perform our analysis in terms of $X$, while making sure to address the fact that our POT data and distributions are defined in terms of $E$. In practical terms, this means we need to <em>add and subtract the threshold value at certain points in the analysis.</em>

Note: the threshold and declustering time evaluation is outside the scope of MUDE.


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.1:</b>   
Apply POT on the timeseries using a declustering time of 48 hours and a threshold of 40mm. Plot the sampled extremes on the timeseries as blue dots.
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">Hint: you can use the function <code>find_peaks</code> from Scipy, as done in PA 2.7.</div>

```python
threshold = 40

distance = 2 #days
peaks, _ = #your code here
print('The shape of the sampled extremes is:', peaks.shape)

plt.figure(figsize=(10, 6))
plt.plot(P['Date'], P['Prec'],'k')
plt.scatter(P.iloc[peaks, 0], P.iloc[peaks, 1], 40, 'cornflowerblue')
plt.xlabel('Time')
plt.ylabel('Precipitation [mm]')
plt.grid()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.2:</b>   Compute the empirical cumulative distribution function of the sampled maxima. Fit the appropriate extreme value distribution function using <code>scipy.stats</code> and assess the fit via an exceedance plot with semi-log scale.
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">Hint: you need to fit the appropriate distribution with a location parameter of 0 on the excesses. You can force the fitting of a distribution with a given value of the parameters using the keyword argument <code>floc</code>.</div>

```python
params_POT = #your code here
print('GPD parameters are: {:.3f}, {:.3f}, {:.3f}'.format(*params_POT))

#your plot here
```

```python
# Time series plot with both sets of extremes
# For Report_solution.md

#your plot here
```

## Task 4: Computing Return Levels

In this section, we are going to use the distributions found in the previous two Tasks to compute the return periods that correspond to the flood of October 2024 and compare the results that both EVA methods provide.


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4.1:</b> create the return level plot for both EVA approaches (values of the random variable in the x-axis and return period on the y-axis; the y-axis in log scale). Consider return periods up to at least 800 years. Add also the return level plot generated using the empirical CDF for both sampling approaches. Plot a line to indicate the magnitude of the event of October 2024 (771mm). 
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>Remember you can use tuple unpacking as an argument for methods of <code>scipy.stats.rv_continuous</code>, like this: <code>*params</code> (see WS 2.7 solution for examples).</p></div>

```python
# YOUR_CODE_HERE (more than one line!)

# plt.figure(figsize=(10, 6))
# plt.plot(YM_range, RT_range, 'r', label = 'YM&GEV')
# plt.plot(POT_range, RT_range, 'cornflowerblue', label = 'POT&GPD')
# YOUR_FIGURE_FORMATTING_CODE_HERE (more than one line!)
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4.2:</b> What would be the return period associated with the event of October 2024? Compare the answer provided by both approaches and reflect on these differences.
</p>
</div>

```python
#YM&GEV
YM_RT = #return period obtained with YM

#POT&GPD
average_n_excesses = #average numbr of excesses per year
POT_prob = #non exceedance probability
RT_POT = #return period obtained with POT

#Print
print(f'The return period obtained with YM+GEV is {YM_RT:.3f}\n'
          f'The return period obtained with POT+GPD is {RT_POT:.3f}\n')
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
    By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
    &copy; 2024 TU Delft. 
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
    <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
  </div>
</div>


<!--tested with WS_2_8_solution.ipynb-->
<!-- #endregion -->
