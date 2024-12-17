---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
---

# GA 2.4: Beary Icy

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.4, Time Series Analysis. For: December 6, 2024*


Winter is coming and it is time to start getting our models ready for the ice classic. Our first goal is to improve the temperature model, as that seems to be an important factor in determining breakup day. Temperature is notoriously hard to predict, but we can analyze historical data to get a better understanding of the patterns.

In this assignment we will analyze a time series from a **single year**; in fact, only the **first 152 days of the year**, from January 1 until June 1. This is the period of interest for the ice classic, as the ice forms in this period, reaching its maximum thickness between January-March, and then starts melting, with breakup day typically happening in April or May.

Remember that we have until April 5 to place a bet. Why, then do we want to fit a model several months beyond this point? This gives us confidence in assessing the ability of the model to predict temperature, so that when we use it on April 5 to make **predictions** about the future, we can understand the uncertainty associated with it.

Let's start by loading the data and plotting it, then we will determine which components should be used to detrend it.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram
```

### Part 1: Load the data and plot it



<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 1.1:</b>   

Do the following:

- load the data
- create time vector
- plot the data

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p><b>Note:</b> the assignment never specified the frequency units; here we use cycles per <b>day</b> as the reference unit. It is also possible to use a different unit, but then your answers will have different values (this will have no effect on the trends used and included in your model).</p></div>

```python
# YOUR_CODE_HERE
# data = YOUR_CODE_HERE # Temperature data
# time_hours = YOUR_CODE_HERE # Time in hours

# SOLUTION
# Reading the data from the file
data = np.loadtxt('temperature.csv')
time_hours = np.arange(0, len(data))
time_days = time_hours / 24
dt = time_days[1] - time_days[0]
fs = 1 / dt

# Plotting the data
plt.figure(figsize=(10, 3))
plt.plot(time_days, data)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
# END SOLUTION BLOCK
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 1.2:</b>   

Use the Markdown cell below to describe the data (you can use a few bullet points). For example, confirm relevant characteristics like number of points, units, describe the values (qualitatively), etc.

</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

- There are 24*152 points (hourly)
- Looks like there is a bit of variation around a central line, but the general increase in temperature throughout the spring is larger than the variation
- the general trend looks linear in the middle, but flattens out at the ends, which is to be expected with a periodic annual cycle
- looks like there is something "funny" happening around day 60 and 120

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>
Here the term "seasonal" is misleading, as there is only one seasonal trend and it is only partly visible (since our data set is only half of a year). The term "periodic" is more appropriate, as it allows for the selection of a period that is not necessarily a year---like the daily cycle!
</p></div>


## Part 2: Extract the Dominant Patterns

We clearly see that the data contains a strong pattern (the general increase in temperature from winter to summer). We will start by fitting a functional model to the data in order to stationarize it. To find the frequency of the seasonal pattern we will use the power spectrum of the data.

We will reuse the function `find_seasonal_pattern` from the workshop.

Remember that for running this function we need to predefine the A-matrix to detrend the data. Since the data only contains the first 5 months of the year, we see that the temperature is increasing over time. What type of model would be most appropriate to remove the seasonal trend? 


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.1:</b>   

Define functions to help carry out this analysis, for example, <code>fit_model</code> and <code>find_frequency</code>.

</p>
</div>

```python
def fit_model(data, time, A, plot=False):
    '''
    Function to find the least squares solution of the data
    data: input data
    time: time vector
    A: A-matrix to fit the data
    plot: boolean to plot the results or not
    '''

    # x_hat = YOUR_CODE_HERE # least squares solution
    # y_hat = YOUR_CODE_HERE # model prediction
    # e_hat = YOUR_CODE_HERE # residuals

    # SOLUTION
    x_hat = np.linalg.solve(A.T @ A, A.T @ data)
    y_hat = A @ x_hat
    e_hat = data - y_hat
    # END SOLUTION BLOCK

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, data, label='Data')
        plt.plot(time, y_hat, label='Estimated data')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [°C]')
        plt.title('Data vs Estimated data')
        plt.grid(True)
        plt.legend()
        plt.subplot(212)
        plt.plot(time, e_hat, label='Residuals')
        plt.xlabel('Time [days]')
        plt.ylabel('Temperature [°C]')
        plt.title('Residuals')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    return x_hat, y_hat, e_hat

def find_frequency(data, time, A, fs, plot=True):
    '''
    Function to find the dominant frequency of the signal
    data: input data
    time: time vector
    A: A-matrix to detrend the data (prior to spectral analysis)
    fs: sampling frequency
    plot: boolean to plot the psd or not
    '''
    # Detrending the data
    _, _, e_hat= fit_model(data, time, A)

    N = len(data)

    # Finding the dominant frequency in e_hat
    # freqs, pxx = periodogram(YOUR_CODE_HERE, fs=YOUR_CODE_HERE, window='boxcar',
    #                          nfft=N, return_onesided=False,
    #                          scaling='density')
    
    # SOLUTION
    # Finding the dominant frequency in e_hat
    freqs, pxx = periodogram(e_hat, fs=fs, window='boxcar',
                                nfft=N, return_onesided=False,
                                scaling='density')
    # END SOLUTION BLOCK

    # finding the dominant frequency and amplitude
    # Note: there are many ways to do this
    # amplitude = YOUR_CODE_HERE # Amplitude of the dominant frequency
    # dominant_frequency = YOUR_CODE_HERE # Dominant frequency

    # SOLUTION
    # finding the dominant frequency and amplitude
    dominant_frequency, amplitude = freqs[np.argmax(pxx)], np.max(pxx)
    # END SOLUTION BLOCK

    # Plotting the PSD
    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(211)
        plt.plot(time, e_hat)
        plt.title('Residuals')
        plt.ylabel('Temperature [°C]')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(freqs[freqs>0], pxx[freqs>0], label='PSD of residuals')
        plt.xlabel('Frequency')
        plt.ylabel('PSD')
        plt.title('Power Spectral Density')
        plt.grid(True)
        plt.plot(dominant_frequency, amplitude, 'ro', label='Dominant Frequency')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()

    return dominant_frequency

```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.2:</b>   

Now provide an A-matrix that removes the trend from the data. There are multiple answers that will work, but some are better than others.

First, use the Markdown cell below to define your A-matrix and include a brief explanation justifying your choice.
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

- The best model (chosen a priori) would be an <b>annual</b> periodic signal, as we know the temperature behaves this way.
- A linear model may approximate the increasing temperature well, but it is not going to fit the ends well.
- It turns out a power law model gives the best quantitative fit, but this would also be problematic if we need to extrapolate beyond the data range.

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>
It is very likely that you left a linear trend term in your A matrix from the WS, especially when copying and pasting. This in the end would not have had a big effect on the resulting model (the x value would be very small), but it is physically not relevant in this temperature application. It <em>would</em>, however, be good to include the intercept term!</p></div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.3:</b>   

Now define the A-matrix in code and extract the seasonal pattern. Continue extracting components until the time series is stationary (you will then summarize your findings in the next task).
</p>
</div>

```python
# YOUR_CODE_HERE

# SOLUTION
A = np.column_stack((np.ones(len(data)), np.cos(2*np.pi*time_days/365), np.sin(2*np.pi*time_days/365)))
dom_f = find_frequency(data, time_days, A, fs=fs)
print(f'Dominant Frequency: {dom_f:.2f}')

# check whether there is still a significant frequency in the residuals
find_frequency(data, time_days, np.column_stack((A, np.cos(2*np.pi*time_days), np.sin(2*np.pi*time_days))) , fs=fs, plot=True)
# END SOLUTION BLOCK
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.4:</b>   

Describe how you have detrended the time series. Include at least: a) the number and types of components used (and their parameters; in task 2.5 you will print those), b) how you decided to stop extracting components.
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

- We have removed a yearly and daily trend as well as an intercept.
- No linear trend can be detected, observation period is too short
- We stopped when the power spectrum showed no significant peaks

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>
You may have considered additional frequencies besides the daily and annual cycles. The PSD values were much lower, but these additional terms did a very good job of matching the data. However, the model would not be very useful for additional year, or outside the scope of the data. In fact, this is a form of "overfitting" (we will discuss overfitting more in week 2.6), which is obvious since the addition of 2 parameters with the offset is better than the 4 parameters with 2 additional frequencies. Note also that there is no physical explanation for a commonly found frequency of ~2 cycles per year.
</p></div>


## Fitting the Functional Model

In the next cell we will fit the model to generate stationary residuals. Above, you may have a periodic signal, where for each dominant frequency $f_i$ ($i=1,2$) the model is:

$$a_i  \cos(2\pi f_i  t) + b_i  \sin(2\pi f_i t)$$ 

However, to report the periodic signals we would like to have the amplitude, phase shift and the frequency of those signals, which can be recovered from:
$$A_i  \cos(2\pi f_i  t + \theta_i)$$
Where the amplitude $A_i = \sqrt{a_i^2 + b_i^2}$ and $\theta_i = \arctan(-b_i/a_i)$

Note: in Section 4.1 book this was shown where the angular frequency $\omega = 2\pi f$ was used.



<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.5:</b>   

Complete the code cell below to create the functional model.

</p>
</div>

```python
def rewrite_seasonal_comp(ak, bk):
    '''
    Function to rewrite the seasonal component in terms of sin and cos
    ak: seasonal component coefficient for cos
    bk: seasonal component coefficient for sin

    returns: Ak, theta_k
    '''
    # YOUR_CODE_HERE

    # SOLUTION
    Ak = np.sqrt(ak**2 + bk**2)
    theta_k = np.arctan2(-bk, ak)
    return Ak, theta_k
    # END SOLUTION BLOCK

# creating the A matrix of the functional model
# A = YOUR_CODE_HERE
# x_hat, y_hat, e_hat = YOUR_CODE_HERE


# SOLUTION
A = np.column_stack((np.ones(len(data)),
                        np.cos(2*np.pi*1*time_days), np.sin(2*np.pi*1*time_days),
                        np.cos(2*np.pi*time_days/365), np.sin(2*np.pi*time_days/365)))

x_hat, y_hat, e_hat0 = fit_model(data, time_days, A)
# END SOLUTION BLOCK

# Plotting the data and the estimated trend
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()

# Plotting the residuals
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat0)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Residuals')
plt.grid(True)

# Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE


# SOLUTION
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION BLOCK

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2.6:</b>   

Are the residuals stationary? State yes or no and describe why in the cell below.

</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

No! There is definitely some strange behavior in the residual; it looks like an offset is present around 60-70 days and a again (downward) around 120 days. There seems to also be a U-shaped pattern, but it is hard to tell if that is significant while the offset is still there.

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>Note that for Part 3, the results will be different than those in the solution depending on which periodic terms you have included.</p></div>


## Part 3: Finding the grizzly

When we look at the residuals after removing the periodic pattern(s), we see that there is still a pattern in the data. From researchers in the Nenana area we have heard that there is a grizzly bear that likes to take a nap (hibernate) in the area. We suspect that the grizzly bear has slept too close to the temperature sensor and has influenced the data. 

In the next cell we will write an offset detection algorithm to find the offset in the data. The offset detection algorithm is based on the likelihood ratio test framework. However, due to the presence of autocorrelation in the residuals, the traditional critical values for the likelihood ratio test are not valid. Therefore, we will use a bootstrap approach to estimate the critical values. Luckily, this is **not** the first time we had to remove a grizzly bear from our data, so we know that the estimated critical values is approximately 100 (i.e. you do not have to find this value yourself!).

## The offset detection algorithm
The offset detection algorithm is based on the likelihood ratio test framework. The likelihood ratio test has a test statistic that is given by:

$$\Lambda = n \log \left( \frac{SSR_0}{SSR_1} \right)$$

$$SSR_i = \sum_{i=1}^n (\hat{e}_i)^2$$

where $SSR_0$ is the sum of the squared residuals for the model without an offset, $SSR_1$ is the sum of the squared residuals for the model with an offset, and $n$ is the number of data points. The likelihood ratio test statistic is compared to a critical value to determine if an offset is present in the data.

The cell below defines several functions which roughly accomplish the following:
 
1. Calculate the sum of the squared residuals for the model without an offset, $SSR_0$.
2. Calculate the sum of the squared residuals for the model with an offset at each possible point, $SSR_1$.
   1. For each possible offset location, we will calculate the sum of the squared residuals for the model with an offset at that data point.
   2. The A-matrix for the model with an offset is the same as the A-matrix for the model without an offset, but with an additional column that is 0 till the data point and 1 after the data point.
3. At each possible offset location, calculate the likelihood ratio test statistic and store it in the `results` vector.
4. We will find the offset location that maximizes the likelihood ratio test statistic, i.e. the location where an offset is *most* likely.
5. We will include the offset in the model and repeat the process until the likelihood ratio test statistic is below the critical value.



<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.1:</b>   

Using the description above and the comments and docstring in the code, fill in the code below to complete the offset detection algorithm.
</p>
</div>

```python
def A1_matrix(A0, break_point):
    '''
    Function to create the A1 matrix
    A0: A matrix under H0
    break_point: break point location
    return: A1 matrix
    A
    '''
    # create the new column and stack it to the A0 matrix
    # YOUR_CODE_HERE
    
    # SOLUTION
    new_col = np.zeros(A0.shape[0])
    new_col[break_point:] = 1
    A1 = np.column_stack((A0, new_col))
    # END SOLUTION BLOCK
    return A1


def LR(e0, e1, cv=100, verbose=True):
    '''
    Function to perform the LR test
    e0: residuals under H0
    e1: residuals under H1
    cv: critical value
    '''
    # n = YOUR_CODE_HERE
    # SSR0 = YOUR_CODE_HERE
    # SSR1 = YOUR_CODE_HERE
    # test_stat = YOUR_CODE_HERE
    
    # SOLUTION
    n = len(e0)
    SSR0 = e0.T @ e0
    SSR1 = e1.T @ e1
    test_stat = n*np.log(SSR0 / SSR1)
    # END SOLUTION

    if test_stat > cv:
        if verbose:
            print(f'Test Statistic: {test_stat:.3f} > Critical Value: {cv:.3f}')
            print('Reject the null hypothesis')
    else:
        if verbose:
            print(f'Test Statistic: {test_stat:.3f} < Critical Value: {cv:.3f}')
            print('Fail to reject the null hypothesis')
    return test_stat

def jump_detection(data, time, A, cv=100, plot=True):
    '''
    Function to detect the jump in the data
    data: input data
    time: time vector
    A: A matrix under H0
    cv: critical value
    plot: boolean to plot the results or not
    '''
    # initialize the results vector
    # results = YOUR_CODE_HERE
    # find the residuals under H0
    # YOUR_CODE_HERE

    # SOLUTION
    results = np.zeros(len(data))
    _, _, e_hat0 = fit_model(data, time, A)
    # END SOLUTION BLOCK

    # loop over the data points
    for i in range(1, len(data)):
        # create the A1 matrix
        # A1 = YOUR_CODE_HERE

        # SOLUTION
        A1 = A1_matrix(A, i)
        # END SOLUTION BLOCK

        # We need this statement to avoid singular matrices
        if np.linalg.matrix_rank(A1) < A1.shape[1]:
            pass
        else:
            # find the residuals under H1
            # _, _, e_hat1 = YOUR_CODE_HERE
            # test_stat = YOUR_CODE_HERE
            # results[i] = YOUR_CODE_HERE

            # SOLUTION
            _, _, e_hat1 = fit_model(data, time, A1)
            test_stat = LR(e_hat0, e_hat1, verbose=False)
            results[i] = test_stat
            # END SOLUTION BLOCK

    results = np.array(results)
    # finding the offset location. 
    # Offset is the location where the test statistic is maximum

    # location = YOUR_CODE_HERE
    # value = YOUR_CODE_HERE

    # SOLUTION
    location = np.argmax(results)
    value = results[location]
    # END SOLUTION BLOCK

    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(time, results)
        plt.plot(time[location], value, 'ro', label='offset location')
        plt.plot([0, max(time)], [cv, cv], 'k--', label='Critical Value')
        plt.xlabel('Time [days]')
        plt.ylabel('Test Statistic')
        plt.title('LR Test')
        plt.grid(True)
        plt.legend()

    return location, value

```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.2:</b>   

Before we implement the offset detection algorithm use the following Markdown cell to describe the following in a few sentences or bullet points:

How is this process similar to the one we used to find a periodic pattern? How is it different?
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

- We remove in a iterative way the most significant pattern in the data
- When detected, we include the pattern in the model and repeat the process
- Stopping criteria are different, however we could have implemented Likelihood Ratio Test for the periodic pattern as well
- Offset detection is based on LR test, periodic pattern detection is based on power spectrum
</p>
</div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.3:</b>   

Now we will implement the offset detection algorithm by using the functions defined above to find the offset in the data. The function will provide figures from which you will be able to determine the offset.
</p>
</div>

```python
# YOUR_CODE_HERE

# SOLUTION
A_offset = A.copy()

while True:
    break_point, test_stat = jump_detection(data, time_days, A_offset)
    print(f'Break Point day: {break_point/24} with : {test_stat:.2f}')
    if test_stat < 100:
        break
    A_offset = A1_matrix(A_offset, break_point) 
# END SOLUTION BLOCK
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.4:</b>   

Write your chosen offset in the cell below (report both the size and location of the offset). 

</p>
</div>


<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>
We made a mistake and asked you this a bit too early, since we did not yet run the offset detection algorithm. You can see the solution later where the parameters of the functional model are printed.
</p></div>


My offset is: ...


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.5:</b>   

Once you have found the offset, identify the offset location and update your A-matrix to include it in the model. Then repeat the process until the likelihood ratio test statistic is below the critical value.
</p>
</div>

```python
# A2 = YOUR_CODE_HERE
# x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

# SOLUTION
A2 = A_offset
x_hat, y_hat, e_hat = fit_model(data, time_days, A2)
# END SOLUTION (PART 1 of 2)

# Plotting the data and the estimated trend
plt.figure(figsize=(10, 3))
plt.plot(time_days, data, label='Original data')
plt.plot(time_days, y_hat, label='Estimated trend')
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Temperature data Nenana, Alaska')
plt.grid(True)
plt.legend()

# Plotting the residuals
plt.figure(figsize=(10, 3))
plt.plot(time_days, e_hat)
plt.xlabel('Time [days]')
plt.ylabel('Temperature [°C]')
plt.title('Residuals')
plt.grid(True)

# Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE

# SOLUTION
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION BLOCK

print(f'Estimated Parameters:')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3.6:</b>   

Use the Markdown cell below to summarize the location and size of the offset(s) you have found. Include the number of components used in the final model.
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

Should be 2 locations, at 58 and 120, with a value of +4.5 and -6 respectively. The final model should have 5 components:
intercept, yearly trend, daily trend, offset at 58 and offset at 120. 

Daily trend is one component, yet it has 2 parameters (amplitude and phase shift), same for the yearly trend.

</p>
</div>


## Part 4: Analyzing the residuals
Now that we have our residuals we can fit an AR model to the residuals. We will start by plotting the ACF of the residuals. We will then fit an AR model to the residuals and report the parameters of the AR model. Using the likelihood ratio test framework we will determine the order of the AR model.

```python
# Lets start with the ACF plot
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat, ax=ax, lags=20);
ax.grid()
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4.1:</b>   

Begin by completing the functions below to define AR(1) (hint: you did this on Wednesday).

</p>
</div>

```python
def AR1(s, time, plot=True):
    '''
    Function to find the AR(1) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    # y = YOUR_CODE_HERE
    # y_lag_1 = YOUR_CODE_HERE
    # A = np.atleast_2d(y_lag_1).T
    # x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

    # SOLUTION
    y = s[1:]
    y_lag_1 = s[:-1]
    A = np.atleast_2d(y_lag_1).T
    x_hat, y_hat, e_hat = fit_model(y, time, A)
    # END SOLUTION

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[1:], y, label='Original Residuals')
        ax[0].plot(time[1:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [°C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()
        
    print(f'Estimated Parameters:')
    print(f'phi = {x_hat[0]:.4f}')

    return x_hat, e_hat

# Estimating the AR(1) model
# phi_hat_ar1, e_hat_ar1 = AR1(YOUR_CODE_HERE)

# SOLUTION
phi_hat_ar1, e_hat_ar1 = AR1(e_hat, time_days)
# END SOLUTION




```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4.2:</b>   

- As you can see, the next task asks you to implement AR(2). State why this is necessary, using the results from the cell above.
- Based on the ACF plot, will the $\phi_2$ parameter in the AR(2) be positive or negative? Why? 
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

- ACF at lag 1 not zero (or not within the confidence interval). So still some autocorrelation left AR(1) is not sufficient.
- When we try to fit an AR(1) model to a higher order AR process, the AR(1) coefficient will try to capture the effect of the higher order AR process.
  - If $\phi_2$ is positive, the AR(1) estimation will overestimate the effect which will lead to a negative autocorrelation at lag 1 in the residuals of the AR(1) model.
  - If $\phi_2$ is negative, the AR(1) estimation will underestimate the effect which will lead to a positive autocorrelation at lag 1 in the residuals of the AR(1) model.
  - Therefore, $\phi_2$ will most likely be positive.	

</p>
</div>


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4.3:</b>   

Now complete the functions to set up AR(2).

</p>
</div>

```python
def AR2(s, time, plot=True):
    '''
    Function to find the AR(2) model of the given data
    s: input data
    return: x_hat, e_hat
    '''
    # y = YOUR_CODE_HERE
    # y_lag_1 = YOUR_CODE_HERE
    # y_lag_2 = YOUR_CODE_HERE
    # A = YOUR_CODE_HERE
    # x_hat, y_hat, e_hat = fit_model(YOUR_CODE_HERE)

    # SOLUTION
    y = s[2:]
    y_lag_1 = s[1:-1]
    y_lag_2 = s[:-2]
    A = np.column_stack((y_lag_1, y_lag_2))
    x_hat, y_hat, e_hat = fit_model(y, time, A)
    # END SOLUTION

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        ax[0].plot(time[2:], y, label='Original Residuals')
        ax[0].plot(time[2:], y_hat, label='Estimated Residuals')
        ax[0].set_xlabel('Time [days]')
        ax[0].set_ylabel('Temperature [°C]')
        ax[0].set_title('Original Data vs Estimated Data')
        ax[0].grid(True)
        ax[0].legend()
        plot_acf(e_hat, ax=ax[1], lags=20)
        ax[1].grid()
        fig.tight_layout()

    print(f'Estimated Parameters:')
    print(f'phi_1 = {x_hat[0]:.4f}, phi_2 = {x_hat[1]:.4f}')

    return x_hat, e_hat

# Estimating the AR(2) model
# phi_hat_ar2, e_hat_ar2 = AR2(YOUR_CODE_HERE)

# SOLUTION
phi_hat_ar2, e_hat_ar2 = AR2(e_hat0, time_days)
# END SOLUTION

```

## Part 5: Report the Results

_Note: you did this on Wednesday! It was optional then, so you are not expected to know this for the exam; however, you should implement the code using the WS as a template, and your interpretation at the end will be part of the grade for this assignment._

Now that we have found the periodic signals in the data and fitted an AR model to the residuals, we can report the results. By combining including the AR (noise) process, we get residuals that are white noise. When the model has white noise residuals, we can also report the confidence intervals of the model. The estimated variance is only consistent when the residuals are white noise.

We will use the unbiased estimate of the variance of the residuals to calculate the confidence intervals. The unbiased estimate of the variance is given by:

$$\hat{\sigma}^2 = \frac{1}{n-p} \sum_{t=1}^{n} \hat{e}_t^2$$

Where $n$ is the number of observations and $p$ is the number of parameters in the model.

The covariance matrix of the parameters is given by:

$$\hat{\Sigma} = \hat{\sigma}^2 (\mathbf{A}^T \mathbf{A})^{-1}$$

Where $\mathbf{A}$ is the design matrix of the model.

<!-- #region -->


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p> <b>Task 5.1:</b>   
<p>
Complete the missing parts of the code cell below. Note that you will need to add one additional term, compared to Wednesday.
</p>
</div>
<!-- #endregion -->

<div style="background-color:#ffa6a6; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"><p>
As you can see here we use <code>e_hat</code> in the A matrix. This is how the AR model is formulated. Note also that this means the previous steps of running the function <code>AR2</code> were technically not required, as we could have just started adding AR terms until the ACF went to 0. However, this is a simplified time series, created from an AR(2) process; <em>real</em> life is more complex, so the previous steps of checking different AR models is necessary. Especially because we would typically also check alternative models, for example moving average (MA), and perhaps even combining the two into an ARMA model.
</p></div>

```python
# combine ar2 and functional model

# A_final = YOUR_CODE_HERE
# x_hat, y_hat, e_hat_final = fit_model(YOUR_CODE_HERE)

# SOLUTION
A_final = np.column_stack((A2[2:], e_hat[1:-1], e_hat[:-2]))
x_hat, y_hat, e_hat_final = fit_model(data[2:], time_days[2:], A_final, plot=True)
# END SOLUTION

# Plotting the acf of the residuals

# fig, ax = plt.subplots(1, 1, figsize=(10, 3))
# plot_acf(YOUR_CODE_HERE, ax=ax, lags=20);
# ax.grid()

# SOLUTION
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_acf(e_hat_final, ax=ax, lags=20);
ax.grid()
# END SOLUTION

# # compute the standard errors
# N = YOUR_CODE_HERE
# p = YOUR_CODE_HERE
# sigma2 = YOUR_CODE_HERE
# Cov = YOUR_CODE_HERE
# se = YOUR_CODE_HERE

# # Extracting the seasonal component coefficients from the estimated parameters
# a_i = YOUR_CODE_HERE
# b_i = YOUR_CODE_HERE
# freqs = YOUR_CODE_HERE

# SOLUTION
# compute the standard errors
N = A_final.shape[0]
p = A_final.shape[1]
sigma2 = np.sum(e_hat_final**2) / (N - p)
Cov = sigma2 * np.linalg.inv(A_final.T @ A_final)
se = np.sqrt(np.diag(Cov))

# Extracting the seasonal component coefficients from the estimated parameters
a_i = np.array([x_hat[1], x_hat[3]])
b_i = np.array([x_hat[2], x_hat[4]])
freqs = np.array([1, 1/365])
# END SOLUTION

# Check if the number of coefficients match the number of frequencies
assert len(a_i) == len(b_i) == len(freqs), 'The number of coefficients do not match'

print(f'Estimated Parameters (standard deviation):')
for i in range(len(x_hat)):
    print(f'x{i} = {x_hat[i]:.3f}\t\t ({se[i]:.3f})')

print('\nThe seasonal component is rewritten as:')
i = 0
for a, b, f in zip(a_i, b_i, freqs):
    A_i, theta_i = rewrite_seasonal_comp(a, b)
    i += 1
    print(f'A_{i} = {A_i:.3f}, theta_{i} = {theta_i:.3f}, f_{i} = {f:.3f}')

```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5.2:</b>   

Now we have the complete functional model. Reflect on it's suitability for capturing the time dependent variation of temperature throughout the spring. Comment specifically on the time series components that were included and which ones have the most significant influence on the result.

Compare your final parameters to the ones you found in the previous tasks (i.e. model without offset, model with offset). Are they similar? If not, why do you think that is?

Comment also on the suitability of this model for predicting the temperature **beyond the betting deadline of April 5**, assuming that you have data up **until** that date. Remember that the ice typically breaks apart 2 to 6 weeks after the betting deadline.
</p>
</div>


_Your answer here._


<div style="background-color:#FAE99E; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Solution:</b>   

Parameters in offset without AR vs with AR: results are similar, including autoreressive terms does not change the model significantly, but it does reduce the variance of the residuals. 

parameters in offset with AR vs model without offset: the parameters are different, as the offset is now included in the model. Daily pattern is not effected, since the offset does not change the daily pattern. The yearly pattern is slightly effected, as the offset changes the overall trend of the data, as well as the intercept.

The model only includes "memory" for 2 points, which is 2 hours. However, from the ACF prior to removing the residuals, we see that the "effect" of the memory lasts around 20 points. This is 20 hours, so the  inclusion of AR(2) is useless and the functional model without these terms would be a fine best estimate for temperature on a given day several weeks in the future.

</p>
</div>


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
