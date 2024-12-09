---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff" -->
# WS 1.2: Mean and Variance Propagation

**Sewer Pipe Flow Velocity**

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.2. Wed Sep 11, 2024.*
<!-- #endregion -->

<!-- #region id="1db6fea9-f3ad-44bc-a4c8-7b2b3008e945" -->
## Overview

In this notebook you will apply the propagation laws for the mean and variance for a function of two independent random variables. You will assess how well the approximations correspond with the <em>simulation-based</em> equivalents. You will also assess the distribution of the function.

_You do not need to turn in this notebook._

### Objectives

1. Observe how uncertainty "propagates" from the inputs to the output of a function by estimating moments of the function of random variables and seeing how they change relative to the moments of the input random variables.
2. Recognize that a non-linear function of random variables that have the (joint) Normal distribution (the inputs) produces a non-Normal random variable (the output).
3. Using _sampling_ (Monte Carlo Simulation) to _validate_ the linearized error propagation technique introduced in the textbook. Specifically, by:
   1. Comparing the estimated moments with that of the sample, and
   2. Comparing the Normal distribution defined by the estimated moments to the sample

### A Note on "Sampling"

We will use Monte Carlo Simulation to create an empirical "sample" of the random values of our function of random variables, $V$ (the output). This is a commonly used approach widely used in science and engineering applications. It is a numerical way of computing the distribution of a function that is useful when analytic approaches are not possible (for example, when the input distributions are non-Normal or the function is non-linear). For our purposes today, Monte Carlo Simulation is quite simple and involves the following steps:

1. Define a function of random variables and the distributions of its input parameters.
2. Create a random sample of each input parameter according to the specified distribution.
3. Create a random sample of the output variable by computing the function for every set of input samples.
4. Evaluate the resulting distribution of the output.

A few key points to recognize are:
1. As the sample size increases, the resulting distribution becomes more accurate.
2. This is a way to get the (approximately) "true" distribution of a function of random variables.
3. Accuracy is relative to the propagation of uncertainty through the function based on the assumed distributions of the input random variables. In other words, MCS can't help you if your function and distributions are poor representations of reality!

### Application: Sewer Pipe Flow Velocity

We will apply Manning's formula for the flow velocity $V$ in a sewer:

$$
V =\frac{1}{n}R^{2/3}S^{1/2}
$$

where $R$ is the hydraulic radius of the sewer pipe (in $m$), $S$ the slope of the pipe (in $m/m$), and $n$ is the coefficient of roughness [$-$].

Both $R$ and $S$ are random variables, as it is known that sewer pipes are susceptible to deformations; $n$ is assumed to be deterministic and in our case $n=0.013$ $s/m^{1/3}$. The sewer pipe considered here has mean values $\mu_R$, $\mu_S$, and standard deviations $\sigma_R$ and $\sigma_S$; $R$ and $S$ are independent.

We are now interested in the mean flow velocity in the sewer as well as the uncertainty expressed by the standard deviation. This is important for the design of the sewer system.

*Disclaimer: the dimensions of the pipe come from a real case study, but some aspects of the exercise are...less realistic.*

### Programming

Remember to use your `mude-base` environment when running this notebook.

Some of the functions below uses <em>keyword arguments</em> to specify some of the parameter values; this is a way of setting "default" values. You can override them when using the function by specifying an alternative syntax. For example, the function here can be used in the following way to return `x=5` and `x=6`, respectively:

```python
def test(x=5)
    return x
   
print(test())
print(test(x=6))
```
Copy/paste into a cell to explore further!

Note also in the cell below that we can increase the default size of the text in our figures to make them more readable!
<!-- #endregion -->

```python id="4fc6e87d-c66e-43df-a937-e969acc409f8"
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import probplot

import ipywidgets as widgets
from ipywidgets import interact

plt.rcParams.update({'font.size': 14})
```

### Theory: Propagation laws for a function of 2 random variables 

We are interested in the mean and variance of $X$, which is a function of 2 random variables: $X=q(Y_1,Y_2)$. The mean and covariance matrix of $Y$ are assumed to be known:

$$\mu_Y = [\mu_1\;\mu_2]^T$$

$$\Sigma_Y = \begin{bmatrix} \sigma^2_1 & Cov(Y_1,Y_2) \\ Cov(Y_1,Y_2) & \sigma^2_2\end{bmatrix}$$

The second-order Taylor series approximation of the mean $\mu_X$ is then given by:

$$\mu_X=\mathbb{E}(q(Y))\approx q(\mu_Y )+\frac{1}{2}\frac{\partial^2 q(\mu_Y )}{\partial Y_1^2 } \sigma_1^2+\frac{1}{2}\frac{\partial^2 q(\mu_Y )}{\partial Y_2^2 }\sigma_2^2+\frac{\partial^2 q(\mu_Y )}{\partial Y_1 \partial Y_2 } Cov(Y_1,Y_2) $$

In most practical situations, the second-order approximation suffices. 

For the variance $\sigma_X^2$ it is common to use only the first-order approximation, given by:

$$\sigma^2_X \approx \left(\frac{\partial q(\mu_Y )}{\partial Y_1 } \right)^2 \sigma^2_1 +\left(\frac{\partial q(\mu_Y )}{\partial Y_2 } \right)^2 \sigma^2_2 + 2\left(\frac{\partial q(\mu_Y )}{\partial Y_1 } \right) \left(\frac{\partial q(\mu_Y )}{\partial Y_2 } \right)  Cov(Y_1,Y_2)$$


## Part 1: Apply the Propagation Laws

We are interested to know how the uncertainty in $R$ and $S$ propagates into the uncertainty of the flow velocity $V$. We will first do this analytically and then implement it in code.

<!-- #region id="bfadcf3f-4578-4809-acdb-625ab3a71f27" -->
<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1:</b>   

Use the Taylor series approximation to find the expression for $\mu_V$ and $\sigma_V$ as function of $\mu_R$, $\sigma_R$, $\mu_S$, $\sigma_S$. Write your answer on paper or using a tablet; later we will learn how to include images directly in our notebooks! For now you can skip this step, as you are not turning this notebook in.
</p>
</div>
<!-- #endregion -->

<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Solution:}$
<b> someone type up the solution in latex please</b>
</div>


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2:</b>   

Complete the function below, such that <code>moments_of_taylor_approximation</code> will compute the approximated $\mu_V$ and $\sigma_V$, as found in the previous Task.
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p><em>Note: there is an intermediate variable defined in the cell below <code>mu_V_0</code> that is not needed; the intention was to provide an easier way to calculate the mean of V, as it represents the first term in the Taylor series approximation.</em></p></div>

```python
def moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S,n):
    """Compute Taylor series approximation of mean and std of V.
    
    Take moments and function parameters as inputs (type float).
    Returns mean and standard deviation of V (type float).
    """
    
    constant = 1/n
    
    # dVdR = YOUR_CODE_HERE
    # dVdS = YOUR_CODE_HERE
    # SOLUTION:
    dVdR = (2/3)*constant*(mu_R**(-1/3))*(mu_S**(1/2))
    dVdS = (1/2)*constant*(mu_R**(2/3))*(mu_S**(-1/2))
    
    # dVdR_2 = (-2/9)*constant*(mu_R**(-4/3))*(mu_S**(1/2))
    # dVdS_2 = (-1/4)*constant*(mu_R**(2/3))*(mu_S**(-3/2))
    # SOLUTION:
    dVdR_2 = (-2/9)*constant*(mu_R**(-4/3))*(mu_S**(1/2))
    dVdS_2 = (-1/4)*constant*(mu_R**(2/3))*(mu_S**(-3/2))
    
    # mu_V_0 = YOUR_CODE_HERE
    # mu_V = YOUR_CODE_HERE
    # SOLUTION:
    mu_V_0 = constant*(mu_R**(2/3))*(mu_S**(1/2))
    mu_V = mu_V_0 + 0.5*dVdR_2*sigma_R**2 + 0.5*dVdS_2*sigma_S**2
    
    # var_V = YOUR_CODE_HERE
    # sigma_V = YOUR_CODE_HERE
    # SOLUTION:
    var_V = (dVdR**2)*sigma_R**2 + (dVdS**2)*sigma_S**2
    sigma_V = np.sqrt(var_V)
    
    return mu_V, sigma_V
```

Now we use the Taylor approximation and make two plots of $\sigma_V$ as a function of $\sigma_R$ for the following cases:
- $\sigma_S$ = 0.002 $m/m$
- $\sigma_S$ = 0 $m/m$ (i.e., slope is deterministic, not susceptible to deformation)

We will use $\mu_R = 0.5 m$ and $\mu_S = 0.015 m/m$, and vary $\sigma_R$ from 0 to 0.1 $m$. 

```python colab={"base_uri": "https://localhost:8080/", "height": 425} id="55ff8dd6-86ef-401a-9a56-02551c348698" outputId="3add4ee9-1054-4726-dc4f-72dca5c1c6c8"
n = 0.013
mu_R = 0.5
mu_S = 0.015
sigma_R = np.linspace(0.0, 0.1, 50)

# case 1 for sigma_S
sigma_S_1 = 0.002
mu_V_1, sigma_V_1 = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S_1, n)

# case 2 for sigma_S
sigma_S_2 = 0
mu_V_2, sigma_V_2 = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S_2, n)

fig, ax = plt.subplots(1, 2, figsize = (16, 6))
# left side plot for case 1 
ax[0].plot(sigma_R, sigma_V_1, linewidth=4)
ax[0].set_ylabel(r'$\sigma_V$ [$m/s$]', size = 20)
ax[0].set_xlabel(r'$\sigma_R$ [$m$]', size = 20)
ax[0].set_title(r'$\sigma_S$ = ' + f'{sigma_S_1} $m/m$, Case 1')
ax[0].set_xlim(0, 0.1)
ax[0].set_ylim(0, 1)
# right side plot for case 2
ax[1].plot(sigma_R, sigma_V_2, linewidth=4)
ax[1].set_ylabel(r'$\sigma_V$ [$m/s$]', size = 20)
ax[1].set_xlabel(r'$\sigma_R$ [m]', size = 20)
ax[1].set_title(r'$\sigma_S$ = ' + f'{sigma_S_2} $m/m$, Case 2')
ax[1].set_xlim(0, 0.1)
ax[1].set_ylim(0, 1)
plt.show()
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3:</b>   
Interpret the figures above, specifically looking at differences between Case 1 and Case 2. Also look at the equations you derived to understand why for Case 1 we get a non-linear relation, and for Case 2 a linear one.
</p>
</div>

<!-- #region id="d3bdade1-2694-4ee4-a180-3872ee17a76d" -->
<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Solution:}$
The standard deviation of $V$ is a non-linear function of $\sigma_R$ and $\sigma_S$ - the left figure shows how $\sigma_V$ increases as function of $\sigma_R$ for a given value $\sigma_S$. 
If $\sigma_S$ is zero, there is no uncertainty in the slope of the pipe, and the standard deviation of $V$ becomes a linear function of $\sigma_R$ (right figure). The uncertainty of $V$ is smaller now, since it only depends on the uncertainty in $R$.
</div>
<!-- #endregion -->

<!-- #region id="a7e4c13f-a2ca-4c2d-a3e2-92d4630715a0" -->
## Part 2: Simulation-Based Propagation 

We will use again the following values:
- $\mu_R = 0.5$ m
- $\mu_S = 0.015$ m/m
- $\sigma_R=0.05$ m
- $\sigma_S=0.002$ m/m

Furthermore, it is assumed that $R$ and $S$ are independent normally distributed random variables. We will generate at least 10,000 simulated realizations each of $R$ and $S$ using a random number generator, and then you need to use these to calculate the corresponding sample values of $V$ and find the moments of that sample.

<!-- #endregion -->

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.1:</b>   
Complete the functions <code>function_of_random_variables</code> and <code>get_samples</code> below to define the function of random variables and then generate a sample of the output from this function, assuming the inputs are also random variables with the Normal distribution. Then find the moments of the samples.
</p>
</div>

```python
def function_of_random_variables(R, S):
    # V = YOUR_CODE_HERE
    # Solution:
    V = 1/n*R**(2/3)*S**(1/2)
    return V

def get_samples(N, sigma_R, mu_R=0.5, mu_S=0.015, sigma_S=0.002, n=0.013):
    """Generate random samples for V from R and S."""
    R = np.random.normal(mu_R, sigma_R, N)
    S = np.random.normal(mu_S, sigma_S, N)
    # V = YOUR_CODE_HERE
    # Solution:
    V = function_of_random_variables(R, S)
    return V

V_samples = get_samples(10000, 0.05)

# mu_V_samples = YOUR_CODE_HERE
# sigma_V_samples = YOUR_CODE_HERE
# Solution:
mu_V_samples = V_samples.mean()
sigma_V_samples = V_samples.std()

print('Moments of the SAMPLES:')
print(f'  {mu_V_samples:.4f} m/s is the mean, and')
print(f'  {sigma_V_samples:.4f} m/s is the std dev.')

mu_V_taylor, sigma_V_taylor = moments_of_taylor_approximation(mu_R, mu_S, 0.05, 0.002, n)
print('\nMoments of the TAYLOR SERIES APPROXIMATION:')
print(f'  {mu_V_taylor:.4f} m/s is the mean, and')
print(f'  {sigma_V_taylor:.4f} m/s is the std dev.')
```

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Task 2.2:}$   
Are the results similar for the linearized and simulated values? Describe the difference quantitatively. Check your result also for the range of values of $\sigma_R$ from 0.01 to 0.10; are they consistent?
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Solution:}$
The estimated moments are both generally within 0.01 m of the result from the simulation when $\sigma_R=0.05$ m. When $\sigma_R$ changes the mean, $\mu_V$, is affected much less than the standard deviation, $\sigma_V$. When $\sigma_R=0.01$ m the Taylor approximation is <em>higher</em> by 0.15 m; when $\sigma_R=0.1$ m the Taylor approximation is <em>lower</em> by 0.30 m
</div>


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Task 2.3:}$   
Run the cell with the sampling algorithm above repeatedly and look at the values printed in the cell output. Which values change? Which values do <em>not</em> change? Explain why, in each case.
</div>


<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Solution:}$
The moments calculated from the sample change because they are computed from random values sampled from the distribution; thus every time we run the code the samples are slightly different. The sample size seems large enough to give accurate results for our purposes, since the values don't change significantly (i.e, less than 1%; less than the difference between the two estimates of the moments). The moments calculated from the Taylor series approximation remain fixed, as they are based on moments of the input variables, not randomly generated samples.
</div>


## Part 3: Validating the Moments with a Distribution

In Part 2 we used a sample of the function of random variables to _validate_ the Taylor approximation (we found that they are generally well-approximated). Now we will assume that the function of random variables has the Normal distribution to validate the moments and see for which range of values they remain a good approximation. This is done by comparing the sample to the assumed distribution; the former is represented by a histogram (also called an empirical probability density function, PDF, when normalized), the latter by a Normal distribution with moments calculated using the Taylor approximation.

We will also use a normal probability plot to assess how well the assumption that $V$ is normally distributed holds up while varying the value of $\sigma_R$, introduced next.

### Theoretical Quantiles with `probplot`

The method `probplot` is built into `scipy.stats` (Documentation [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)) and _validates_ a probability model by comparing samples (i.e., data) to a theoretical distribution (in this case, Normal). The "Theoretical quantile" that is plotted on the x-axis of this plot and measures the distance from the median of a distribution, normalized by the standard deviation, such that $\mathrm{quantile}=q\cdot\sigma$. For example, $q=-1.5$ is $\mu-1.5\cdot\sigma$. The vertical axis is the value of the random variable.

Because we are comparing a theoretical distribution and a sample (data) on the same plot, one of the lines is the Normal PDF, which of course will have an exact match with the _theoretical quantiles_. This is why the Normal PDF will plot as a straight line in `probplot`. Comparing the (vertical) distance between the samples and the theoretical distribution (the red line) allows us to _validate_ our model. In particular, it allows us to validate the model for different regions of the distribution. In your interpretation, for example, you should try and identify whether the model is a good fit for the center and/or tails of the distribution.

Note that `probplot` needs to know what to use for samples (you will tell it this), and what type of theoretical distribution you are using (we already did this for you...`norm`).


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.1:</b>   
Complete the function <code>validate_distribution</code> below (instructions are in the docstring) to plot the empirical probability density function (PDF) of $V$ using your simulated samples. Also plot the Normal PDF in the same figure using the moments computed from the error propagation law. 
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p><em>Hint: if you are struggling with the code below, re-read the introduction to Part 3 carefully!</em></p></div>

```python colab={"base_uri": "https://localhost:8080/", "height": 475, "referenced_widgets": ["b560714d739d431d85b3ca1a9b378c8f", "56b7808a3e2241679b15d517565eaf85", "d867da2ab3d441599b8356ac8e493611", "481c67caa6d1405ea2e00cfe6dbfa32f", "392504e006074b76af62e617c4cde70e", "b0d26f90109f4e0eb6839f0ba43ba980", "ea4c3dc473df41a684cfe7fd1e7fb35d"]} id="80005a5a-510b-4236-a2d6-184d9569eed4" outputId="80ae9e8d-e450-4e17-f092-fbf09fc885e6"
def validate_distribution(N, sigma_R, mu_R=0.5, mu_S=0.015, sigma_S=0.002, n=0.013):
    """Generate samples and plots for V
    
    Compares the sampled distribution of V to a Normal distribution defined
    by the first moments of the error propagation law.
    
    Comparison is made via two plots:
      1. PDF of V~N(mu,sigma) (the approximation) and a histogram (sample)
      2. Probability plot, compares quantiles of sample and CDF of V
    
    Only a plot is returned.
    
    MUDE students fill in the missing code (see: YOUR_CODE_HERE):
      1. Generate samples and find moments
      2. Enter data for the histogram
      3. Define the moments of the Normal distribution to be plotted
      4. Identify the appropriate variables to be printed in the plot titles
      5. Enter the data required for the probability plot
    """
    
    # Generate a sample and compute moments
    # V_samples = YOUR_CODE_HERE
    # mu_V_samples = YOUR_CODE_HERE
    # sigma_V_samples = YOUR_CODE_HERE
    # SOLUTION:
    V_samples = get_samples(N, sigma_R)
    mu_V_samples = V_samples.mean()
    sigma_V_samples = V_samples.std()
    
    # mu_V_taylor, sigma_V_taylor = YOUR_CODE_HERE
    # SOLUTION:
    mu_V_taylor, sigma_V_taylor = moments_of_taylor_approximation(mu_R, mu_S, sigma_R, sigma_S, n)

    # Create left-side plot with histogram and normal distribution
    # Plot histogram
    xmin = 0
    xmax = 10
    x = np.linspace(xmin, xmax, 100)
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    
    # ax[0].hist(YOUR_CODE_HERE, bins = 40, density = True, 
    #            label = 'Empirical PDF of V')
    # SOLUTION:
    ax[0].hist(V_samples, bins = 40, density = True, 
               label = 'Empirical PDF of V')
    
    # Add normal pdf in same figure
    # ax[0].plot(x, norm.pdf(x, YOUR_CODE_HERE, YOUR_CODE_HERE), color = 'black',
    #            lw = 2.5, label='Normal PDF')
    # SOLUTION:
    # **NOTE IN PARTICULAR WHICH mu AND sigma ARE USED!!!**
    ax[0].plot(x, norm.pdf(x, mu_V_taylor, sigma_V_taylor), color = 'black',
               lw = 2.5, label='Normal PDF')
    ax[0].legend()
    ax[0].set_xlabel('V [$m/s$]')
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('Density')
    # ax[0].set_title(f'Simulation with {N} simulated realizations'
    #                 + '\n' + f'mean = {round(YOUR_CODE_HERE, 3)}' 
    #                 f'm/s and std = {round(YOUR_CODE_HERE, 3)} m/s')
    # SOLUTION:
    # **NOTE IN PARTICULAR WHICH mu AND sigma ARE USED!!!**
    ax[0].set_title(f'Simulation with {N} simulated realizations'
                    + '\n' + f'mean = {round(mu_V_samples, 3)}' 
                    f'm/s and std = {round(sigma_V_samples, 3)} m/s')
    
    # Add probability plot in right-side panel
    # probplot(YOUR_CODE_HERE, dist = norm, fit = True, plot = ax[1])
    # SOLUTION:
    # **NOTE IN PARTICULAR WHICH mu AND sigma ARE USED!!!**
    probplot(V_samples, dist = norm, fit = True, plot = ax[1])
    ax[1].legend(['Generated samples', 'Normal fit'])
    ax[1].get_lines()[1].set_linewidth(2.5)
    plt.show()

validate_distribution(10000, 0.01)
```

<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p><em>Note: in the figure above (and widget below) the right-hand plot by default is labeled "Theoretical quantiles," but in fact it is the q value described above (the number of standard deviations).</em></p></div>


### Validate the Distribution of $V$ for Various $\sigma_R$

The code below uses a widget to call your function to make the plots and add a slider to change the values of $\sigma_R$ and visualize the change in the distributions.


<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Task 3.2:}$
Run the cell below, then play with the slider to change $\sigma_R$. How well does the error propagation law match the "true" distribution (the samples)? State your conclusion and explain why. Check also whether there is an impact for different $\sigma_R$ values.
</p>
</div>

```python colab={"base_uri": "https://localhost:8080/", "height": 475, "referenced_widgets": ["b560714d739d431d85b3ca1a9b378c8f", "56b7808a3e2241679b15d517565eaf85", "d867da2ab3d441599b8356ac8e493611", "481c67caa6d1405ea2e00cfe6dbfa32f", "392504e006074b76af62e617c4cde70e", "b0d26f90109f4e0eb6839f0ba43ba980", "ea4c3dc473df41a684cfe7fd1e7fb35d"]} id="80005a5a-510b-4236-a2d6-184d9569eed4" outputId="80ae9e8d-e450-4e17-f092-fbf09fc885e6"
@interact(sigma_R=(0, 0.1, 0.005))
def samples_slideplot(sigma_R):
    validate_distribution(50000, sigma_R);
```

<!-- #region id="782c842e-ceb8-4e3c-b767-1f3efa4fb9b2" -->
<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\mathbf{Solution:}$
Using a different value for $\sigma_R$ has several impacts:
    
- For a larger $\sigma_R$ $\sigma_V$ will become larger (we saw this in Task 2) and $\mu_V$ will become smaller.
- For a larger $\sigma_R$ the PDF will become wider and less peaked (we saw this in Task 2).
- Depending on the value of $\sigma_R$, the more <em>extreme</em> values of the random variable $V$ deviate from that expected by the Normal distribution, evidenced by the distance between the blue dots and the red lines.
- $V$ does not follow a normal distribution; it is skewed slightly and the tails deviate most; the amount of deviation depends on the values of $\sigma_R$ and $\sigma_S$.

The reason for this is that $V$ is a non-linear function of the normally distributed random variables $R$ and $S$, due to the non-linearity $V$ will <em>not</em> be Normally distributed.
</div>
<!-- #endregion -->

<div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Task 3.3:}$
To further validate the error propagation law, estimate the probability that the "true" velocity of the tunnel is in the inaccurate range of values (assuming that the Normal distribution is a suitable model for the distribution of the function of random variables).

<em>Hint: recall that in this notebook a quantile, $q$, is defined as a standard deviation, and that the probability of observing a random variable $X$ such that $P[X\lt q]$ can be found with the CDF of the standard Normal distribution, evaluated in Python using <code>scipy.stats.norm.cdf(q)</code>.</em>
</p>
</div>


<div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p><em>

$\textbf{Note:}$
As we saw in 3.2, the Normal distribution does not fit the tails of the distribution of the function of random variables perfectly. Although this means using the Normal distribution may not be a good way of estimating the probability of "being in the tails," the approach in this Task is still suitable for getting an idea of the order of magnitude, and observing how sever this "error" maybe for different assumptions of $\sigma_R$.</em></p></div>

```python
# p = YOUR_CODE_HERE

# Solution:
p = 2*norm.cdf(-3)

print(f'The probability is {p:0.6e}')

# extra solution
print(f'The probability is {2*norm.cdf(-2.5):0.6e} for sigma_R = 0.01')
print(f'The probability is {2*norm.cdf(-3.0):0.6e} for sigma_R = 0.05')
print(f'The probability is {2*norm.cdf(-3.5):0.6e} for sigma_R = 0.10')
```

<div style="background-color:#FAE99E; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">

$\textbf{Solution:}$
Since the probability plots give the quantiles, we simply need to recall that the probability of being in the "tails" is twice the probability of being less than one of the quantiles (as a negative value); the factor two is due to counting both tails. As we can see, the probability is generally small (no more than about 1%). It is interesting that the probability decreases by factor 100 as the value of $\sigma_R$ increases from 0.01 to 0.10.
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
