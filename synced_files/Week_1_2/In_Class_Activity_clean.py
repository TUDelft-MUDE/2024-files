# ---

# ---

# %% [markdown]

# %% [markdown] cell_id="21f9833788f64e78a35bc8cac535e76d" deepnote_cell_type="markdown"

# %% cell_id="10af2251a77c4f84a32d01bed7350da9" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=1559 execution_start=1694006986292 source_hash
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import scipy.optimize as opt

# %% [markdown] cell_id="b1d3e3d2f92c4de29aba4aa61c525867" deepnote_cell_type="markdown" jp-MarkdownHeadingCollapsed=true

# %% [markdown]

# %% cell_id="47af53f26ebc41b2a90a63dd3b02ab0f" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=33 execution_start=1694006987852 source_hash
data = np.loadtxt('data/days.csv', dtype=str, delimiter=';', skiprows=1)
data = np.char.replace(data, ',', '.')
data = data.astype(float)

data[0:10]

# %% [markdown] jp-MarkdownHeadingCollapsed=true

# %% [markdown] cell_id="1b15837c64e748b89fafad1f8007a399" deepnote_cell_type="markdown"

# %% cell_id="2aaf111f8cdd40959fb3496188242efa" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=32 execution_start=1694006987878 source_hash
np.shape(data)

# %% [markdown] cell_id="e901697b36064391b4a62d78c955dd6b" deepnote_cell_type="markdown"

# %% [markdown] cell_id="43139b45e34f4252a6a270336ca401ba" deepnote_cell_type="markdown"

# %% cell_id="efc536f1b0804d50ae01f3013a9d1367" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=20 execution_start=1694006987934 source_hash
mean = np.mean(data[:,1])
std = np.std(data[:,1])

print(f'Mean: {mean:.3f}\n\
Standard deviation: {std:.3f}')

# %% [markdown] cell_id="2a1ddf21c02141dc8dfebee83c602a73" deepnote_cell_type="markdown"

# %% cell_id="a67a4c42e202489e8801bfb67ca95fbc" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=229 execution_start=1694007029078 source_hash
plt.scatter(data[:, 0], data[:, 1], label='Measured data')
plt.xlabel('Year [-]')
plt.ylabel('Number of days/year [-]')
plt.title(f'Number of days per year between {data[0,0]:.0f}-{data[-1,0]:.0f}')
plt.grid()

# %% [markdown] cell_id="37edb558a10b47d88b3e4f683da56221" deepnote_cell_type="markdown"

# %% [markdown] cell_id="a05297ff6298401192d49f8e257ff9ec" deepnote_cell_type="markdown" jp-MarkdownHeadingCollapsed=true

# %% [markdown]

# %% cell_id="4d72dacfdbb543d9925a40a30eac2944" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=22 execution_start=1694007048665 source_hash
def regression(x, y):
    '''
    Determine linear regression

    Input: x = array, x values
           y = array, y values

    Output: r_sq = coefficient of determination
            q = intercept of the line
            m = slope of the line
    '''

    regression = sci.linregress(x, y)
    r_sq = regression.rvalue**2
    q = regression.intercept 
    m = regression.slope 

    print(f'Coefficient of determination R^2 = {r_sq:.3f}')
    print(f'Intercept q = {q:.3f} \nSlope m = {m:.3f}')

    return r_sq, q, m

# %% cell_id="31d247d5cafc4c9690c39bc736f04758" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=20 execution_start=1694007051941 source_hash
r_sq, q, m = regression(data[:,0], data[:,1])

# %% [markdown]

# %% [markdown] cell_id="143e5d1bf8324d9f80fca4af9a0d162c" deepnote_cell_type="markdown"

# %% cell_id="650d74731c11439684ba4de949b2127b" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=16 execution_start=1694007056091 source_hash
def calculate_line(x, m, q):
    '''
    Determine y values from linear regression

    Input: x = array
           m = slope of the line
           q = intercept of the line

    Output: y = array
    '''

    y = m * x + q

    return y

# %% cell_id="cd3c13f84f124211af2d5e251754d636" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=10 execution_start=1694007059430 source_hash
line = calculate_line(data[:,0], m, q)

# %% cell_id="4424bf1d4bf244baaa24cfcf0b5782bc" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=246 execution_start=1694007094334 source_hash
fig, axes = plt.subplots(1, 2,figsize = (12, 4))

axes[0].scatter(data[:,0], data[:,1], label = 'Observations')
axes[0].plot(data[:,0], line, color='r', label='Fitted line')
axes[0].set_ylabel('Number of days/year [-]')
axes[0].set_xlabel('Year [-]')
axes[0].grid()
axes[0].legend()
axes[0].set_title('(a) Number of days as function of the year')

axes[1].scatter(data[:,1], line)
axes[1].plot([105, 145],[105, 145], line, color = 'k')
axes[1].set_xlim([105, 145])
axes[1].set_ylim([105, 145])
axes[1].set_ylabel('Predicted number of days/year [-]')
axes[1].set_xlabel('Observed number of days/year [-]')
axes[1].grid()
axes[1].set_title('(b) Observed and predicted number of days');

# %% [markdown] cell_id="b564cab0d51c40f2aa3c7ebb9affaade" deepnote_cell_type="markdown"

# %% [markdown] cell_id="67e44fda81f24273b8d28edd35a50d87" deepnote_cell_type="markdown"

# %% cell_id="b9e8d87cb37e4950ad597f3cc9a71b49" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=30 execution_start=1693551237449 source_hash
def RMSE(data, fit_data):
    '''
    Compute the RMSE

    RMSE = [sum{(data - fit_data)^2} / N]^(1/2)

    Input: data = array with real measured data
           fit_data = array with predicted data
    
    Output: RMSE
    '''

    diff_n = (data - fit_data)**2
    mean = np.mean(diff_n)

    error = mean**(1/2)
    print(f'RMSE = {error:.3f}')
    return error

# %% cell_id="a8a3a34164ce4347b238194128ae1ce2" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=28 execution_start=1693551237454 source_hash
RMSE_line = RMSE(data[:,1], line)

# %% [markdown] cell_id="b564cab0d51c40f2aa3c7ebb9affaade" deepnote_cell_type="markdown"

# %% [markdown]

# %%
def rbias(data, fit_data):
    '''
    Compute the relative bias

    rbias = [sum{(fit_data-data) / |data|}]/N

    Input: data = array with real measured data
           fit_data = array with predicted data
    
    Output: relative bias
    '''
    bias = np.mean((fit_data-data)/data)

    print(f'rbias = {bias:.3f}')
    return bias

# %%
rbias_line = rbias(data[:,1], line)

# %% [markdown] cell_id="b564cab0d51c40f2aa3c7ebb9affaade" deepnote_cell_type="markdown"

# %% [markdown] cell_id="19496dbefd3247f09c2226579c7b665f" deepnote_cell_type="markdown" jp-MarkdownHeadingCollapsed=true

# %% [markdown]

# %% cell_id="adf17125447a424f915d421e541b8a6e" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=9 execution_start=1693551237510 source_hash
def conf_int(x, y, alpha):
    '''
    Compute the confidence intervals

    Input: x = array, observations
           y = array, predictions
           alpha = float, confidence interval

    Output: k = float, width of the confidence interval
    '''
    sd_error = (y - x).std()
    k = sci.norm.ppf(1-alpha/2)*sd_error

    return k

# %%
k = conf_int(data[:,1], line, 0.05)
ci_low = line - k
ci_up = line + k

plt.scatter(data[:,0], data[:,1], label = 'Observations')
plt.plot(data[:,0], line, color='r', label='Fitted line')
plt.plot(data[:,0], ci_low, '--k')
plt.plot(data[:,0], ci_up, '--k')
plt.ylabel('Number of days/year [-]')
plt.xlabel('Year [-]')
plt.grid()
plt.legend()
plt.title('Number of days as function of the year')

# %% [markdown] cell_id="b564cab0d51c40f2aa3c7ebb9affaade" deepnote_cell_type="markdown"

# %% [markdown] cell_id="a19fb5b7e1cd4c32b435b8b967933700" deepnote_cell_type="markdown" jp-MarkdownHeadingCollapsed=true

# %% [markdown]

# %% [markdown] cell_id="ca8a6b9d68234ae69a799a3f4f3866a2" deepnote_cell_type="markdown"

# %% cell_id="ebd3c8f852c24c9d8128ae35482824de" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=22 execution_start=1693551238912 source_hash
def parabola(x, a, b, c):
    '''
    Compute the quadratic model

    y = a * x^2 + b * x + c

    Input: x = array, independent variable
           a, b, c = parameters to be optimized

    Output: y = array, dependent variable
    '''

    y = a * x**2 + b * x + c
    return y

# %% cell_id="6897b364543142b4adc2d85ffbcb7d42" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=29 execution_start=1693551238923 source_hash
popt_parabola, pcov_parabola = opt.curve_fit(parabola, data[:,0], data[:,1])

print(f'Optimal estimation for parameters:\n\
a = {popt_parabola[0]:.3e}, b = {popt_parabola[1]:.3f}, c = {popt_parabola[2]:.3f}\n')

print(f'Covariance matrix for parameters:\n\
Sigma = {pcov_parabola}')

# %% [markdown] cell_id="10efd81771064ce0ac4d50095be06e23" deepnote_cell_type="markdown"

# %% cell_id="182a633995af49a9ba16ac67f2d0181d" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=55 execution_start=1693551238924 source_hash
fitted_parabola = parabola(data[:,0], *popt_parabola)

# %% [markdown] cell_id="6cc6a36c668f4ee2b26d41b27d335691" deepnote_cell_type="markdown"

# %% cell_id="4221e9ef1da1463eb5f524d9a458ab0b" deepnote_cell_type="code" deepnote_to_be_reexecuted=true execution_millis=13 execution_start=1693551238973 source_hash
k = conf_int(data[:,1], fitted_parabola, 0.05)
ci_low_2 = fitted_parabola - k
ci_up_2 = fitted_parabola + k

plt.scatter(data[:,0], data[:,1], label = 'Observations')
plt.plot(data[:,0], fitted_parabola, color='r', label='Fitted line')
plt.plot(data[:,0], ci_low_2, '--k')
plt.plot(data[:,0], ci_up_2, '--k')
plt.ylabel('Number of days/year [-]')
plt.xlabel('Year [-]')
plt.grid()
plt.legend()
plt.title('Number of days as function of the year')

# %% [markdown]

# %%
RMSE_parabola = RMSE(data[:,1], fitted_parabola)
R2_parabola = 1-((data[:,1]-fitted_parabola)**2).mean()/(data[:,1].var())
print(f'Coefficient of determination = {R2_parabola:.3f}')
rbias_parabola = rbias(data[:,1], fitted_parabola)

# %% [markdown] cell_id="b564cab0d51c40f2aa3c7ebb9affaade" deepnote_cell_type="markdown"

# %% [markdown]

