import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci
import scipy.optimize as opt

data = np.loadtxt('data/days.csv', dtype=str, delimiter=';', skiprows=1)
data = np.char.replace(data, ',', '.')
data = data.astype(float)

data[0:10]

np.shape(data)

mean = np.mean(data[:,1])
std = np.std(data[:,1])

print(f'Mean: {mean:.3f}\n\
Standard deviation: {std:.3f}')

plt.scatter(data[:, 0], data[:, 1], label='Measured data')
plt.xlabel('Year [-]')
plt.ylabel('Number of days/year [-]')
plt.title(f'Number of days per year between {data[0,0]:.0f}-{data[-1,0]:.0f}')
plt.grid()

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

r_sq, q, m = regression(data[:,0], data[:,1])

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

line = calculate_line(data[:,0], m, q)

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
axes[1].set_title('(b) Observed and predicted number of days')

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

RMSE_line = RMSE(data[:,1], line)

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

rbias_line = rbias(data[:,1], line)

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

popt_parabola, pcov_parabola = opt.curve_fit(parabola, data[:,0], data[:,1])

print(f'Optimal estimation for parameters:\n\
a = {popt_parabola[0]:.3e}, b = {popt_parabola[1]:.3f}, c = {popt_parabola[2]:.3f}\n')

print(f'Covariance matrix for parameters:\n\
Sigma = {pcov_parabola}')

fitted_parabola = parabola(data[:,0], *popt_parabola)

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

RMSE_parabola = RMSE(data[:,1], fitted_parabola)
R2_parabola = 1-((data[:,1]-fitted_parabola)**2).mean()/(data[:,1].var())
print(f'Coefficient of determination = {R2_parabola:.3f}')
rbias_parabola = rbias(data[:,1], fitted_parabola)

