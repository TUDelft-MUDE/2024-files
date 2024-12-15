import numpy as np
import matplotlib.pylab as plt
import pandas as pd

data=pd.read_csv(filepath_or_buffer='justIce.csv',index_col=0)
data.index = pd.to_datetime(data.index, format="%Y-%m-%d")

data_2021 = data.loc['2021']
h_ice = (data_2021.to_numpy()).ravel()
t_days = ((data_2021.index - data_2021.index[0]).days).to_numpy()

x = t_days
y = h_ice

degree = 4
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)
x_fit = np.linspace(0, 100, 100)
y_fit = polynomial(x_fit)
derivative = polynomial.deriv()
y_derivative = derivative(x_fit)

plt.figure(figsize=(15, 4))
plt.scatter(x, y, label='Data', color='blue')
plt.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})', color='red')
plt.xlabel('day of year')
plt.ylabel('Ice Thickness [m]')
plt.title('Ice Thickness measurements (2021)')
plt.legend()
plt.grid()
plt.show()

dh_dt_FD = (h_ice[1:]-h_ice[:-1])/(t_days[1:]-t_days[:-1]) 
dh_dt_BD = (h_ice[1:]-h_ice[:-1])/(t_days[1:]-t_days[:-1]) 
dh_dt_CD = (h_ice[1:]-h_ice[:-1])/(t_days[1:]-t_days[:-1]) 

fig, ax1 = plt.subplots(figsize=(15, 4))
ax1.plot(x_fit, y_derivative, label='Derivative', color='magenta')
ax1.set_ylabel('growth rate [m/day]', color='magenta')
ax1.tick_params(axis='y', labelcolor='magenta')
ax1.scatter(t_days[:-1], dh_dt_FD,
            color='blue', marker='o', label='dh_dt_FD Forward Difference')
ax1.scatter(t_days[1:], dh_dt_BD,
            color='red', marker='o', label='dh_dt_BD Backward Difference')
ax1.scatter((t_days[1:]+t_days[:-1])/2, dh_dt_CD,
            color='purple', marker='o', label='dh_dt_CD Central Difference')

ax2 = ax1.twinx()
ax2.scatter(x, y, color='green', marker='x', label='h_ice Measurements')
ax2.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})', color='green',linestyle='--',alpha=0.5)
ax2.set_ylabel('Ice Thickness [m]', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Ice thickness (2021)')
fig.tight_layout()  # Adjust layout to prevent overlap
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_xlabel('Day of year')
plt.show()

num_samples = 6 # 

np.random.seed(13)  #setting seed
indices = np.random.choice(len(x_fit), size=num_samples, replace=False)
sampled_t_days = x_fit[indices]
sampled_h_ice = y_fit[indices]

sorted_indices = np.argsort(sampled_t_days)
sampled_t_days = sampled_t_days[sorted_indices]
sampled_h_ice = sampled_h_ice[sorted_indices]

plt.figure(figsize=(15,4))
plt.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})', color='grey',linestyle='--',alpha=0.5)
plt.scatter(sampled_t_days, sampled_h_ice, label='Randomly Sampled Points', color='blue')
plt.xlabel('t_days')
plt.ylabel('h_ice')
plt.title('Randomly Sampled Points from Fitted polynomial')
plt.legend()
plt.show()

dh_dt_FD_sampled_from_fit = (sampled_h_ice[1:]-sampled_h_ice[:-1])/(sampled_t_days[1:]-sampled_t_days[:-1]) 
dh_dt_BD_sampled_from_fit = (sampled_h_ice[1:]-sampled_h_ice[:-1])/(sampled_t_days[1:]-sampled_t_days[:-1]) 
dh_dt_CD_sampled_from_fit = (sampled_h_ice[1:]-sampled_h_ice[:-1])/(sampled_t_days[1:]-sampled_t_days[:-1]) 

fig, ax1 = plt.subplots(figsize=(15, 4))
ax1.plot(x_fit, y_derivative, label='Derivative', color='magenta')
ax1.set_ylabel('growth rate [m/days]', color='magenta')
ax1.tick_params(axis='y', labelcolor='magenta')
ax1.scatter(sampled_t_days[:-1], dh_dt_FD_sampled_from_fit,
            color='blue', marker='o', label='dh_dt_FD Forward Difference')
ax1.scatter(sampled_t_days[1:], dh_dt_BD_sampled_from_fit,
            color='red', marker='o', label='dh_dt_BD Backward Difference')
ax1.scatter((sampled_t_days[1:]+sampled_t_days[:-1])/2, dh_dt_CD_sampled_from_fit,
            color='purple', marker='o', label='dh_dt_CD Central Difference')

ax2 = ax1.twinx()
ax2.scatter(sampled_t_days, sampled_h_ice, color='green', marker='x', label='ice measurements*')
ax2.plot(x_fit, y_fit, label=f'Polynomial fit (degree {degree})', color='green',linestyle='--',alpha=0.5)
ax2.set_ylabel('Polynomial Fit', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Ice Thickness (2021)')
fig.tight_layout()  
ax1.legend(loc='lower left')
ax2.legend(loc='upper right')
ax1.set_xlabel('Day of year')

plt.show()

plt.show()
fig.savefig('central_diff_illustration.svg')

estimated_h_ice_FD = [h_ice[0]]  
estimated_h_ice_BD = [h_ice[0]]  
estimated_h_ice_CD = [h_ice[1]]  

for i in range(1, len(t_days) - 1):
    delta_t = t_days[i] - t_days[i-1]
    next_point_FD = estimated_h_ice_FD[-1] + dh_dt_FD[i-1] * delta_t
    estimated_h_ice_FD.append(next_point_FD)

