# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import interpolate
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

np.random.seed(42)
noise_level = 1.0

data_x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]).transpose()
data_t = 0.8 * data_x + 4.75 + np.random.normal(scale=noise_level,size=data_x.shape)

x_val = np.linspace(np.min(data_x),np.max(data_x),1000)
t_val = 0.8*x_val + 4.75 + np.random.normal(scale=noise_level,size=x_val.shape)
x_val = x_val.reshape(-1,1)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x.flatten(), data_t, 'x', color='blue', markersize=10, label='Data')
ax.set_title('Linear Data Example', fontsize=16)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('t', fontsize=14)
ax.legend(fontsize=14)
ax.grid(True)
plt.show()

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
model = 

# %% [markdown]

# %%
n_epochs = 10000
N_print = 10**(int(np.log10(n_epochs)) - 1)

for epoch in range(n_epochs):
    model.partial_fit(data_x, data_t.flatten())

    MLP_prediction = 
    MLP_valprediction = 
    
    if epoch%N_print==0 or epoch==n_epochs-1: 
        print((f'Epoch: {epoch:6d}/{n_epochs}, '
               + f'MSE: {mean_squared_error(data_t, MLP_prediction.reshape(-1,1)):0.4f}, '
               + f'Real loss: {mean_squared_error(t_val,MLP_valprediction):0.4f}'))

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(data_x, MLP_prediction, "-o", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

# %% [markdown]

# %%
print(f'Model coefficients: {model.coefs_}')
print(f'Model intercepts: {model.intercepts_}')

# %% [markdown]

# %%
model = YOUR_CODE_HERE

n_epochs = 10000
N_print = 10**(int(np.log10(n_epochs)) - 1)

for epoch in range(n_epochs):
    model.partial_fit(data_x, data_t.flatten())

    MLP_prediction = YOUR CODE HERE
    MLP_valprediction = YOUR CODE HERE
    
    if epoch%N_print==0 or epoch==n_epochs-1: 
        print((f'Epoch: {epoch:6d}/{n_epochs}, '
               + f'MSE: {mean_squared_error(data_t, MLP_prediction.reshape(-1,1)):0.4f}, '
               + f'Real loss: {mean_squared_error(t_val,MLP_valprediction):0.4f}'))

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(data_x, MLP_prediction, "-o", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

# %% [markdown]

# %% [markdown]

# %%
model = YOUR_CODE_HERE

n_epochs = 10000
N_print = 10**(int(np.log10(n_epochs)) - 1)

for epoch in range(n_epochs):
    model.partial_fit(data_x, data_t.flatten())

    MLP_prediction = YOUR CODE HERE
    MLP_valprediction = YOUR CODE HERE
    
    if epoch%N_print==0 or epoch==n_epochs-1: 
        print((f'Epoch: {epoch:6d}/{n_epochs}, '
               + f'MSE: {mean_squared_error(data_t, MLP_prediction.reshape(-1,1)):0.4f}, '
               + f'Real loss: {mean_squared_error(t_val,MLP_valprediction):0.4f}'))

# %%

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(data_x, MLP_prediction, "-o", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

# %% [markdown]

# %% [markdown]

# %%
gnss = pd.read_csv('./data/gnss_observations2.csv')
dates_gnss = pd.to_datetime(gnss['dates'])
gnss_obs = (gnss['observations[m]']).to_numpy() * 1000

def to_days_years(dates):
    '''Convert the observation dates to days and years.'''
    
    dates_datetime = pd.to_datetime(dates)
    time_diff = (dates_datetime - dates_datetime[0])
    days_diff = (time_diff / np.timedelta64(1,'D')).astype(int)
    
    days = days_diff.to_numpy()
    
    return days

days_gnss = to_days_years(dates_gnss)

# %%
plt.figure(figsize=(15,5))
plt.plot(days_gnss, gnss_obs, 'o', mec='black', label = 'GNSS')
plt.legend()
plt.title('GNSS observations of land deformation')
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.show()

# %% [markdown]

# %% [markdown]

# %%
X = days_gnss.reshape(-1, 1)
t = gnss_obs.reshape(-1, 1)

# %%
X_train, X_val, t_train, t_val  = YOUR_CODE_HERE

# %%
plt.figure(figsize=(15,5))
plt.plot(X_train, t_train, 'o', mec='green', label = 'Training')
plt.plot(X_val, t_val, 'o', mec='blue', label = 'Validation')
plt.title('GNSS observations of land deformation - training and validation datasets')
plt.legend()
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.show()

# %% [markdown]

# %% [markdown]

# %%
input_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)

t_train_scaled = target_scaler.fit_transform(t_train)
t_val_scaled = target_scaler.transform(t_val)

# %%
plt.figure(figsize=(15,5))
plt.plot(X_train_scaled, t_train_scaled, 'o', mec='green', label = 'Training')
plt.plot(X_val_scaled, t_val_scaled, 'o', mec='blue', label = 'Validation')
plt.title('Normalized GNSS dataset')
plt.legend()
plt.ylabel('Normalized displacement [-]')
plt.xlabel('Normalized time [-]')
plt.show()

# %% [markdown]

# %% [markdown]

# %%
model_gnss = YOUR CODE HERE

# %% [markdown]

# %%
train_losses = []
val_losses = []

epochs = YOUR CODE HERE

for epoch in range(YOUR CODE HERE):
    model_gnss.partial_fit(X_train_scaled, t_train_scaled.flatten())

    
    train_pred = YOUR CODE HERE
    train_loss = YOUR CODE HERE
    train_losses.YOUR CODE HERE

    
    val_pred = YOUR CODE HERE
    val_loss = YOUR CODE HERE
    val_losses.YOUR CODE HERE

    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# %% [markdown]

# %%
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', c='b')
plt.plot(val_losses, label='Validation Loss', c='r')
plt.title('Training, Validation, and Test Losses over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# %%

plt.figure(figsize=(15,5))
plt.plot(X_train, t_train, 'o', mec='green', label = 'Training')
plt.plot(X_val, t_val, 'o', mec='blue', label = 'Validation')

x_plot = np.linspace(np.min(X),np.max(X),1000).reshape(-1,1)
y_plot = model_gnss.predict(input_scaler.transform(x_plot))
plt.plot(x_plot,target_scaler.inverse_transform(y_plot.reshape(-1,1)),color='orange',linewidth=5,label='Network predictions')

plt.title('Obvserved vs Predicted Values')
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.legend()
plt.show()

# %% [markdown]

# %% [markdown]

# %%
train_losses = []
val_losses = []

epochs = YOUR CODE HERE

for epoch in range(YOUR CODE HERE):
    model_gnss.partial_fit(X_train_scaled, t_train_scaled.flatten())

    
    train_pred = YOUR CODE HERE
    train_loss = YOUR CODE HERE
    train_losses.YOUR CODE HERE

    
    val_pred = YOUR CODE HERE
    val_loss = YOUR CODE HERE
    val_losses.YOUR CODE HERE

    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# %%

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', c='b')
plt.plot(val_losses, label='Validation Loss', c='r')
plt.title('Training and Validation Losses over Epochs with a better model')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# %%

plt.figure(figsize=(15,5))
plt.plot(X_train, t_train, 'o', mec='green', label = 'Training')
plt.plot(X_val, t_val, 'o', mec='blue', label = 'Validation')

x_plot = np.linspace(np.min(X),np.max(X),1000).reshape(-1,1)
y_plot = new_model_gnss.predict(input_scaler.transform(x_plot))
plt.plot(x_plot,target_scaler.inverse_transform(y_plot.reshape(-1,1)),color='orange',linewidth=5,label='Network predictions')

plt.title('Obvserved vs Predicted Values')
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.legend()
plt.show()

# %% [markdown]

# %% [markdown]

# %%
gw = pd.read_csv('./data/groundwater_levels2.csv')
dates_gw = pd.to_datetime(gw['dates'])
gw_obs = (gw['observations[mm]']).to_numpy()

# ------------------------------------------------------------- #

def to_days_years(dates):
    '''Convert the observation dates to days and years.'''
    
    dates_datetime = pd.to_datetime(dates)
    time_diff = (dates_datetime - dates_datetime[0])
    days_diff = (time_diff / np.timedelta64(1,'D')).astype(int)
    
    days = days_diff.to_numpy()
    years = days/365
    
    return days, years

# ------------------------------------------------------------- #

days_gnss, years_gnss = to_days_years(dates_gnss)
days_gw, years_gw = to_days_years(dates_gw)

interp = interpolate.interp1d(days_gw, gw_obs)

GW_at_GNSS_times = interp(days_gnss)

# ------------------------------------------------------------- #

A_gnss = np.ones((len(dates_gnss), 3))
A_gnss[:,1] = days_gnss
A_gnss[:,2] = GW_at_GNSS_times

y_gnss = gnss_obs

m_gnss = np.shape(A_gnss)[0]
n_gnss = np.shape(A_gnss)[1]

# ------------------------------------------------------------- #

std_gnss = 15 

Sigma_Y_gnss = np.identity(len(dates_gnss))*std_gnss**2

# ------------------------------------------------------------- #

def BLUE(A, y, Sigma_Y):
    """Calculate the Best Linear Unbiased Estimator
    
    Write a docstring here (an explanation of your function).
    
    Function to calculate the Best Linear Unbiased Estimator
    
    Input:
        A = A matrix (mxn)
        y = vector with obervations (mx1)
        Sigma_Y = Varaiance covariance matrix of the observations (mxm)
    
    Output:
        xhat = vector with the estimates (nx1)
        Sigma_Xhat = variance-covariance matrix of the unknown parameters (nxn)
    """
    
    Sigma_Xhat = np.linalg.inv(A.T @ np.linalg.inv(Sigma_Y) @ A)
    xhat = Sigma_Xhat @ A.T @ np.linalg.inv(Sigma_Y) @ y
    
    return xhat, Sigma_Xhat 

# ------------------------------------------------------------- #

xhat_gnss, Sigma_Xhat_gnss = BLUE(A_gnss, y_gnss, Sigma_Y_gnss)

def plot_residual(date, y_obs, yhat, data_type, A,
                  Sigma_Xhat, Sigma_Y, true_disp):

    ehat = y_obs - yhat

    
    Sigma_Yhat = A @ Sigma_Xhat @ A.T
    std_y = np.sqrt(Sigma_Yhat.diagonal())

    
    Sigma_ehat = Sigma_Y - Sigma_Yhat
    std_ehat = np.sqrt(Sigma_ehat.diagonal())

    
    k99 = norm.ppf(1 - 0.5*0.01)
    confidence_interval_y = k99*std_y
    confidence_interval_res = k99*std_ehat

    
    plt.figure(figsize = (15,5))
    plt.plot(date, y_obs, 'k+',  label = 'Observations')
    plt.plot(date, yhat,  label = 'Fitted model')
    plt.fill_between(date, (yhat - confidence_interval_y), 
                     (yhat + confidence_interval_y), facecolor='orange',
                     alpha=0.4, label = '99% Confidence Region')
    plt.plot(date, true_disp, label = 'True model')
    plt.legend()
    plt.ylabel(data_type + ' Displacement [mm]')
    plt.xlabel('Time')
    plt.title(data_type + ' Observations and Fitted Model')

    return ehat

# ------------------------------------------------------------- #

k_true = 0.1
R_true = -25 
a_true = 180
d0_true = 10

disp_gnss  = (d0_true + R_true*(1 - np.exp(-days_gnss/a_true)) 
              + k_true*GW_at_GNSS_times) 

yhat_gnss = A_gnss @ xhat_gnss
ehat_gnss_1 = plot_residual(dates_gnss, y_gnss, yhat_gnss,
                             'GNSS', A_gnss, 
                             Sigma_Xhat_gnss, Sigma_Y_gnss, disp_gnss)

# ------------------------------------------------------------- #

# %%

plt.figure(figsize=(15,5))
plt.plot(days_gnss, yhat_gnss,  label = 'BLUE model', color='black')
plt.plot(days_gnss, disp_gnss, label='True model', color='orange')

x_plot = np.linspace(np.min(X),np.max(X),1000).reshape(-1,1)
y_plot = new_model_gnss.predict(input_scaler.transform(x_plot))
plt.plot(x_plot,target_scaler.inverse_transform(y_plot.reshape(-1,1)),color='purple',linewidth=3,label='Network')

plt.title('Obvserved vs Predicted Values')
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.legend()
plt.show()

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

