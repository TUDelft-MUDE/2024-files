
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

np.random.seed(42)
noise_level = 1.0

data_x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]).transpose()
data_t = 0.8 * data_x + 4.75 + np.random.normal(scale=noise_level,size=data_x.shape)

x_val = np.linspace(np.min(data_x),np.max(data_x),1000)
t_val = 0.8*x_val + 4.75 + np.random.normal(scale=noise_level,size=x_val.shape)
x_val = x_val.reshape(-1,1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x.flatten(), data_t, 'x', color='blue', markersize=10, label='Data')
ax.set_title('Linear Data Example', fontsize=16)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('t', fontsize=14)
ax.legend(fontsize=14)
ax.grid(True)
plt.show()

model = YOUR CODE HERE

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

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(data_x, MLP_prediction, "-", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

print(f'Model coefficients: {model.coefs_}')
print(f'Model intercepts: {model.intercepts_}')

model = YOUR CODE HERE

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

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(x_val, MLP_valprediction, "-", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

model = YOUR CODE HERE

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

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data_x, data_t, ".", markersize=20, label="Data")
ax.plot(x_val, MLP_valprediction, "-", markersize=10, label="Prediction")

ax.set_title("Linear Data Example", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("t", fontsize=14)

ax.legend(fontsize=14)

ax.grid(True)

plt.show()

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

plt.figure(figsize=(15,5))
plt.plot(days_gnss, gnss_obs, 'o', mec='black', label = 'GNSS')
plt.legend()
plt.title('GNSS observations of land deformation')
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.show()

X = days_gnss.reshape(-1, 1)
t = gnss_obs.reshape(-1, 1)

X_train, X_val, t_train, t_val  = YOUR CODE HERE

plt.figure(figsize=(15,5))
plt.plot(X_train, t_train, 'o', mec='green', label = 'Training')
plt.plot(X_val, t_val, 'o', mec='blue', label = 'Validation')
plt.title('GNSS observations of land deformation - training and validation datasets')
plt.legend()
plt.ylabel('Displacement [mm]')
plt.xlabel('Time [days]')
plt.show()

input_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_val_scaled = input_scaler.transform(X_val)

t_train_scaled = target_scaler.fit_transform(t_train)
t_val_scaled = target_scaler.transform(t_val)

plt.figure(figsize=(15,5))
plt.plot(X_train_scaled, t_train_scaled, 'o', mec='green', label = 'Training')
plt.plot(X_val_scaled, t_val_scaled, 'o', mec='blue', label = 'Validation')
plt.title('Normalized GNSS dataset')
plt.legend()
plt.ylabel('Normalized displacement [-]')
plt.xlabel('Normalized time [-]')
plt.show()

model_gnss = YOUR CODE HERE

train_losses = YOUR CODE HERE
val_losses = YOUR CODE HERE

epochs = YOUR CODE HERE

for epoch in range(YOUR CODE HERE):
    model_gnss.partial_fit(X_train_scaled, t_train_scaled.flatten())

    # Calculate training loss
    train_pred = YOUR CODE HERE
    train_loss = YOUR CODE HERE
    train_losses.YOUR CODE HERE

    # Calculate validation loss
    val_pred = YOUR CODE HERE
    val_loss = YOUR CODE HERE
    val_losses.YOUR CODE HERE

    # Print losses every 500 epochs
    if epoch % 500 == 0:
        print(f'Epoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', c='b')
plt.plot(val_losses, label='Validation Loss', c='r')
plt.title('Training, Validation, and Test Losses over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

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

new_model_gnss = YOUR CODE HERE
train_losses = YOUR CODE HERE
val_losses = YOUR CODE HERE

epochs = YOUR CODE HERE

for epoch in range(YOUR CODE HERE):
    new_model_gnss.partial_fit(X_train_scaled, t_train_scaled.flatten())

    # Calculate training loss
    train_pred = YOUR CODE HERE
    train_loss = YOUR CODE HERE
    train_losses.YOUR CODE HERE

    # Calculate validation loss
    val_pred = YOUR CODE HERE
    val_loss = YOUR CODE HERE
    val_losses.YOUR CODE HERE

    # Print losses every 500 epochs
    if epoch % 500 == 0:
        print(f'Epoch {epoch}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss', c='b')
plt.plot(val_losses, label='Validation Loss', c='r')
plt.title('Training and Validation Losses over Epochs with a better model')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

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

