
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"data/bridges.csv")

data.describe()

from matplotlib import cm

loc1 = data[data['node'] == 16] # point at 25% span
loc2 = data[data['node'] == 29] # point at midspan
loc3 = data[data['node'] == 41] # point at 75% span

fig,axes = plt.subplots(1,3,figsize=(10,3))

axes[0].scatter(loc1['dy'], loc1['location'], c=loc1['sample'], s=0.5)
axes[0].set_xlabel('displacement at 25% span [m]')
axes[0].set_ylabel('crack location [m]')

axes[1].scatter(loc2['dy'], loc1['location'], c=loc2['sample'], s=0.5)
axes[1].set_xlabel('displacement at 50% span [m]')
axes[1].set_ylabel('crack location [m]')

axes[2].scatter(loc3['dy'], loc1['location'], c=loc3['sample'], s=0.5)
axes[2].set_xlabel('displacement at 75% span [m]')
axes[2].set_ylabel('crack location [m]')

plt.tight_layout()
plt.show()

features = loc2['dy'].to_numpy().reshape(-1,1)
targets = loc2['location'].to_numpy().reshape(-1,1)

X_train, X_val_test, t_train, t_val_test = train_test_split(features, targets, test_size=0.20, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(X_val_test, t_val_test, test_size=0.50, random_state=24)

scaler_x = MinMaxScaler()

scaler_x.fit(X_train)

normalized_X_train = scaler_x.transform(X_train)
normalized_X_val = scaler_x.transform(X_val)

scaler_t = MinMaxScaler()

scaler_t.fit(t_train)

normalized_t_train = scaler_t.transform(t_train)
normalized_t_val = scaler_t.transform(t_val)

def get_mini_batches(X, t, batch_size):
    """
    This function generates mini-batches from the given input data and labels.

    Parameters:
    X (numpy.ndarray): The features.
    t (numpy.ndarray): The targets corresponding to the input data.
    batchsize (int): The size of each mini-batch.

    Returns:
    list: A list of tuples where each tuple contains a mini-batch of the input data and the corresponding targets.
    """
    # Generate permutations
    perm = np.random.permutation(len(X))
    X_train_perm = X[perm]
    t_train_perm = t[perm]
    
    # Generate mini-batches
    X_batches = []
    t_batches = []
    for i in range(0, len(X_train_perm), batch_size):
        X_batches.append(X_train_perm[i:i+batch_size])
        t_batches.append(t_train_perm[i:i+batch_size])

    return list(zip(X_batches, t_batches))

learning_rate = 0.001
n_epochs = 20
batch_size = 64

def train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate, verbose=True):
    train_loss_list = []
    val_loss_list = []
    model.learning_rate_init = learning_rate

    # Fix random seed for reproducibility
    np.random.seed(42)
    
    for epoch in range(n_epochs):
        
        # Generate mini-batches
        mini_batches = get_mini_batches(normalized_X_train, normalized_t_train, batch_size)
        
        # Train model on mini-batches
        for X_batch, t_batch in mini_batches:
            model.partial_fit(X_batch, t_batch.flatten())
        
        # Compute loss on training and validation sets
        train_loss = mean_squared_error(normalized_t_train, model.predict(normalized_X_train))
        
        # Compute loss on validation set
        val_loss = mean_squared_error(normalized_t_val, model.predict(normalized_X_val))

        # Store loss values
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Print training progress
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss_list[-1]:.4f} - Val Loss: {val_loss:.4f}")
        
    return train_loss_list, val_loss_list

model = MLPRegressor(hidden_layer_sizes = (10, 5), 
                    activation = 'tanh',
                    random_state=1)

train_loss_list, val_loss_list = train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate)

plt.figure(figsize=(8, 6))  # Set the figure size

x_axis = list(range(len(train_loss_list)))

plt.scatter(x_axis, val_loss_list, label='Validation loss', color='red', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.scatter(x_axis, train_loss_list, label='Training loss', color='blue', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

plt.xlabel('Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Loss curves', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)

plt.yscale('log')

plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.gca().set_facecolor('#f2f2f2')
plt.show()

normalized_X_range = np.linspace(0,1,100).reshape(-1,1)
X_range = scaler_x.inverse_transform(normalized_X_range)

normalized_y_range = model.predict(normalized_X_range).reshape(-1,1)
y_range = scaler_t.inverse_transform(normalized_y_range)

fig,axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(X_range, y_range, label=r"Network $y(x)$", color='k')
axes[0].scatter(X_train,t_train,s=0.5, label='Training data')
axes[0].set_xlabel('displacement at 50% span [m]')
axes[0].set_ylabel('crack location [m]')

axes[1].plot(X_range, y_range,label=r"Network $y(x)$", color='k')
axes[1].scatter(X_val,t_val,s=0.5, label='Validation data')
axes[1].set_xlabel('displacement at 50% span [m]')
axes[1].set_ylabel('crack location [m]')

axes[0].legend()
axes[1].legend()
plt.tight_layout()

y_train = scaler_t.inverse_transform(model.predict(normalized_X_train).reshape(-1,1))
y_val = scaler_t.inverse_transform(model.predict(normalized_X_val).reshape(-1,1))

fig,axes = plt.subplots(1,2,figsize=(8,3))

axes[0].scatter(t_train,y_train,s=0.5)

axes[0].set_title('Training dataset')
axes[0].set_xlabel('target crack location [m]')
axes[0].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_train), np.min(y_train))
max_val = max(np.max(t_train), np.max(y_train))
axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[0].legend()

axes[1].scatter(t_val,y_val,s=0.5)

axes[1].set_title('Validation dataset')
axes[1].set_xlabel('target crack location [m]')
axes[1].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_val), np.min(y_val))
max_val = max(np.max(t_val), np.max(y_val))
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[1].legend()

plt.tight_layout()

features = np.array([loc1['dy'].to_numpy(), loc2['dy'].to_numpy(), loc3['dy'].to_numpy()]).transpose()
targets = loc2['location'].to_numpy().reshape(-1,1)

X_train, X_val_test, t_train, t_val_test = train_test_split(features, targets, test_size=0.20, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(X_val_test, t_val_test, test_size=0.50, random_state=24)

scaler_x = MinMaxScaler()
scaler_x.fit(X_train)

normalized_X_train = scaler_x.transform(X_train)
normalized_X_val = scaler_x.transform(X_val)

scaler_t = MinMaxScaler()
scaler_t.fit(t_train)

normalized_t_train = scaler_t.transform(t_train)
normalized_t_val = scaler_t.transform(t_val)

model = MLPRegressor(hidden_layer_sizes = (10, 5), 
                    activation = 'tanh')

learning_rate = 0.001
n_epochs = 200
batch_size = 64

train_loss_list, val_loss_list = train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate)

plt.figure(figsize=(8, 6))  # Set the figure size

x_axis = list(range(len(train_loss_list)))

plt.scatter(x_axis, val_loss_list, label='Validation loss', color='red', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.scatter(x_axis, train_loss_list, label='Training loss', color='blue', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

plt.xlabel('Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Loss curves', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)

plt.yscale('log')

plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.gca().set_facecolor('#f2f2f2')
plt.show()

y_train = scaler_t.inverse_transform(model.predict(normalized_X_train).reshape(-1,1))
y_val = scaler_t.inverse_transform(model.predict(normalized_X_val).reshape(-1,1))

fig,axes = plt.subplots(1,2,figsize=(8,3))

axes[0].scatter(t_train,y_train,s=0.5)

axes[0].set_title('Training dataset')
axes[0].set_xlabel('target crack location [m]')
axes[0].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_train), np.min(y_train))
max_val = max(np.max(t_train), np.max(y_train))
axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[0].legend()

axes[1].scatter(t_val,y_val,s=0.5)

axes[1].set_title('Validation dataset')
axes[1].set_xlabel('target crack location [m]')
axes[1].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_val), np.min(y_val))
max_val = max(np.max(t_val), np.max(y_val))
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[1].legend()

plt.tight_layout()

layer_sizes = [10, 20] 
layer_numbers = [1, 2, 3, 4, 5, 6]

val_loss_grid = np.zeros((len(layer_sizes), len(layer_numbers)))

for i, lsize in enumerate(layer_sizes):
    
    # Loop over all numbers of hidden layers
    for j, lnumber in enumerate(layer_numbers):
    
        # get tuple of hidden layer sizes
        layers = (lsize,) * lnumber
        print("Training NN with hidden layers:  {}".format(layers))
        
        # Create the ANN model with the given hidden layer sizes and activation function
        # Fix random_state to make sure results are reproducible
        model = MLPRegressor(hidden_layer_sizes=layers, activation='relu', random_state=0)
        
        _,  val_loss_list = train_model(model, 
                                        normalized_X_train, 
                                        normalized_t_train,
                                        normalized_X_val, 
                                        normalized_t_val, 
                                        n_epochs=500, 
                                        batch_size=64,
                                        learning_rate=0.001,
                                        verbose=False
                                        )
    
        val_loss_grid[i,j] = val_loss_list[-1]
        
        print("     Final validation loss:    {:.4e}\n".format(val_loss_grid[i,j]))

min_size, min_number = np.unravel_index(np.argmin(val_loss_grid), val_loss_grid.shape)
print("\n\nModel with {} layers and {} neurons per layer gave lowest loss of {:.4e}".format(layer_numbers[min_number], layer_sizes[min_size], val_loss_grid[min_size, min_number]))

rows = layer_sizes
cols = layer_numbers

plt.figure(figsize=(10, 10))
plt.imshow(val_loss_grid, cmap='jet', interpolation='nearest')

plt.colorbar(label='Validation Loss')

plt.xticks(range(len(cols)), cols)
plt.yticks(range(len(rows)), rows)

plt.xlabel('Number of Layers')
plt.ylabel('Number of Neurons')

plt.title('Validation Loss Grid')
plt.show()

normalized_X_test = scaler_x.transform(X_test)

layers = (layer_sizes[min_size],) * layer_numbers[min_number]
model = MLPRegressor(hidden_layer_sizes=layers, activation='tanh', random_state=0)

_,  val_loss_list = train_model(model, 
                                normalized_X_train, 
                                normalized_t_train,
                                normalized_X_val, 
                                normalized_t_val, 
                                n_epochs=500, 
                                batch_size=64,
                                learning_rate=0.001
                                )

y_train = scaler_t.inverse_transform(model.predict(normalized_X_train).reshape(-1,1))
y_val = scaler_t.inverse_transform(model.predict(normalized_X_val).reshape(-1,1))
y_test = scaler_t.inverse_transform(model.predict(normalized_X_test).reshape(-1,1))

fig,axes = plt.subplots(1,3,figsize=(10,3))

axes[0].scatter(t_train,y_train,s=0.5)

axes[0].set_title('Training dataset')
axes[0].set_xlabel('target crack location [m]')
axes[0].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_train), np.min(y_train))
max_val = max(np.max(t_train), np.max(y_train))
axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[0].legend()

axes[1].scatter(t_val,y_val,s=0.5)

axes[1].set_title('Validation dataset')
axes[1].set_xlabel('target crack location [m]')
axes[1].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_val), np.min(y_val))
max_val = max(np.max(t_val), np.max(y_val))
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[1].legend()

axes[2].scatter(t_test,y_test,s=0.5)

axes[2].set_title('Test dataset')
axes[2].set_xlabel('target crack location [m]')
axes[2].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_test), np.min(y_test))
max_val = max(np.max(t_test), np.max(y_test))
axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[2].legend()

plt.tight_layout()

