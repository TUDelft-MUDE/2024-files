# ----------------------------------------
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ----------------------------------------
data = pd.read_csv(r"data/bridges.csv")

# ----------------------------------------
#summary of the data
data.describe()

# ----------------------------------------
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

# ----------------------------------------
features = loc2['dy'].to_numpy().reshape(-1,1)
targets = loc2['location'].to_numpy().reshape(-1,1)

X_train, X_val_test, t_train, t_val_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=24)

# ----------------------------------------
scaler_x = MinMaxScaler()

scaler_x.fit(YOUR_CODE_HERE)

normalized_X_train = scaler_x.transform(YOUR_CODE_HERE)
normalized_X_val = scaler_x.transform(YOUR_CODE_HERE)



# ----------------------------------------
scaler_t = MinMaxScaler()

scaler_t.fit(YOUR_CODE_HERE)

normalized_t_train = scaler_t.transform(YOUR_CODE_HERE)
normalized_t_val = scaler_t.transform(YOUR_CODE_HERE)



# ----------------------------------------
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

# ----------------------------------------
learning_rate = 0.001
n_epochs = 20
batch_size = 64

# ----------------------------------------
def train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate):
    train_loss_list = []
    val_loss_list = []
    model.learning_rate_init = learning_rate
    
    # Fix random seed for reproducibility
    np.random.seed(42)

    for epoch in range(n_epochs):
        
        # Generate mini-batches
        mini_batches = get_mini_batches(YOUR_CODE_HERE)
        
        # Train model on mini-batches
        for X_batch, t_batch in mini_batches:
            YOUR_CODE_HERE
        
        YOUR_CODE_HERE # Hint: may be more than one line

        # Store loss values
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Print training progress
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss_list[-1]:.4f} - Val Loss: {val_loss:.4f}")
        
    return train_loss_list, val_loss_list

    



# ----------------------------------------
model = MLPRegressor(YOUR_CODE_HERE, YOUR_CODE_HERE)

train_loss_list, val_loss_list = train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate)

# ----------------------------------------
# Create a scatter plot with enhanced styling
plt.figure(figsize=(8, 6))  # Set the figure size

x_axis = list(range(len(train_loss_list)))

#Create a scatter plot
plt.scatter(x_axis, YOUR_CODE_HERE, label='Validation loss', color='red', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.scatter(x_axis, YOUR_CODE_HERE, label='Training loss', color='blue', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

# Add labels and a legend with improved formatting
plt.xlabel('Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Loss curves', fontsize=16, fontweight='bold')
plt.legend(loc='upper right', fontsize=12)

# Set the y-axis to be logarithmic
plt.yscale('log')

# Customize the grid appearance
plt.grid(True, linestyle='--', alpha=0.5)

# Customize the tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a background color to the plot
plt.gca().set_facecolor('#f2f2f2')
plt.show()

# ----------------------------------------
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

# ----------------------------------------
y_train = YOUR_CODE_HERE
y_val = YOUR_CODE_HERE

fig,axes = plt.subplots(1,2,figsize=(8,3))

axes[0].scatter(YOUR_CODE_HERE,YOUR_CODE_HERE,s=0.5)
axes[0].set_xlabel('target crack location [m]')
axes[0].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_train), np.min(y_train))
max_val = max(np.max(t_train), np.max(y_train))
axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[0].legend()

axes[1].scatter(YOUR_CODE_HERE,YOUR_CODE_HERE,s=0.5)
axes[1].set_xlabel('target crack location [m]')
axes[1].set_ylabel('predicted crack location [m]')

min_val = min(np.min(t_val), np.min(y_val))
max_val = max(np.max(t_val), np.max(y_val))
axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit', alpha=0.7)
axes[1].legend()

plt.tight_layout()

# ----------------------------------------
features = np.array([loc1['dy'].to_numpy(), loc2['dy'].to_numpy(), loc3['dy'].to_numpy()]).transpose()
targets = loc2['location'].to_numpy().reshape(-1,1)

# YOUR_CODE_HERE




# ----------------------------------------
# YOUR_CODE_HERE


# ----------------------------------------
# YOUR_CODE_HERE


# ----------------------------------------
layer_sizes = [YOUR_CODE_HERE]
layer_numbers = [YOUR_CODE_HERE]

# Create a grid for the coordinate pairs and store them in an array
val_loss_grid = np.zeros((len(layer_sizes), len(layer_numbers)))

# Loop over all the layer sizes
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


# Extract the hyperparameters that gave the lowest loss and print
min_size, min_number = np.unravel_index(np.argmin(val_loss_grid), val_loss_grid.shape)
print("\n\nModel with {} layers and {} neurons per layer gave lowest loss of {:.4e}".format(layer_numbers[min_number], layer_sizes[min_size], val_loss_grid[min_size, min_number]))

# ----------------------------------------
# Define the row and column labels
rows = layer_sizes
cols = layer_numbers

plt.figure(figsize=(10, 10))
plt.imshow(val_loss_grid, cmap='jet', interpolation='nearest')

# Add a colorbar
plt.colorbar(label='Validation Loss')

# Add the row and column labels
plt.xticks(range(len(cols)), cols)
plt.yticks(range(len(rows)), rows)

plt.xlabel('Number of Layers')
plt.ylabel('Number of Neurons')

plt.title('Validation Loss Grid')
plt.show()

# ----------------------------------------
# Normalize the test inputs
normalized_X_test = scaler_x.transform(X_test)

# Set up NN
# Fix random_state=0 to make sure this is consistent with the model in the loop above
layers = (layer_sizes[min_size],) * layer_numbers[min_number]
model = MLPRegressor(hidden_layer_sizes=layers, activation='tanh', random_state=0)

# train NN
_,  val_loss_list = train_model(model, 
                                normalized_X_train, 
                                normalized_t_train,
                                normalized_X_val, 
                                normalized_t_val, 
                                n_epochs=500, 
                                batch_size=64,
                                learning_rate=0.001
                                )

# YOUR_CODE_HERE



