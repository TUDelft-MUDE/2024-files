import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

file_path = r"data/features_BAK.pk" 
with open(file_path, 'rb') as handle:
    features = pickle.load(handle)

file_path = r"data/targets_BAK.pk"
with open(file_path, 'rb') as handle:
    targets = pickle.load(handle)

print('Dimensions of features (X):', features.shape)
print('Dimensions of targets  (t):', targets.shape)

X_train, X_val_test, t_train, t_val_test = train_test_split(features, targets, test_size=0.20, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(X_val_test, t_val_test, test_size=0.50, random_state=24)

scaler_diameters = MinMaxScaler()
scaler_diameters.fit(X_train)

normalized_X_train = scaler_diameters.transform(X_train)
normalized_X_val = scaler_diameters.transform(X_val)

scaler_pressures = MinMaxScaler()
scaler_pressures.fit(t_train)

normalized_t_train = scaler_pressures.transform(t_train)
normalized_t_val = scaler_pressures.transform(t_val)

model = MLPRegressor(hidden_layer_sizes = (10, 5), 
                    activation = 'tanh')

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

    
        
        
        

        

def train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate):
    train_loss_list = []
    val_loss_list = []
    model.learning_rate_init = learning_rate
    
    for epoch in range(n_epochs):
        
        # Generate mini-batches
        mini_batches = get_mini_batches(normalized_X_train, normalized_t_train, batch_size)
        
        # Train model on mini-batches
        for X_batch, t_batch in mini_batches:
            model.partial_fit(X_batch, t_batch)
        
        # Compute loss on training and validation sets
        train_loss = mean_squared_error(normalized_t_train, model.predict(normalized_X_train))
        
        # Compute loss on validation set
        val_loss = mean_squared_error(normalized_t_val, model.predict(normalized_X_val))

        # Store loss values
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Print training progress
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss_list[-1]:.4f} - Val Loss: {val_loss:.4f}")
        
    return train_loss_list, val_loss_list

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

model.score(normalized_X_train, normalized_t_train)

model.score(normalized_X_val, normalized_t_val)

layer_sizes = [10, 20, 50, 100] 
layer_numbers = [1, 2, 3, 4]

val_loss_grid = np.zeros((len(layer_sizes), len(layer_numbers)))

for i, lsize in enumerate(layer_sizes):
    
    # Loop over all numbers of hidden layers
    for j, lnumber in enumerate(layer_numbers):
    
        # get tuple of hidden layer sizes
        layers = (lsize,) * lnumber
        print("Training NN with hidden layers:  {}".format(layers))
        
        # Create the ANN model with the given hidden layer sizes and activation function
        model = MLPRegressor(hidden_layer_sizes=layers, activation='tanh')
        
        _,  val_loss_list = train_model(model, 
                                        normalized_X_train, 
                                        normalized_t_train,
                                        normalized_X_val, 
                                        normalized_t_val, 
                                        n_epochs=20, 
                                        batch_size=64,
                                        learning_rate=0.001
                                        )
    
        val_loss_grid[i,j] = val_loss_list[-1]
        
        print("     Loss:    {:.4e}\n".format(val_loss_grid[i,j]))

min_size, min_number = np.unravel_index(np.argmin(val_loss_grid), val_loss_grid.shape)
print("\n\nModel with {} layers and {} neurons per layer gave lowest loss of {:.4e}".format(layer_numbers[min_number], layer_sizes[min_size], val_loss_grid[min_size, min_number]))

layers = (layer_sizes[min_size],) * layer_numbers[min_number]
model = MLPRegressor(hidden_layer_sizes=layers, activation='tanh')

_,  val_loss_list = train_model(model, 
                                normalized_X_train, 
                                normalized_t_train,
                                normalized_X_val, 
                                normalized_t_val, 
                                n_epochs=20, 
                                batch_size=64,
                                learning_rate=0.001
                                )

import matplotlib.pyplot as plt

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

normalized_X_test = scaler_diameters.transform(X_test)
normalized_t_test = scaler_pressures.transform(t_test)

model.score(normalized_X_test, normalized_t_test)

estimated_pressure = scaler_pressures.inverse_transform(model.predict(normalized_X_test))
error = t_test - estimated_pressure

node_ID = 0
error_node = error[:, node_ID]

fig, ax = plt.subplots()

ax.hist(error_node, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

ax.set_xlabel('Error of prediction [m]', labelpad=15, color='#333333')
ax.set_ylabel('Frequency', labelpad=15, color='#333333')
ax.set_title(f'Error of node {node_ID} across test scenarios', pad=15, color='#333333', weight='bold')

fig.tight_layout()

start_time = time.time()
y_pred_test = model.predict(X_test)
total_time = time.time() - start_time

num_test_sims = len(y_pred_test)

data_driven_exec_time_per_sim = total_time/num_test_sims
print(f'Data-driven model took {data_driven_exec_time_per_sim:.7f} seconds for {num_test_sims} scenarios')

original_time_per_sim = 0.04

speed_up = np.round(original_time_per_sim/data_driven_exec_time_per_sim, 2)
print('The data-driven model is', speed_up,'times faster than original simulator per scenario.')

diameters = np.array([[800, 1000, 1100, 600, 450,
                       900, 700, 700, 300, 1100,
                       400, 700, 350, 1100, 600,
                       450, 400, 350, 1100, 900,
                       600, 600, 1100, 700, 500,
                       400, 450, 350, 700, 350,
                       1000, 400, 400, 400, 350,
                       900, 300, 1000, 400, 300,
                       450, 400, 450, 350, 1100,
                       900, 450, 800, 800, 300,
                       1100, 600, 300, 700, 1000,
                       1000, 800, 800]])
normalized_diameters = scaler_diameters.transform(diameters)

normalized_predictions = model.predict(normalized_diameters)

predictions = scaler_pressures.inverse_transform(normalized_predictions)

print(f'{predictions.min():.3f}')
print(f'{predictions.max():.3f}')
print(f'{predictions.mean():.3f}')

