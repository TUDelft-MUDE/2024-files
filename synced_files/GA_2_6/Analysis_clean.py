# ---

# ---

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %% [markdown]

# %% [markdown]

# %%
file_path = r"data/features_BAK.pk" 
with open(file_path, 'rb') as handle:
    features = pickle.load(handle)

file_path = r"data/targets_BAK.pk"
with open(file_path, 'rb') as handle:
    targets = pickle.load(handle)

# %% [markdown]

# %%
print('Dimensions of features (X):', features.shape)
print('Dimensions of targets  (t):', targets.shape)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown] id="0491cc69"

# %%
X_train, X_val_test, t_train, t_val_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=24)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
scaler_diameters = MinMaxScaler()
scaler_diameters.fit(X_train)

normalized_X_train = scaler_diameters.transform(X_train)
normalized_X_val = scaler_diameters.transform(X_val)

# %% [markdown]

# %%
scaler_pressures = MinMaxScaler()
scaler_pressures.fit(t_train)

normalized_t_train = scaler_pressures.transform(t_train)
normalized_t_val = scaler_pressures.transform(t_val)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
model = MLPRegressor(YOUR_CODE_HERE, YOUR_CODE_HERE)

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
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
    
    perm = np.random.permutation(len(X))
    X_train_perm = X[perm]
    t_train_perm = t[perm]
    
    
    X_batches = []
    t_batches = []
    for i in range(0, len(X_train_perm), batch_size):
        X_batches.append(X_train_perm[i:i+batch_size])
        t_batches.append(t_train_perm[i:i+batch_size])

    return list(zip(X_batches, t_batches))

# %% [markdown]

# %%
learning_rate = 0.001
n_epochs = 20
batch_size = 64

# %% [markdown]

# %%
def train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate):
    train_loss_list = []
    val_loss_list = []
    model.learning_rate_init = learning_rate
    
    for epoch in range(n_epochs):
        
        
        mini_batches = get_mini_batches(YOUR_CODE_HERE)
        
        
        for X_batch, t_batch in mini_batches:
            YOUR_CODE_HERE
        
        YOUR_CODE_HERE 

        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss_list[-1]:.4f} - Val Loss: {val_loss:.4f}")
        
    return train_loss_list, val_loss_list

# %%
train_loss_list, val_loss_list = train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate)

# %% [markdown]

# %%

plt.figure(figsize=(8, 6))  

x_axis = list(range(len(train_loss_list)))

plt.scatter(x_axis, YOUR_CODE_HERE, label='Validation loss', color='red', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
plt.scatter(x_axis, YOUR_CODE_HERE, label='Training loss', color='blue', marker='.', s=100, alpha=0.7, edgecolors='black', linewidths=0.5)

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

# %% [markdown]

# %%
model.score(YOUR_CODE_HERE, YOUR_CODE_HERE)

# %%
model.score(YOUR_CODE_HERE, YOUR_CODE_HERE)

# %% [markdown]

# %% [markdown]

# %% [markdown] id="0491cc69"

# %% [markdown]

# %%

layer_sizes = [10, 20, 50, 100] 
layer_numbers = [1, 2, 3, 4]

val_loss_grid = np.zeros((len(layer_sizes), len(layer_numbers)))

for i, lsize in enumerate(layer_sizes):
    
    
    for j, lnumber in enumerate(layer_numbers):
    
        
        layers = (lsize,) * lnumber
        print("Training NN with hidden layers:  {}".format(layers))
        
        
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

# %% [markdown]

# %% [markdown]

# %%

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

# %% [markdown]

# %% [markdown]

# %%
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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
normalized_X_test = scaler_diameters.transform(X_test)
normalized_t_test = scaler_pressures.transform(t_test)

# %%
model.score(normalized_X_test, normalized_t_test)

# %% [markdown]

# %%
estimated_pressure = scaler_pressures.inverse_transform(model.predict(normalized_X_test))
error = t_test - estimated_pressure

# %%
node_ID = 0
error_node = error[:, node_ID]

# %% [markdown]

# %%
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

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
start_time = time.time()
y_pred_test = model.predict(X_test)
total_time = time.time() - start_time

num_test_sims = len(y_pred_test)

data_driven_exec_time_per_sim = total_time/num_test_sims
print(f'Data-driven model took {data_driven_exec_time_per_sim:.7f} seconds for {num_test_sims} scenarios')

# %% [markdown]

# %%
original_time_per_sim = 0.04

speed_up = np.round(original_time_per_sim/data_driven_exec_time_per_sim, 2)
print('The data-driven model is', speed_up,'times faster than original simulator per scenario.')

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
YOUR_CODE_HERE 

# %% [markdown]

# %% [markdown]

