# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Project 10: Handling the pressure - Machine learning for predicting pressure in Water Distribution Systems
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px; height: auto; margin: 0"\>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px; height: auto; margin: 0"\>
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.6. Due: Friday, Dec 22, 2023.*

# ## 📝 Specifications

# This notebook is divided into five parts:
# 1) Data pre-processing.
# 2) Defining and training a multilayer perceptron (MLP).
# 3) Optimization of the MLP hyperparameters.
# 4) Model assessment.
# 5) Model usage.
#
# **Completition requirements:**
# By the end of this notebook, you should have:
# - Implemented all the code cells for:
#   - Splitting the data into training, validation, and testing sets
#   - Normalizing the data
#   - Instantiating an MLP
#   - Training the MLP with a training loop
#   - Defining a grid-search hyperoptimization
#   - Assess the accuracy of the MLP
#   - Assess the speed of the MLP
#   - Use it to predict the pressure of a particular example
# - Generated and exported all of the relevant plots for the report
# - Answered all the questions in the report
#
# *Complete this assignment by the end of the session at 12:30. This means having a single report for your group with all the plots, analysis and interpretation completed.*
#
# **Working method:**
#
# Each of the parts of the notebook can be coded independently. However, in order to run the code in each part, the code in the previous parts should be in place.

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# This notebook includes boxes with the formatting shown here to list the questions you are expected to answer in your report. You are not expected to write your answers here.

# ## 🔙 Background

# ### 💧 Water distribution systems

# A water distribution system transports water from sources, like wells or reservoirs, to various locations where water is needed, like homes, shops, and factories. 
#
# A basic system consists of sources of water supply and demand points for water connected by pipe lines. Figure 1 shows an example system where there are two supply centers and ten demand nodes. This transmission system can connect sparse populations, and it can be considered as a simple network of one reservoir and few nodes and pipes. Nevertheless, in a city of moderate size, there may be a number of supply centers and hundreds of demand points.

# <div style="display: flex; flex-direction: row;">
#     <div style="flex: 50%;">
#         <center>
#             <img src="./figs/WDSAsset 1v1.png" width="400"/>
#             <figcaption><b>Figure 1.</b> Simplified scheme of a branched water distribution system.</figcaption>
#         </center>
#     </div>
#     <div style="flex: 50%;">
#         <center>
#             <img src="./figs/BAK.png" width="700"/>
#             <figcaption><b>Figure 2.</b> Numerical results for the BakRyan water distribution system.</figcaption>
#         </center>
#     </div>
# </div>

# Water utilities rely on hydrodynamic models to properly design and control water distribution systems (WDSs). These physically-based models compute the  pressures at all the junctions, as illustrated in Figure 2. In this figure, we can see the water network of BakRyan with the pressure at each node of the network represented by the colour. In water distribution systems, pressure is a fundamental variable. Without sufficient pressure in the system, the network is not able to supply water to the users. 
#
# We can use pressure estimations to ensure proper water pressure, efficient flow, and reliable distribution of water to consumers. For obtaining these estimations, we tipically use hydrodynamic models. However, the computational speed of these models is often insufficient for some applications in civil engineering such as optimisation or real-time control, especially in large networks. 
#
# One alternative to address this issue is developing data-driven models. These models can be trained using results from simulations done with the physically-based model. The objective of the data-driven model we will develop is to estimate pressure at each node of the water network but in a shorter time.

# ## ✅ Application

# In this notebook, you will create a Multilayer Perceptron (MLP) for estimating the nodal pressures from the BakRyan water distribution system (Figure 2). This system has 58 pipes and 35 nodes. Your task is to create and train this Artificial Neural Network, exemplified in Figure 3. The MLP should estimate the pressures while being faster to run than the physically-based model (which usually takes 0.04 seconds to run per simulation). Furthermore, you will hyperoptimize the MLP to improve its performance.
#
# **Your input data will be a vector of pipe diameters. The output data will be a vector of nodal pressures.**
#
# Mathematically, we can express the our application as follows:
#
# $$
# y = \phi(x; W)
# $$
#
# where:
#
# $y$: output data (nodal pressures, units: mwc*)
#
# $\phi$: represents the Artificial Neural Network
#
# $x$: input data (pipe diameters, units: m)
#
# $W$: parameters of the MLP (unitless)
#
# Having pairs of input-output data (diameters, $x$, and pressures, $y$), it is our task to find the set of parameters, $W$, that best fit the data. Note that there are no observed pressures at some of the nodes.
#
# _*In water engineering, it is common to express the value of the pressure in meters of water column (mwc). This unit is equivalent to the pressure exerted by a column of water of 1 meter in height._

# <div>
# <center><img src="./figs/ANN_image2.png" width="600"/>
# <figcaption><b>Figure 3.</b> Artificial neural network representation, with a zoomed-in view of how a single neuron works. For this notebook, we have 58 input features (pipe diameters) and 35 outputs (nodal pressures).</figcaption></center>
# </div>

# ## 📔 Preliminaries

# ### Libraries

# To run this notebook you need to have installed the following packages:
# - Numpy
# - Matplotlib
# - Pickle
# - Scikit-learn

# +
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# -

# ### Load the database 

# For the purposes of this notebook, there is an already existing database that you can use to create and train the MLPs.

# +
file_path = r"data/features_BAK.pk" 
with open(file_path, 'rb') as handle:
    features = pickle.load(handle)

file_path = r"data/targets_BAK.pk"
with open(file_path, 'rb') as handle:
    targets = pickle.load(handle)
# -

# We can explore the content of each of these variables. In total, we collected 10000 examples of the BakRyan system with random configurations of the available diameters (of the 58 pipes). 
#
# As input features (X), we use the diameters of all the pipes in the network, and each configuration of diameters is related 1-1 with the pressure at all the nodes in the system.

print('Dimensions of features (X):', features.shape)
print('Dimensions of targets  (t):', targets.shape)

# ## 1. Data Pre-Processing

# ### Splitting the data into training, validation, and testing sets

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 1.1:</b>   
#
# In machine learning, it's common to split the dataset into three parts: a training set, a validation set, and a test set. 
#
# Your task is to write a Python code snippet that splits a given dataset into these three parts. The dataset consists of `features` and `targets`.
#
# The dataset should be split as follows:
#
# - 80% of the data should go to the training set.
# - 10% of the data should go to the validation set.
# - 10% of the data should go to the test set.
#
# The splitting should be done in a way that shuffles the data first to ensure that the training, validation, and test sets are representative of the overall distribution of the data. You can set the random state for the shuffling to ensure that the results are reproducible; you can use the values 42 for the first split and 24 for the second split.
#
# The resulting training, validation, and test sets should be stored in the variables `X_train`, `t_train`, `X_val`, `t_val`, `X_test`, and `t_test`.
#
# **Hint:** You can use the `train_test_split` function from the `sklearn.model_selection` module to perform the splitting.

# + [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>Note here that the training, validation and test sets are created in two steps, due to the way <code>train_test_split</code> is implemented in <code>sklearn</code>. Thus, in the second split you value used for <code>test_size</code> should <b>not</b> be 0.10!</p></div>
# -

X_train, X_val_test, t_train, t_val_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=42)
X_val, X_test, t_val, t_test = train_test_split(YOUR_CODE_HERE, YOUR_CODE_HERE, test_size=YOUR_CODE_HERE, random_state=24)

# ### Normalizing the data

# Now, we normalize the data using the MinMaxScaler from scikit-learn. This scaler transforms the data to be between 0 and 1. This is important because the ANN will be trained using the gradient descent algorithm, which is sensitive to the scale of the data. Notice that we use the training data to fit the scaler. This is important because we assume that the model only sees the training data and we do not use any of the validation or testing data.

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 1.2:</b>   
#
# In machine learning, it's often beneficial to normalize the feature variables to a specific range. This can help the model converge faster during training and can also prevent certain features from dominating others due to their scale.
#
# Your task is to write a Python code snippet that normalizes the feature variables of a training set and a validation set to the range [0, 1]. The feature variables are stored in the variables `X_train` and `X_val`.
#
# You should use the `MinMaxScaler` class from the `sklearn.preprocessing` module to perform the normalization. This class scales and translates each feature individually such that it is in the given range on the training set.
#
# The normalized features should be stored in the variables `normalized_X_train` and `normalized_X_val`.
#
# <em>Note: we do this task for you.</em>

# The `MinMaxScaler` should be fitted on the training features only. 

# +
scaler_diameters = MinMaxScaler()
scaler_diameters.fit(X_train)

normalized_X_train = scaler_diameters.transform(X_train)
normalized_X_val = scaler_diameters.transform(X_val)
# -

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 1.3:</b>   
#
# Your task is to write a Python code snippet that normalizes the target variables of a training set and a validation set to the range [0, 1]. The target variables are stored in the variables `t_train` and `t_val`.
#
# The normalized targets should be stored in the variables `normalized_t_train` and `normalized_t_val`.
#
# <em>Note: we do this task for you.</em>

# +
scaler_pressures = MinMaxScaler()
scaler_pressures.fit(t_train)

normalized_t_train = scaler_pressures.transform(t_train)
normalized_t_val = scaler_pressures.transform(t_val)
# -

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# 1.1) What is the purpose of splitting a dataset into training, validation, and test sets in the context of machine learning?
#
# 1.2) What part of the pre-processing improves the representativity of the overall distribution of the data?
#
# 1.3) Why should the `MinMaxScaler` be fitted on the training data only, and then used to transform both the training and validation data?

# ## 2. Defining and training an MLP

# Now, we will define a Multilayer Perceptron (MLP). In Scikit-learn, the MLP is defined in the MLPRegressor class, you can see the documentation [here](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html). This class has many hyperparameters that can be tuned to improve the performance of the model. Notice that in Scikit-learn, the model and the optimizer are defined in the same class. This means that we do not need to define an optimizer separately; therefore, some hyperparameters are related to the optimizer. For example, the learning rate. We will indicate the optimization hyperparameters in the next section; for now, we will only define the model hyperparameters.

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 2.1:</b>   
# You are tasked with setting up a Multi-Layer Perceptron (MLP). The MLP should have the following characteristics:
#
#  - The hidden layer sizes are defined as a tuple. For example, if we want to have two hidden layers with 10 and 5 neurons, respectively, we would write: hidden_layer_sizes=(10,5). Notice that we only specify the hidden layer sizes, the input and output sizes will be automatically inferred when we train the model. 
#  - The activation function can be one of the following: 'identity', 'logistic', 'tanh', 'relu'.
#
# The configured MLP regressor should be stored in a variable named `model`.

model = MLPRegressor(YOUR_CODE_HERE, YOUR_CODE_HERE)


# Now that we have a model, we need to train it! Now, we will define a training loop that will train the model using the training data and will evaluate the model using the validation data.

# ### Training the model

# Scikit-learn offers the possibility to directly train a model using the `fit` method. However, we will define a training loop to have more control over the training process. This will allow us to evaluate the model at each epoch and observe its training.
#
# The first step towards training a model is defining a function that transforms our training dataset into random mini-batches. This is a common practice used for training neural networks due to their computational efficiency and their ability to help the model generalize better. This practice generally leads to better computational efficiency, faster convergence and better generalization performance.

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


# The following figure illustrates both the way we split the original dataset and how we further split the training dataset into mini-batches. At every epoch the training dataset is shuffled and each mini-batch is considered **in isolation** by the network. The gradients coming from the single mini-batches are used to update the weights of the network (the randomness involved is why we say we are using **Stochastic** Gradient Descent).
#
# <div>
# <center><img src="./figs/minibatching.png" width="600"/>
# <figcaption><b>Figure 4.</b> Dataset splitting, mini-batching and the stochastic nature of MLP training.</figcaption></center>
# </div>
#
# Now, we will define some hyperparameters for the training loop. These hyperparameters are related to the optimization process.
# Define the following hyperparameters:
# - `learning_rate` (float): The learning rate of the optimizer.
# - `n_epochs` (int): The number of epochs to train the model. (For time reasons, we will only train the model for 20 epochs. However, you can increase this number to improve the performance of the model.)
# - `batch_size` (int): The size of each mini-batch.

learning_rate = 0.001
n_epochs = 20
batch_size = 64


# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 2.2:</b> 
#
# In this exercise, you are tasked with implementing a function to train a neural network model. The function should also compute and store the loss on the training and validation sets at each epoch. The loss function to be used is the Mean Squared Error (MSE), which is defined as:
#
# $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (t_i - y_i)^2 $$
#
# where $t_i$ is the actual target, $y_i$ is the predicted value, and $n$ is the number of samples.
#
# The function should be named `train_model` and should take the following parameters:
#
# - `model`: An instance of a neural network model that we want to train.
# - `normalized_X_train`: The normalized training data.
# - `normalized_t_train`: The normalized training labels.
# - `normalized_X_val`: The normalized validation data.
# - `normalized_t_val`: The normalized validation labels.
# - `n_epochs`: The number of epochs to train the model for.
# - `batch_size`: The size of each mini-batch.
# - `learning_rate`: The learning rate for the model.
#
# The function should perform the following steps:
#
# 1. Initialize two empty lists, `train_loss_list` and `val_loss_list`, to store the training and validation losses at each epoch.
#
# 2. Loop over the specified number of epochs. For each epoch:
#
#     a. Generate mini-batches from the normalized training data and labels using a function `get_mini_batches(normalized_X_train, normalized_t_train, batch_size)`.
#
#     b. For each mini-batch, update the model's weights using the `partial_fit` method of the model.
#
#     c. Compute the MSE loss on the training set and append it to `train_loss_list`.
#
#     d. Compute the MSE loss on the validation set and append it to `val_loss_list`.
#
#     e. Print the training progress including the current epoch and the training and validation losses.
#
#     f. Return the `train_loss_list` and `val_loss_list` lists.
#
# Your task is to write the Python code that implements the `train_model` function.

def train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate):
    train_loss_list = []
    val_loss_list = []
    model.learning_rate_init = learning_rate
    
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


train_loss_list, val_loss_list = train_model(model, normalized_X_train, normalized_t_train, normalized_X_val, normalized_t_val, n_epochs, batch_size, learning_rate)

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 2.3:</b>   
#     Plot the validation and training loss curves. Add this plot to your report.
# </div>

# +
# Create a scatter plot with enhanced styling
plt.figure(figsize=(8, 6))  # Set the figure size

x_axis = list(range(len(train_loss_list)))

# Create a scatter plot
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
# -

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 2.4:</b>   
# Using the `score` function of the MLP Regressor, compute the R2 score of the model on the training and validation sets.

model.score(YOUR_CODE_HERE, YOUR_CODE_HERE)

model.score(YOUR_CODE_HERE, YOUR_CODE_HERE)

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# 2.1) Based on the shape of the loss curves, what can you indicate about the fitting capabilities of the model? (Is it overfitting, underfitting, or neither?)
#
# 2.2) How do you explain the difference between the values of training and validation score?

# ## 3. Hyperparameter tuning

# + [markdown] id="0491cc69"
# <div style="background-color:#facb8e; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%"> <p>For Tasks 3 and 4 the code is 100% complete, but you are expected to read it thoroughly to help understand the analysis and provide answers in your report.</p></div>
# -

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 3.1:</b>   
#     Create a grid-search strategy to find hyperparameters that give the best prediction on the validation set. Vary the number of layers and number of hidden units per layer. You can assume that all the hidden layers have the same number of hidden units.
# </div>

# +
# define coordinate vectors for grid
layer_sizes = [10, 20, 50, 100] 
layer_numbers = [1, 2, 3, 4]

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


# Extract the hyperparameters that gave the lowest loss and print
min_size, min_number = np.unravel_index(np.argmin(val_loss_grid), val_loss_grid.shape)
print("\n\nModel with {} layers and {} neurons per layer gave lowest loss of {:.4e}".format(layer_numbers[min_number], layer_sizes[min_size], val_loss_grid[min_size, min_number]))
# -

# Let's use our test data to visualize our best-performing model and test its predictive capabilities. First, re-initialize & train the model with the optimal hyperparameters.

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 3.2:</b>   
#     Re-initialize & train the model with the optimal hyperparameters.
#     
# The reconfigured MLP regressor should be stored in a variable named `model`.

# +
# Set up NN
layers = (layer_sizes[min_size],) * layer_numbers[min_number]
model = MLPRegressor(hidden_layer_sizes=layers, activation='tanh')

# train NN
_,  val_loss_list = train_model(model, 
                                normalized_X_train, 
                                normalized_t_train,
                                normalized_X_val, 
                                normalized_t_val, 
                                n_epochs=20, 
                                batch_size=64,
                                learning_rate=0.001
                                )
# -

# Here is the Python code to plot the matrix `val_loss_grid` with the specified row and column labels:
#
#

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 3.3:</b>   
#     Plot the validation loss grid. Add this plot to your report.
# </div>

# +
import matplotlib.pyplot as plt

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
# -

#
#
# This code will create a heatmap where the color intensity represents the validation loss. The colorbar on the side provides a reference for the loss values. The row and column labels represent the number of neurons and layers, respectively.

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# 3.1) How does hyperparameter tuning in machine learning relate to the concept of model complexity?
#
# 3.2) From the plot, what is the impact of increasing the number of hidden layers on the model's ability to capture complex patterns in the data?

# ## 4. Model assessment

# ### Accuracy

# Now, we are going to test the model's accuracy on the test dataset. First, we need to scale the test data using the scaler that we used for the training data. Then, we can use the `score` method of the model to compute the R2 score on the test data.

normalized_X_test = scaler_diameters.transform(X_test)
normalized_t_test = scaler_pressures.transform(t_test)

model.score(normalized_X_test, normalized_t_test)

# More than a performance metric as the score, we can observe the errors of different nodes for all the test database.

estimated_pressure = scaler_pressures.inverse_transform(model.predict(normalized_X_test))
error = t_test - estimated_pressure

node_ID = 0
error_node = error[:, node_ID]

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 4.1:</b>   
#     Plot the distribution of errors. Add this plot to your report.
# </div>

# +
fig, ax = plt.subplots()

# Create a histogram
ax.hist(error_node, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)

# Axis formatting
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
# -

# This plot gives us an idea of the expected errors we can encounter when using the model. However, it is also important that the MLP is faster than original simulator (which takes around 0.04 seconds per simulation).

# ### Speed

# We can calculate the time per scenario that the model takes.

# +
start_time = time.time()
y_pred_test = model.predict(X_test)
total_time = time.time() - start_time

num_test_sims = len(y_pred_test)

data_driven_exec_time_per_sim = total_time/num_test_sims
print(f'Data-driven model took {data_driven_exec_time_per_sim:.7f} seconds for {num_test_sims} scenarios')
# -

# Considering that the original model can take up to 0.04 seconds per scenario, we can estimate the potential gain in speed-up. (Speed-up = original_time/Data-driven_model_time)
#

# +
original_time_per_sim = 0.04

speed_up = np.round(original_time_per_sim/data_driven_exec_time_per_sim, 2)
print('The data-driven model is', speed_up,'times faster than original simulator per scenario.')
# -

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# 4.1) The score indicates a high fitting, is that reflected in the plot of the errors? Why?
#
# 4.2) Is the the plot of errors centered around zero? If not, what does that mean?
#
# 4.3) How diverse can the speed up values be if you run the cell multiple times? Why?
#
# 4.4) What would occur with the speed up if you increase the number of neurons in the hidden layers?

# ## 5. Model usage

# Now that we have a trained model, we can use it to predict the nodal pressures for a given set of diameters. Let's try it out!

# The water utility in charge of the BakRyan water distribution network is completing the installation of a system with the following diameters:
#
#     [ 800, 1000, 1100,  600,  450,  900,  700,  700,  300, 1100,  400, 700,  350, 1100,  600,  450,  400,  350, 1100,  900,  600,  600, 1100,  700,  500,  400,  450,  350,  700,  350, 1000,  400,  400, 400,  350,  900,  300, 1000,  400,  300,  450,  400,  450,  350,1100,  900,  450,  800,  800,  300, 1100,  600,  300,  700, 1000, 1000,  800,  800]

# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Task 5.1:</b>   
#     Using your model, predict the nodal pressures for this set of diameters, and report the lowest, highest and mean pressures in the system. Note that you will need to define the diameters, normalize them then use the model to make a prediction.
# </div>

YOUR_CODE_HERE 

# <div style="background-color:#FFC5CB; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <b>Questions:</b>
#
# 5.1) What is the minimum that your model predicts for this network?
#
# 5.2) How confident are you in the prediction of your model? Why?

# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png"/>
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png"/>
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
