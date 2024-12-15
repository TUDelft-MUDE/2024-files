# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
data = YOUR_CODE_HERE

# %% [markdown]

# %%
YOUR_CODE_HERE

# %% [markdown]

# %%
X = YOUR_CODE_HERE
Y = YOUR_CODE_HERE

# %% [markdown]

# %% [markdown]

# %%
rng = np.random.default_rng()
print(type(rng))
rng.random()

# %% [markdown]

# %% [markdown]

# %%
print('integers:', rng.integers(5))
print('random:', rng.random(5))
print('choice:', rng.choice(np.array(5)))
print('bytes:', rng.bytes(5))

# %% [markdown]

# %% [markdown]

# %%
rng = np.random.default_rng(seed=14)
print('integers:', rng.integers(5))
print('random:', rng.random(5))
print('choice:', rng.choice(np.array(5)))
print('bytes:', rng.bytes(5))

# %% [markdown]

# %% [markdown]

# %%

test_array_length = 5
test_array = rng.integers(low=100, high=200, size=test_array_length)

random_indices = YOUR_CODE_HERE

print('The randomized indices are:', random_indices)
print('The randomized array becomes:', test_array[random_indices])

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %% [markdown]

# %%
def split_data(X, Y, proportions):
    """Split input and output into 3 subsets for ML model.

    Arguments
    =========
    X, Y:        ndarrays where rows are number of observations
                    (both arrays have identical number of rows)
    proportions: list with decimal fraction of original data defining
                 allocation into three parts (train, validate, test sets,
                 respectively). The list is len(proportions)=3, and
                 contains floats that should sum to 1.0.

    Returns
    =======
    X_train, X_val, X_test, Y_train, Y_val, Y_test:
     6 ndarrays (3 splits each for input and output), where the number of
     columns corresponds to the original input and output (respectively)
     and the sum of the number of rows is equal to the rows of the original
     input/output.
    """
    assert YOUR_CODE_HERE, "Contract broken: 3 proportions must be provided"
    assert YOUR_CODE_HERE, "Contract broken: sum of proportions should be one"
    assert YOUR_CODE_HERE, "Contract broken: X and Y arrays must have same dimensions"

    
    np.random.default_rng(seed=42)

    
    indices = YOUR_CODE_HERE

    
    YOUR_CODE_HERE 

    assert YOUR_CODE_HERE, "Contract broken: generated datasets don't have same accumulated length as original"
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# %% [markdown]

# %%
split_proportions = YOUR_CODE_HERE
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = split_data(YOUR_CODE_HERE)

# %% [markdown]

# %%
def plot_allocation(X, Y,
                    X_train, X_val, X_test,
                    Y_train, Y_val, Y_test):

    set_of_X_and_Y = np.hstack((X,Y.reshape((100,1))))
    
    which_set_am_i = np.zeros((len(Y), 75))
    
    for i in range(len(X_train)):
        matching_rows = np.all(X==X_train[i], axis=1)
        which_set_am_i[np.where(matching_rows)[0],:] = 1
    for i in range(len(X_val)):
        matching_rows = np.all(X==X_val[i], axis=1)
        which_set_am_i[np.where(matching_rows)[0],:] = 2

    for i in range(len(X_test)):
        matching_rows = np.all(X==X_test[i], axis=1)
        which_set_am_i[np.where(matching_rows)[0],:] = 3
        
    fig, ax = plt.subplots()
    ax.imshow(which_set_am_i)

    ax.set_title('Colors indicate how data is split')
    ax.set_xlabel('Width is arbitrary')
    ax.set_ylabel('Row of original data set')
    
    print('The number of data in each set is:')
    print(f'       training: {sum(which_set_am_i[:,0]==1)}')
    print(f'     validation: {sum(which_set_am_i[:,0]==2)}')
    print(f'        testing: {sum(which_set_am_i[:,0]==3)}')
    print(f'  none of above: {sum(which_set_am_i[:,0]==0)}')

plot_allocation(X, Y,
                X_train, X_val, X_test,
                Y_train, Y_val, Y_test)

# %% [markdown]

