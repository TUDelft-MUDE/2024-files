# ---

# ---

# %% [markdown]

# %% [markdown]

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %%

int_var = 
float_var = 
bool_var = 
str_var = 
list_var = 
tuple_var = 
dict_var = 

assert type(int_var) == int, f'Expected int but got {type(int_var)}'            
assert type(float_var) == float, f'Expected float but got {type(float_var)}'    
assert type(bool_var) == bool, f'Expected bool but got {type(bool_var)}'        
assert type(str_var) == str, f'Expected str but got {type(str_var)}'            
assert type(list_var) == list, f'Expected list but got {type(list_var)}'        
assert type(tuple_var) == tuple, f'Expected tuple but got {type(tuple_var)}'    
assert type(dict_var) == dict, f'Expected dict but got {type(dict_var)}'        

# %%

cwd = 
print(cwd)

files = 
print(files)

data_dir = 
print(data_dir)

data_abs = 
data_rel = 

assert data_abs == data_rel, 'Data read using absolute path and relative path are not the same' 

# %% [markdown]

# %%
def harmonic_series(n):
    """
    This function calculates the harmonic series of n
    """
    result = 
    return result

n = 100
x = 
y = 

assert harmonic_series(1) == 1, f'Expected 1 but got {harmonic_series(1)}'
assert harmonic_series(2) == 1.5, f'Expected 1.5 but got {harmonic_series(2)}'
assert harmonic_series(3) == 1.8333333333333333, f'Expected 1.8333333333333333 but got {harmonic_series(3)}'

data = 

# %% [markdown]

# %%
def fibonacci(n):
    """
    This function will return the n-th numbers of the fibonacci sequence
    """
    
    return result

assert fibonacci(0) == 0, f'Expected 0 but got {fibonacci(0)}'    
assert fibonacci(1) == 1, f'Expected 1 but got {fibonacci(1)}'    
assert fibonacci(2) == 1, f'Expected 1 but got {fibonacci(2)}'    
assert fibonacci(3) == 2, f'Expected 2 but got {fibonacci(3)}'    

# %%
def fib_sequence(n):
    """
    This function will return the first n numbers of the fibonacci sequence
    """
    
    result = [0, 1]
    if n == 0:
        return []
    elif n == 1:
        return [0]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result

assert fib_sequence(0) == [], f'Expected [] but got {fib_sequence(0)}'                              
assert fib_sequence(1) == [0], f'Expected [0] but got {fib_sequence(1)}'                            
assert fib_sequence(2) == [0, 1], f'Expected [0, 1] but got {fib_sequence(2)}'                      
assert fib_sequence(3) == [0, 1, 1], f'Expected [0, 1, 1] but got {fib_sequence(3)}'                
assert fib_sequence(4) == [0, 1, 1, 2], f'Expected [0, 1, 1, 2] but got {fib_sequence(4)}'          
assert fib_sequence(5) == [0, 1, 1, 2, 3], f'Expected [0, 1, 1, 2, 3] but got {fib_sequence(5)}'    

# %% [markdown]

