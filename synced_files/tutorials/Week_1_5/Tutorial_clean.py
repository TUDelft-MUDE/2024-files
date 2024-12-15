import numpy as np
import matplotlib.pyplot as plt
import os

int_var = #TODO: create an integer variable
float_var = #TODO: create a float variable
bool_var = #TODO: create a boolean variable
str_var = #TODO: create a string variable
list_var = #TODO: create a list variable
tuple_var = #TODO: create a tuple variable
dict_var = #TODO: create a dictionary variable

assert type(int_var) == int, f'Expected int but got {type(int_var)}'            # This will throw an error if the type is incorrect
assert type(float_var) == float, f'Expected float but got {type(float_var)}'    # This will throw an error if the type is incorrect
assert type(bool_var) == bool, f'Expected bool but got {type(bool_var)}'        # This will throw an error if the type is incorrect
assert type(str_var) == str, f'Expected str but got {type(str_var)}'            # This will throw an error if the type is incorrect
assert type(list_var) == list, f'Expected list but got {type(list_var)}'        # This will throw an error if the type is incorrect
assert type(tuple_var) == tuple, f'Expected tuple but got {type(tuple_var)}'    # This will throw an error if the type is incorrect
assert type(dict_var) == dict, f'Expected dict but got {type(dict_var)}'        # This will throw an error if the type is incorrect

cwd = #TODO: get the current working directory
print(cwd)

files = #TODO: get all the files in the current working directory
print(files)

data_dir = #TODO: find the path to the data folder
print(data_dir)

data_abs = #TODO: read the data using absolute path
data_rel = #TODO: read the data using relative path

assert data_abs == data_rel, 'Data read using absolute path and relative path are not the same' # This will throw an error if the data is not the same

def harmonic_series(n):
    """
    This function calculates the harmonic series of n
    """
    result = #TODO: calculate the harmonic series of n
    return result

n = 100
x = #TODO: create a list of n values from 1 to n
y = #TODO: calculate the harmonic series of x, using list comprehension

assert harmonic_series(1) == 1, f'Expected 1 but got {harmonic_series(1)}'
assert harmonic_series(2) == 1.5, f'Expected 1.5 but got {harmonic_series(2)}'
assert harmonic_series(3) == 1.8333333333333333, f'Expected 1.8333333333333333 but got {harmonic_series(3)}'

data = #TODO: create a 2D array with x and y using np.column_stack()

def fibonacci(n):
    """
    This function will return the n-th numbers of the fibonacci sequence
    """
    #TODO: calculate the n-th number of the fibonacci sequence
    return result

assert fibonacci(0) == 0, f'Expected 0 but got {fibonacci(0)}'    # This will throw an error if the result is incorrect 
assert fibonacci(1) == 1, f'Expected 1 but got {fibonacci(1)}'    # This will throw an error if the result is incorrect
assert fibonacci(2) == 1, f'Expected 1 but got {fibonacci(2)}'    # This will throw an error if the result is incorrect
assert fibonacci(3) == 2, f'Expected 2 but got {fibonacci(3)}'    # This will throw an error if the result is incorrect

def fib_sequence(n):
    """
    This function will return the first n numbers of the fibonacci sequence
    """
    # result = #TODO: calculate the first n numbers of the fibonacci sequence
    result = [0, 1]
    if n == 0:
        return []
    elif n == 1:
        return [0]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result

assert fib_sequence(0) == [], f'Expected [] but got {fib_sequence(0)}'                              # This will throw an error if the result is incorrect
assert fib_sequence(1) == [0], f'Expected [0] but got {fib_sequence(1)}'                            # This will throw an error if the result is incorrect
assert fib_sequence(2) == [0, 1], f'Expected [0, 1] but got {fib_sequence(2)}'                      # This will throw an error if the result is incorrect
assert fib_sequence(3) == [0, 1, 1], f'Expected [0, 1, 1] but got {fib_sequence(3)}'                # This will throw an error if the result is incorrect
assert fib_sequence(4) == [0, 1, 1, 2], f'Expected [0, 1, 1, 2] but got {fib_sequence(4)}'          # This will throw an error if the result is incorrect
assert fib_sequence(5) == [0, 1, 1, 2, 3], f'Expected [0, 1, 1, 2, 3] but got {fib_sequence(5)}'    # This will throw an error if the result is incorrect

