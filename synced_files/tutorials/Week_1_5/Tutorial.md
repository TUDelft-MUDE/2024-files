<userStyle>Normal</userStyle>

# Week 1.5: Programming Tutorial

<h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
    <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
</h1>
<h2 style="height: 10px">
</h2>

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.5. September 30, 2024.*

_This notebook was prepared by Berend Bouvy and used in an in-class demonstration on Monday._


## Objects

```python
import numpy as np
import matplotlib.pyplot as plt
import os
```

```python
# Objective: Create a variable of each type in Python
int_var = #TODO: create an integer variable
float_var = #TODO: create a float variable
bool_var = #TODO: create a boolean variable
str_var = #TODO: create a string variable
list_var = #TODO: create a list variable
tuple_var = #TODO: create a tuple variable
dict_var = #TODO: create a dictionary variable


# Asserts
assert type(int_var) == int, f'Expected int but got {type(int_var)}'            # This will throw an error if the type is incorrect
assert type(float_var) == float, f'Expected float but got {type(float_var)}'    # This will throw an error if the type is incorrect
assert type(bool_var) == bool, f'Expected bool but got {type(bool_var)}'        # This will throw an error if the type is incorrect
assert type(str_var) == str, f'Expected str but got {type(str_var)}'            # This will throw an error if the type is incorrect
assert type(list_var) == list, f'Expected list but got {type(list_var)}'        # This will throw an error if the type is incorrect
assert type(tuple_var) == tuple, f'Expected tuple but got {type(tuple_var)}'    # This will throw an error if the type is incorrect
assert type(dict_var) == dict, f'Expected dict but got {type(dict_var)}'        # This will throw an error if the type is incorrect



```

```python
# relative path vs absolute path
# relative path: relative to the current working directory
# absolute path: full path from the root directory

# Create a variable that contains the current working directory using the os module
cwd = #TODO: get the current working directory
print(cwd)

# Get all the files in the current working directory
files = #TODO: get all the files in the current working directory
print(files)

# find path to data in data folder
data_dir = #TODO: find the path to the data folder
print(data_dir)

# read the data using absolute path and relative path
data_abs = #TODO: read the data using absolute path
data_rel = #TODO: read the data using relative path

# Asserts
assert data_abs == data_rel, 'Data read using absolute path and relative path are not the same' # This will throw an error if the data is not the same

```

$$ H_n = \sum_{k=1}^{n} \frac{1}{k} $$

```python
def harmonic_series(n):
    """
    This function calculates the harmonic series of n
    """
    result = #TODO: calculate the harmonic series of n
    return result

# Plotting
n = 100
x = #TODO: create a list of n values from 1 to n
y = #TODO: calculate the harmonic series of x, using list comprehension

#TODO: plot x and y, with labels and title

# asserts
assert harmonic_series(1) == 1, f'Expected 1 but got {harmonic_series(1)}'
assert harmonic_series(2) == 1.5, f'Expected 1.5 but got {harmonic_series(2)}'
assert harmonic_series(3) == 1.8333333333333333, f'Expected 1.8333333333333333 but got {harmonic_series(3)}'

# save x, y data
data = #TODO: create a 2D array with x and y using np.column_stack()
#TODO: save data to a csv file
```

<!-- #region -->
[Wikipedia: Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_sequence)


$$ F_n = F_{n-1} + F_{n-2} \quad \text{for} \quad n \geq 2 \quad \text{with} \quad F_0 = 0, \quad F_1 = 1$$

<!-- #endregion -->

```python
def fibonacci(n):
    """
    This function will return the n-th numbers of the fibonacci sequence
    """
    #TODO: calculate the n-th number of the fibonacci sequence
    return result

# Asserts
assert fibonacci(0) == 0, f'Expected 0 but got {fibonacci(0)}'    # This will throw an error if the result is incorrect 
assert fibonacci(1) == 1, f'Expected 1 but got {fibonacci(1)}'    # This will throw an error if the result is incorrect
assert fibonacci(2) == 1, f'Expected 1 but got {fibonacci(2)}'    # This will throw an error if the result is incorrect
assert fibonacci(3) == 2, f'Expected 2 but got {fibonacci(3)}'    # This will throw an error if the result is incorrect
```

```python
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

# Asserts
assert fib_sequence(0) == [], f'Expected [] but got {fib_sequence(0)}'                              # This will throw an error if the result is incorrect
assert fib_sequence(1) == [0], f'Expected [0] but got {fib_sequence(1)}'                            # This will throw an error if the result is incorrect
assert fib_sequence(2) == [0, 1], f'Expected [0, 1] but got {fib_sequence(2)}'                      # This will throw an error if the result is incorrect
assert fib_sequence(3) == [0, 1, 1], f'Expected [0, 1, 1] but got {fib_sequence(3)}'                # This will throw an error if the result is incorrect
assert fib_sequence(4) == [0, 1, 1, 2], f'Expected [0, 1, 1, 2] but got {fib_sequence(4)}'          # This will throw an error if the result is incorrect
assert fib_sequence(5) == [0, 1, 1, 2, 3], f'Expected [0, 1, 1, 2, 3] but got {fib_sequence(5)}'    # This will throw an error if the result is incorrect

```

**End of notebook.**
<h2 style="height: 60px">
</h2>
<h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
    <style>
        .markdown {width:100%; position: relative}
        article { position: relative }
    </style>
    <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
      <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
    </a>
    <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
      <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
    </a>
    <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
      <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
    </a>
    
</h3>
<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
