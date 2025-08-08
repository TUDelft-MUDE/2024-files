# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
# ---

# %% [markdown]
# # PA 1.6: Boxes and Bugs
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px; height: auto; margin: 0" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px; height: auto; margin: 0" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6. Due: before Friday, Oct 4, 2024.*

# %% [markdown]
# The purpose of this notebook is to introduce a few useful Python topics:
#
# 1. Visualize a matrix with `plt.matshow` (Matplotlib)
# 2. Filling in the contents of a matrix ($m$ x $n$ Numpy array) with specific patterns and values
# 3. Illustrate the difference between `range` and `np.arange`
#
#
# ## Context
#
# For many scientific computing applications, in particular the field of numerical analysis, we formulate and solve our problems using matrices. The matrix itself is an arbitrary collection of values, however, the formulation and discretization of the problem will dictate a specific structure and meaning to the arrangement of the values inside a given matrix. When solving ordinary and partial differential equations with numerical schemes, we discretize space and time into discrete points or intervals, and the values of interest are specified by the elements of a matrix or vector. For example, a vector of the quantity $y$ can be discretized as `y = [y0, y1, y2, ... , yn]`, where each element `yi` refers to the $n$ spatial coordinate of $y_i$. For 2D problems, or perhaps problem with a temporal component (a dependence on time), we need to encode this information in matrices. Thus, when implementing numerical schemes, it is important to be able to fill in the values of a matrix in an efficient and reliable way.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Topic 1: Visualizing a Matrix
#
# At this point you should be familiar with the Numpy library and its key data type the `ndarray`. In other words, it should be very obvious why executing something like this:
#
# ```python
# import numpy as np
# x = np.array([1, 4, 7, 9])
# ```
#
# returns something like this:
# ```python
# numpy.ndarray
# ```
#
# We have already also used Numpy to create 2D arrays to represent matrices. Often one of the challenges of working with matrices is visualizing their contents, especially when the matrices become very large. Fortunately there is a Matplotlib method that makes visualizing matrices very easy: `matshow`. When using the conventional import statement `import matplotlib.pyplot as plt`, we can use this method as `plt.matshow`.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1:</b>   
# Use the Python <code>help</code> function to view the docstring (documentation) of the matrix visualization method.
# </p>
# </div>

# %%
# help(plt.matshow)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.2:</b>   
# Run the cell below to visualize the A matrix. Change the values and rerun the cell to see the effect, especially noting that each "square" corresponds to an element in the matrix. Simple, right?!
# </p>
# </div>

# %%
A = [[1, 2, 1, 1],
     [2, 3, 2, 2],
     [1, 2, 1, 1],
     [1, 2, 1, 1]]

plt.matshow(A)
plt.show()

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3:</b>   
# Run the cell below to see how a 100x100 matrix filled with random values looks.
# </p>
# </div>

# %%
A = np.random.rand(100, 100)
plt.matshow(A)
plt.show()

# %% [markdown]
# That's pretty much all there is to it. Note that the axes indicate the row and column indices.

# %% [markdown]
# ## Topic 2: Filling a Matrix
#
# Now that we can visualize the contents of a matrix, lets find an efficient way to fill it with specific values, focusing on creating specific patterns in an efficient way with our Python code. First, let's recall a few more important things about Numpy arrays, focusing on the particular case of making a 2-dimensional arrays to represent 2-dimensional matrices.
#
# One of the first things to remember is that Numpy uses a parameter `shape` to define the dimension and length of each axis of an array. For the 2D case, this means an $m$-by-$n$ matrix is specified with a tuple containing two elements: `(m, n)`.
#
# Second, Numpy has _many_ methods that make it easy to create a matrix and fill it with specific values. Check out a cool list here: [Numpy array creation routines](https://numpy.org/doc/2.0/reference/routines.array-creation.html#). Some commonly used methods are:
# - `np.zeros`
# - `np.ones`
# - `np.full`
#
# Third there are many Numpy methods that can _modify_ an existing matrix (see the same list linked above), for example: `np.fill_diagonal`. 
#
# Finally, remember that arrays are quite smart when it comes to indexing. For example, we can use the `range` method (part of the standard Python library) to things to specific indices in an array.
#
# With these tips in mind, let's go over a few warm-up exercises to see how to easily manipulate matrices.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1:</b>   
#
# Refresh your memory on the <code>range</code> function by printing the documentation. Then comment the help line and confirm that you can use the function by using a list comprehension to print: a) values from 1 to 5, then b) values 2, 4, 6, 8, 10. 
#
# </p>
# </div>

# %%
# # help(range)
# 
# print('Part a:')
# [print(i) for i in YOUR_CODE_HERE]
# print('Part b:')
# [print(i) for i in YOUR_CODE_HERE]

# SOLUTION
print('Part 1:')
[print(i) for i in range(6)]

print('Part 2:')
[print(i) for i in range(2, 11, 2)]

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2:</b>   
#
# Use a Numpy method to create a 3x3 matrix filled with value 1.
#
# </p>
# </div>

# %%
# A = YOUR_CODE_HERE
# 
# assert np.all(A==1)
# assert A.shape==(3, 3)

# SOLUTION
A = np.ones((3,3))

assert np.all(A==1)
assert A.shape==(3, 3)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3:</b>   
#
# Use a Numpy method to create a 3x3 matrix filled with value 3 on the diagonal and 0 elsewhere.
#
# </p>
# </div>

# %%
# A = YOUR_CODE_HERE
# np.YOUR_CODE_HERE
# 
# assert np.all(A.diagonal()==3)
# assert A.sum()==9
# assert A.shape==(3, 3)

# SOLUTION
A = np.zeros((3,3))
np.fill_diagonal(A, 3)

assert np.all(A.diagonal()==3)
assert A.sum()==9
assert A.shape==(3, 3)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.4:</b>   
#
# Use a Numpy method to create a 10x10 matrix, then assign every other element in the <em>diagonal</em> of the matrix to the value 1 using <code>range</code> and indexing. Use <code>plt.matshow()</code> to confirm that the matrix plot looks like a checkerboard.
#
# </p>
# </div>

# %%
# A = YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(10, 10)
# assert A.sum()==5
# assert np.sum(A==1)==5
# assert np.sum(A==0)==95

# SOLUTION
A = np.zeros((10,10))
A[range(1, 10, 2), range(1, 10, 2)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(10, 10)
assert A.sum()==5
assert np.sum(A==1)==5
assert np.sum(A==0)==95

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.5:</b>   
#
# Use a Numpy method to create a 5x5 matrix, fill the diagonal with value 5, then use <code>range</code> and indexing to assign the diagonal above and below the center diagonal to the value 1. The solution is illustrated in the imported Markdown figure below.
#
# </p>
# </div>

# %% [markdown]
# ![](./images\matrix01.svg)

# %%
# A = YOUR_CODE_HERE
# YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE
# A[YOUR_CODE_HERE, YOUR_CODE_HERE] = YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(5, 5)
# assert A.sum()==(5*5 + 2*4)

# SOLUTION
A = np.zeros((5, 5))
np.fill_diagonal(A, 5)
A[range(4), range(1, 5)] = 1
A[range(1, 5), range(4)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(5, 5)
assert A.sum()==(5*5 + 2*4)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.6:</b>   
# Create the matrix illustrated in the figure below, where the values are either 0 or 1.
# </p>
# </div>

# %% [markdown]
# ![](./images\matrix02.svg)

# %%
# A = YOUR_CODE_HERE
# for i in YOUR_CODE_HERE:
#     YOUR_CODE_HERE

# plt.matshow(A)
# plt.show()

# assert A.shape==(10, 10)
# assert A.sum()==25

# SOLUTION
A = np.zeros((10, 10))
for i in range(0, 10, 2):
    A[i, range(0, 10, 2)] = 1

plt.matshow(A)
plt.show()

assert A.shape==(10, 10)
assert A.sum()==25

# %% [markdown]
# ## Topic 3: a `range` and `arange`
#
# The previous part used `range` to fill in the items of a matrix. However, you may also be familiar with a method from the Numpy library called `arange`. On the one hand, both methods do similar things, which can roughly be described as follows:
#
# - if one input, a, is given, count integers from 0 to a
# - if two inputs, a and b, are given, count integers from a to b
# - if three inputs, a, b and c, are given, count from a to b by (integer!) increment c
# - in all cases, exclude b
#
# Despite these similarities they return different object types, which often leads to confusion or errors if used without explicitly accounting for this difference. Let's take a closer look to find out more.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.1:</b>   
#
# Print the documentation for <code>np.arange</code> and compare it to <code>range</code> until you can identify the differences.
#
# </p>
# </div>

# %%
help(np.arange)

# %% [markdown]
# In particular, note the following sentences in the docstring for `np.arange`:
#
# ```
# For integer arguments the function is roughly equivalent to the Python
# built-in :py:class:`range`, but returns an ndarray rather than a ``range``
# instance.
#
# When using a non-integer step, such as 0.1, it is often better to use
# `numpy.linspace`.
# ```
#
# The main difference is that `np.arange` **returns an array!**

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.2:</b>   
#
# Confirm that you understand the usage of <code>np.arange</code> by creating the same two sets of integer values as in Task 2.1 (integers 0 through 5 and 2 through 10 by 2's), except this time you will produce Numpy arrays in addition the printing the indices.
#
# </p>
# </div>

# %%
# x = YOUR_CODE_HERE
# print('Part 1:')
# [print(i) for i in YOUR_CODE_HERE]
# 
# assert type(x) == np.ndarray
# assert np.all(x == [0, 1, 2, 3, 4, 5])
# 
# x = YOUR_CODE_HERE
# print('Part 2:')
# [print(i) for i in YOUR_CODE_HERE]
# 
# assert type(x) == np.ndarray
# assert np.all(x == [2, 4, 6, 8, 10])

# SOLUTION
x = np.arange(6)
print('Part 1:')
[print(i) for i in x]

assert type(x) == np.ndarray
assert np.all(x == [0, 1, 2, 3, 4, 5])

x = np.arange(2, 11, 2)
print('Part 2:')
[print(i) for i in x]

assert type(x) == np.ndarray
assert np.all(x == [2, 4, 6, 8, 10])

# %% [markdown]
# This Part is not meant to be complicated; rather, it is meant to explicitly indicate the difference between `range` and `np.arange` to help you debug your code more easily. The **main takeaway** is that you should use `range` when you are iterating through indices and don't need to use the indices as values, whereas `np.arange` is necessary when the indices are needed as values. It is also good to recognize that `range` is part of the standard Python library, whereas `np.arange` is not (it is part of Numpy). This is because `range` returns a `range` object, whereas `np.arange` returns a Numpy array.

# %% [markdown]
# **End of notebook.**
#
# <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ccc;">
#   <div style="display: flex; justify-content: flex-end; gap: 20px; align-items: center;">
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="width:100px; height:auto;" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="width:88px; height:auto;" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#   </div>
#   <div style="font-size: 75%; margin-top: 10px; text-align: right;">
#     By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a>
#     &copy; 2024 TU Delft. 
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
#     <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515"><img style="width:auto; height:15; vertical-align:middle" src="https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg" alt="DOI https://doi.org/10.5281/zenodo.16782515"></a>
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
