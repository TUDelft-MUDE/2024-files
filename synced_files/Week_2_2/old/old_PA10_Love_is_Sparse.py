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
# # Programming Assignment 10: Love Is Sparse
#
# <h1 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; top: 60px;right: 30px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" style="width:100px" />
#     <img src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" style="width:100px" />
# </h1>
# <h2 style="height: 10px">
# </h2>
#
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.2. Due: complete this PA prior to class on Friday, Nov 24, 2023.*

# %% [markdown]
# ## Overview of Assignment
#
# This assignment will introduce you to the concept of sparse matrices in Python and how they can be useful to speed up computations and reduce file sizes. To this end, we will be using the `scipy.sparse` library.
#
# ## Reading
#
# Keep the `scipy.sparse` [documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html) handy. Some of the work you'll do is based off this [blog](https://www.sefidian.com/2021/04/28/python-scipy-sparse-matrices-explained/), so you may find it helpful. In addition, if you don't know what a byte is, you may want to read up on [Wikipdia here](https://en.wikipedia.org/wiki/Byte) (not all of it, as long as you recognize that it is a measure of storage space on a computer).The concepts you learn here are applied to the Finite Element Method in this [book chapter](https://mude.citg.tudelft.nl/book/fem/matrix.html), which you are expected to read during Week 2.2.
#
# **Note:** you probably skipped over all the links in the paragraph above. While we forgive you for your haste, just remember to revisit some of them if you are struggling to finish the questions below!
#
# ## Assignment Criteria
#
# **You will pass this assignment as long as your respository fulfills the following criteria:**  
#
# - You have completed this notebook and it runs without errors

# %%
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import timeit


# %% [markdown]
# ## Task 1: Why sparse?
#
# Some matrices have a lot of zeros, with such an example given below. When this is the case, the way we store the actual information of the matrix (the non-zero elements) can have a big impact on computation speed and storage demands. Formats which handle this by only storing non-zero elements are called sparse, and have very different internal representations of the data to the matrices you have been familiarized with in previous programming assignments.
#
# ![Sparse matrix](images/sparse_matrix.png)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1:</b>   
#     
# - Create a function (`create_dense`) which returns a square matrix of arbitrary size. 
# - The function will take as input the size N (such that the matrix is N x N) and one float between 0 and 1, which represents the approximate fraction of the elements of the matrix which are non-zero (it doesn't have to be super accurate).
#     
# For now it just return a regular Numpy matrix. To do this, you can use <a href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html">numpy.random.rand</a> to create a random set of values between 0 and 1 and then threshold the entries with a simple boolean operator.
# </p>
# </div>

# %%
def create_dense(size: int, percentage: float) -> np.array:
    matrix = YOUR_CODE_HERE
    matrix[YOUR_CODE_HERE] = 0
    return matrix


# %% [markdown]
# Now we will test that you set it up correctly:

# %%
# Quick Test
test_size = YOUR_CODE_HERE
test_percentage = YOUR_CODE_HERE
matrix = create_dense(test_size, test_percentage)
assert np.count_nonzero(matrix) < test_percentage*1.1*test_size**2

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2:</b>   
# Use <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html">array.nbytes</a> to find out how much space a 1000x1000 matrix with 10% non-zero elements takes. Try to explain where this number came from! (Hint: the answer is in the assert statement)
# </p>
# </div>

# %%
my_matrix_size = YOUR_CODE_HERE
assert my_matrix_size == 8*test_size**2

# %% [markdown]
# Next we will explore how to use `scipy.sparse`, and how this reduces the data size of the matrix. The [ documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html) gives us many different types of formats to choose from, so we'll explore two of them: BSR (Block Sparse Row) and CSR (Compressed Sparse Row). 

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3:</b>   
#     Complete the code below to make a CSR and BSR matrix from the <code>matrix</code> variable.
# </p>
# </div>

# %%
csr_matrix = YOUR_CODE_HERE
bsr_matrix = YOUR_CODE_HERE

# %% [markdown]
# Let's compare the new storage requirements and see how much of an improvement we got (it should approach the value used above for `test_percentage`, but not reach it exactly):

# %%
print(f"CSR matrix size: {csr_matrix.data.size} bytes")
print(f"Compared to the normal matrix, CSR uses this fraction of space: {csr_matrix.data.nbytes/my_matrix_size:0.3f}")
print(f"BSR matrix size: {bsr_matrix.data.size} bytes")
print(f"Compared to the normal matrix, BSR uses this fraction of space: {bsr_matrix.data.nbytes/my_matrix_size:0.3f}")

# %% [markdown]
# ## Task 2: [What is love?](https://www.youtube.com/watch?v=HEXWRTEbj1I)
#
# Let's look into a small example of how sparse matrices can also help improve calculation speeds. We'll study the mysterious case of a massive friend group with a concerning love circle and how we can predict how each person feels.

# %% [markdown]
# We know there is a certain pecking order in this group, and neighbours in this order have a love-hate relationship which can be quantified with a simple differential equation:
#
# $$
# \begin{pmatrix}
# \frac{dn_1}{dt}\\
# \frac{dn_2}{dt} \\
# \end{pmatrix} 
# =
# \begin{pmatrix}
# 0 & 1\\
# -1 & 0 \\
# \end{pmatrix} 
# \begin{pmatrix}
# n_1\\
# n_2 \\
# \end{pmatrix} 
# $$

# %% [markdown]
# The state of any given person indicates how much they love the group in general. So in this case, person 2 doesn't like it when person 1 is happy. If we extend this to a four case scenario we'd get the following matrix:
# $$
# \begin{pmatrix}
# \frac{dn_1}{dt}\\
# \frac{dn_2}{dt}\\
# \frac{dn_3}{dt}\\
# \frac{dn_4}{dt}\\
# \end{pmatrix} 
# =
# \begin{pmatrix}
# 0  & 1  & 0 & -1 \\
# -1 & 0  & 1 & 0  \\
# 0  & -1 & 0 & 1  \\
# 1  & 0  & -1 & 0  \\
# \end{pmatrix} 
# \begin{pmatrix}
# n_1 \\
# n_2 \\
# n_3 \\
# n_4 \\
# \end{pmatrix} 
# $$
#
# What happens if we extend it to even more people?
#
# Coincidentally this is very similar to how we use individual elements in the Finite Element Method! We can easily operationalize it using the method `ix_`, for which a simple example is provided in the code cell below (this example is generic to illustrate `ix_` usage and is not related to the love circle!):

# %%
blank = np.zeros(shape=(4, 4))
blueprint = np.array([[0, 0.5], 
                      [1, 0.5]])

for i in range(2):
    # First argument will be used for rows
    # Second for columns
    blank[np.ix_([i*2, i*2 + 1], [1, 2])] = blueprint
    
print(blank)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 4:</b>   
#     Generate the matrix <code>relationship</code> for the differential equation for 1000 people. Use the <a href="https://numpy.org/doc/stable/reference/generated/numpy.ix_.html"><code>numpy.ix_</code></a> function to make your life easier. 
# </p>
# </div>

# %%
N = 1000
relationship = np.zeros(shape=(N, N))

YOUR_CODE_HERE

# %% [markdown]
# Finally, we are going to use the forward Euler method to simulate this differential equation for a total of 5 seconds over 1000 iterations. This has already been implemented in the `test` method.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 5:</b>   
#     Find the time it takes to evaluate the relationship using <code>timeit</code> by entering the function you wish to evaluate as a string. HINT: you have already learned how to convert a matrix into a sparse format, and the function is defined for you. Run the code cell and compare the performances of the different matrix formats. Which one is faster? How much space do they take?
# </p>
# </div>

# %%
N_ITS = 1000
T = 5 # Seconds
dt = T/N_ITS

def test(rel_matrix):
    state = np.zeros(N); state[0] = 1
    for i in range(N_ITS):
        state = state + rel_matrix @ state * dt

csr_matrix = YOUR_CODE_HERE
bsr_matrix = YOUR_CODE_HERE
print(f"Standard: {timeit.timeit('YOUR_CODE_HERE', globals=globals(), number=10)/10:.4f}")
print(f"CSR: {timeit.timeit('YOUR_CODE_HERE', globals=globals(), number=10)/10:.4f}")
print(f"BSR: {timeit.timeit('YOUR_CODE_HERE', globals=globals(), number=10)/10:.4f}")
    

# %% [markdown]
# One final consideration when using sparse matrices is that it can take a long time to generate them from a regular matrix. You can test this out by placing the matrix generation inside or outside the timeit code to compare their performances.

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
#     &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. 
#     This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
#   </div>
# </div>
#
#
# <!--tested with WS_2_8_solution.ipynb-->
