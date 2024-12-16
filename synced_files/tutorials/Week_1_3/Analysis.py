# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
# ---

# %% [markdown]
# # Week 1.3: Programming Tutorial
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3. September 16, 2024.*
#
# _This notebook was prepared by Berend Bouvy and used in an in-class demonstration on Monday (the first of several programming tutorials)._

# %%
def add(a, b):
    result = a+b
    return result

def gen_xhat(A, y):
    x_hat = np.linalg.inv(A.T @ A) @ A.T @ y
    return x_hat


# %%
a = 1
b = 2
result = add(a, b)

result

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Replace 'file.csv' with the path to your CSV file
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)  # 'skip_header=1' skips the first row (header)



# %%
t = data[:,0]
y = data[:,1]
n_rows = data.shape[0]
n_cols = data.shape[1]

# %%
plt.plot(t, y,'o')
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

# %%
one_vector = np.ones(n_rows)

# %%
print(one_vector+ t)

# %%
A = np.column_stack((one_vector, t))

# %%
x_hat = gen_xhat(A, y)
y_hat = A @ x_hat

# %%
plt.plot(t, y,'o')
plt.plot(t, y_hat)
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

# %%
A_new = np.column_stack((one_vector, t, t**2))
x_hat_new = gen_xhat(A_new, y)
y_hat_new = A_new @ x_hat_new
plt.plot(t, y,'o')
plt.plot(t, y_hat_new)
plt.title('t vs y')
plt.xlabel('t')
plt.ylabel('y')

# %%
e_hat = y-y_hat_new

# %% [markdown]
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png" />
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png" />
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
