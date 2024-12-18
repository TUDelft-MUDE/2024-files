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
# # PA 1.8: Equations Done Symply
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.8. Due: complete this PA prior to class on Friday, Oct 25, 2024.*

# %% [markdown]
# This notebook should be completed _after_ you read the [book chapter on Sympy](https://mude.citg.tudelft.nl/2024/book/programming/week_1_8.html).
#
# The exercises in this notebook are directly from WS 1.7 and PA 1.7, to give you a very practical application for Sympy. See the [WS 1.7 solution here](https://mude.citg.tudelft.nl/2024/files/Week_1_7/WS_1_7_solution/) for reference.

# %%
import sympy as sym

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1:</b>   
#
# Fill in the missing pieces of code to get Sympy to reproduce the equations and analyses required from WS 1.7. The comments in the code provide explicit instructions.
#
# </p>
# </div>

# %%
# Define the Gumbel distribution using SymPy

x, beta, mu = sym.symbols('x beta mu')
#F_gumbel = YOUR CODE HERE

# solution
F_gumbel = sym.exp(-sym.exp(-(x-mu)/beta))
display(F_gumbel)

# %%
# Invert the Gumbel distribution using SymPy to find x ~ non-exceedence probability, beta, mu as done in the workshop by hand
Prob_non_exc = sym.symbols('Prob_non_exc')
#YOUR CODE HERE
#x_sol = YOUR CODE HERE

# solution
Prob_non_exc = sym.symbols('Prob_non_exc')
eq1 = sym.Eq(Prob_non_exc,F_gumbel)
display(eq1)
x_sol = sym.solve(eq1, x)[0]

display(x_sol)

# %%
# Evaluate your inverted Gumbel distribution using SymPy for the min, 0.25, 0.5, 0.75 and max non-exceedence probabilities as done in the workshop by hand
# You should find:
#3.353
#23.89
#32.97
#44.48
#115.3

Prob_non_exc_list = [1/773, 0.25, 0.5, 0.75, 772/773]
for i in range(len(Prob_non_exc_list)):
    display(x_sol.subs({beta:13.097, mu:28.167, Prob_non_exc:Prob_non_exc_list[i]}))

# %%
# Use SymPy to find the probability density function of the Gumbel distribution, is it equal to the function you found in the book?
#f_gumbel = YOUR CODE HERE

# solution
f_gumbel = sym.diff(F_gumbel, x)
display(f_gumbel)

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2:</b>   
#
# Fill in the missing pieces of code to get Sympy to reproduce the equations and analyses required from PA 1.7.
#
# </p>
# </div>

# %% [markdown]
# Define the Piecewise equation from PA 1.7 in SymPy (hint, see https://docs.sympy.org/latest/modules/functions/elementary.html#piecewise)
#
# $$\begin{equation}
# f(x)=
#     \begin{cases}
#         0.1 & \text{if } 0 < x < 3.6 \\
#         2(x-5) &  5 < x < 5.8 \\
#         0 & \text{elsewhere}
#     \end{cases}
# \end{equation}$$
#
# Use the provided function to plot it for $-1<x<6$

# %%
#f = YOUR CODE HERE
#sym.plot(f,(x,-1,6));

#solution
f = sym.Piecewise((0, x< 0), (0.1 , x < 3.6), (0, x < 5), (2*(x-5),x<5.8), (0, True))
sym.plot(f, (x,-1,6));

# %%
# Integrate the piecewise probability density function to find the cumulative distribution function as done numerically in PA1.7
x = sym.symbols('x')
#F = YOUR CODE HERE
#sym.plot(F, (x,-1,6));

# solution
F = sym.integrate(f, (x,-sym.oo,x))
#or
#F = sym.integrate(f,x)
display(F)
display(F.rewrite(sym.Piecewise).simplify())
sym.plot(F, (x,-1,6));

# %% [markdown]
# This cell will check your work to make sure you have completed the assignment properly.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3:</b>   
#
# Run the cell below to check your work, but don't change anything. If the cell runs without error, you will pass the assignment once you commit it and push it to GitHub.
#
# </p>
# </div>

# %%
import numpy as np
assert sym.simplify(F_gumbel - sym.exp(-sym.exp((mu - x)/beta))) == 0 , 'Error: Gumbel distribution is not correct'
assert sym.simplify(x_sol + beta*sym.log(sym.exp(-mu/beta)*sym.log(1/Prob_non_exc))) == 0, 'Error: inverted Gumbel distribution is not correct'
assert sym.simplify(f_gumbel - sym.exp(-((x-mu)/beta+sym.exp(-(x-mu)/beta)))/beta) == 0, 'Error: probability densily function is not correct'
assert np.allclose(np.array([0.        , 0.        , 0.05555556, 0.13333333, 0.21111111,       0.28888889, 0.36      , 0.36      , 0.40938272, 1.        ]), sym.lambdify(x, F.rewrite(sym.Piecewise).simplify())(np.linspace(-1,6,10))), 'Error: Piecewise cumulative distribution function is not correct'

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
