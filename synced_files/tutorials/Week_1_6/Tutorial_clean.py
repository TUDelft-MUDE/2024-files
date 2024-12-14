# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: TAMude
#     language: python
#     name: python3
# ---

# # Week 1.5: Programming Tutorial
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6. October 7, 2024.*
#
# _This notebook was prepared by Berend Bouvy and used in an in-class demonstration on Monday._

# ## Objects

import numpy as np
import matplotlib.pyplot as plt
from func import *
import timeit

# import data 
data = np.loadtxt('data.txt')
x = data[:,0]
y = data[:,1]


A = #TODO: create the matrix A
x_hat, y_hat = #TODO: solve the system of equations
print(f"x_hat = {x_hat}")

# +
# runtimes

funcs = [FD_1, FD_2, FD_3, FD_4]

assert np.allclose(funcs[0](x, y), funcs[1](x,y)), "FD_1 and FD_2 are not equal"
assert np.allclose(funcs[0](x, y), funcs[2](x,y)), "FD_1 and FD_3 are not equal"
assert np.allclose(funcs[0](x, y), funcs[3](x,y)), "FD_1 and FD_4 are not equal"

runtime = np.zeros(4)
for i in range(4):
    runtime[i] = timeit.timeit(lambda: funcs[i](x, y), number=1000)

print(f"runtimes: {runtime}")
# -

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
