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
# # PA 1.1: Catch Them All
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.1, 2024.*

# %% [markdown]
# In this assignment we will check out a fundamental Python data type: dictionaries. We will also use a `*.py` file to import data to help understand what a dictionary is.
#
# While dictionaries are fun and useful, the *real* purpose of this assignment is to confirm that you are comfortable using Jupyter notebooks, as we will use them a lot in MUDE. If you have used these `*.ipynb` files before, this will be easy although the VSC interface is slightly different than Jupyter software).
#
# If you have never used a Jupyter notebook before, don't panic: your teachers are available to help (don't forget about question hours on Tue and Thu!). Hopefully one of your group members can also help you get acquainted with these files and how to use them.
#
# If you find notebooks to be "simple" we ask that you _please help your fellow group members if they are unfamiliar with this tool!_ This will help you become a more collaborative group later on in the quarter, so help each other out now while the assignments are easier.
#
# ## Part 0: Making sure you can run Python in the notebook
#
# In a simple sense, Jupyter Notebooks are documents that consist of many _cells._ Some _cells_ are formatted text (Markdown, like this one), others can be used to execute Python code. Because we have already set up a _conda environment_ that includes the Python programming language, all we have to do is tell VS Code which environment to use and it will take care of sending the code in the Python cells to the environment so that they can be executed as desired. Then the _output_ from running that code is displayed in the space below the Python cell from which it was executed. Let's try it with a simple `print()` statement!
#
# _If you did the installation of Miniconda and Jupyter properly, you should be able to execute Python cell below_

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 0:</b>   
# Confirm that Python and your conda environment are working properly by running the cell below. If you did the installation correctly, this should involve the following:
# <ol>
#     <li>Click the triangular "run" icon next to the cell.</li>
#     <li>At the top of your VSC window you will be asked to select a Python environment. Choose the conda environment you just created, <code>mude-base</code></li>
#     <li>Confirm that the output of the cell is generated as expected.</li>
#     <li>That's it!</li>
# </ol>
# <em>Note that you might also be prompted to install a few VSC extensions, if you did not already do so. Read the message carefully so you know what VSC is installing; however, note that you don't really have to do anything except hit "yes" because VSC is pretty good at getting Python to run on your computer.</em>
# </p>
# </div>

# %%
print('If you connected to an environment with Python,'
      ' this sentence will be printed below.')

# %% [markdown]
#
# ## Part 1: What is a Dictionary?
#
# When getting started, if you are simply _using_ dictionaries (we will write them later), there are really only a few fundamental things to know about a dictionary:
# 1. It is a Python data type that stores key-value pairs.
# 2. It is common to define a _key_ as a string (e.g., `'this is a string'`)
# 3. The _value_ can be almost anything. 
# 4. It uses a syntax with square brackets to get the _value_ associated with a given _key_.
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.1:</b>   
# run the cell below to see what happens when you define a dictionary and then print it.
# </p>
# </div>

# %%
test_dictionary = {'key1': 'value1', 'key2': 'value2'}
print(test_dictionary)

# %% [markdown]
# To access the value, you can use the following syntax:
#
# ```
# test_dictionary['KEY']
# ```

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.2:</b>   
# run the cell below to access the value for <code>key2</code>.
# </p>
# </div>

# %%
test_dictionary['key2']

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1.3:</b>   
# run the cell below to access add a key-value pair for <code>key3</code>.
# </p>
# </div>

# %%
test_dictionary['key3'] = 'value3'
print(test_dictionary['key3'])

# %% [markdown]
# That's it! Now you are ready to start using a dictionary with "real" data!

# %% [markdown]
# ## Part 2: Using a dictionary with "real" data
#
# In this part we illustrate how easy it is to load text-based files into notebooks, then operate on them with Python code.

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 2.1:</b>   
# open up the contents of the file <code>catchme.py</code> and <b>read</b> it. This file defines a dictionary called <code>data</code>, which we will import into our notebook and access the content by running the cell below.
# </p>
# </div>

# %%
import catchme
data = catchme.data

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 2.2:</b>   
# If the cell above ran without error, you should be use the dictionary <code>data</code> in exactly the same way as <code>test_dictionary</code>. To check whether you understand how a dictionary works, see if you can print the <em>value</em> for the <em>species</em> of this "thing" in the dictionary.
# </p>
# </div>

# %%
YOUR_CODE_HERE

# %% [markdown]
# **That's it!** This was a short PA...later ones will be longer.

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
