# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # PA 1.3: Data Cleaning and Boosting Productivity
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3. Due: before Friday, Sep 20, 2023.*

# This notebook consists of two parts:
#
# 1. Data Cleaning, with task 1.1 - 2.6
# 2. Boosting Productivity, with task 3.1 - 3.10
#
# **Remember that there is a survey that must be completed to pass this PA.**
#
# [Here is a link to the survey](https://forms.office.com/e/saRwPUyL8d).

# ## Data cleaning
#
# Often we get data in a file that contains unexpected and odd things inside. If not removed in a proper way, they can cause problems in our analysis. For example, NaNs, infinite values, or just really large outliers may cause things in our code to behave in an unexpected way. It is good practice to **get in the habit of visualizing and processing datasets before you start using them!** This programming assignment will illustrate this process.
#
# Topics in this assignment includes two tasks: 
# 1. Finding "odd" values in an array and removing them
# 2. Using plots to identify other "oddities" that can be removed
#
# We will need one csv file, `data_2.csv`, to complete this assignment.

# +
# use the mude-base environment

import numpy as np
import matplotlib.pyplot as plt
# -

# ### Task 1: Importing and Cleaning the array
#
# In a previous week we looked at how to read in data from a csv, plot a nice graph and even find the $R^2$ of the data. This week, an eager botany student, Johnathan, has asked us to help him analyze some data: 1000 measurements have just been completed over the 100 m of greenhouse and are ready to use in `data_2.csv`. Jonathan happens to have a lot of free time but not that much experience taking measurements. Thus, there is some noise in the data and some problematic data that are a result of an error in the measurement device. Let's help them out!

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.1:</b>   
# Import the data as 2 numpy arrays: distance and temperature. Tip, makes use of the function <code>numpy.genfromtxt</code>.
# </p>
# </div>

distance, temperature = np.genfromtxt("auxiliary_files/data_2.csv", skip_header = 1, delimiter=",", unpack=True)

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.2:</b>   
# In the code cell below, evaluate the size of the array.
# </p>
# </div>

temperature.size

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.3:</b>   
#  Check if there are NaN (not a number) values in the temperature array. You can use the numpy method <code>isnan</code>, which returns a boolean vector (False if it is not a NaN, and True if it is a NaN). Save the result in the variable <code>temperature_is_nan</code>. The code block below will also help you inspect the results.
# </p>
# </div>

# +
temperature_is_nan = np.isnan(temperature)

print("The first 10 values are:", temperature_is_nan[0:10])
print(f"There are {temperature_is_nan.sum()} NaNs in array temperature")
# -

# Let's slice the array using the `temperature_is_nan` array we just found to eliminate the NaNs. We can use the symbol `~`, which denotes the opposite: we want to keep those where np.isnan gives False as an answer.

temperature = temperature[~temperature_is_nan]

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.4:</b>   
# Check the size again, and make sure you recognize that we over-wrote the variable `temperature`. This will have an impact on other cells where you use this variable, for example, if you re-run the cell below Task 1.3, the result will be different, because the array contents have changed!
#
# How big is the array now? How many values were removed?
# </p>
# </div>

temperature.size


# But now we have a problem: our `distance` array still has the entries that correspond to the bad entries in `temperature`. We can see that the dimensions of the arrays no longer match:

distance.size == temperature.size

# Also, we don't know what the index of the removed values were, since we over-wrote `temperature`! Luckily we have our `temeprature_is_nan` array, which records the indices with Nans, which we can also use to update our `distance` array.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 1.5:</b>   
# Use the boolean array from Task 1.3 to remove the matching entries in the distance array, then check that it has the same length as temperature.
# </p>
# </div>

distance = distance[~temperature_is_nan]
distance.size==temperature.size

# ### Task 2: Visualizing the Dataset
#
# Now we can plot the temperature with distance to see what it looks like.

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# It looks like there are some outliers in the dataset still! Let's investigate:

print(temperature.min())
print(temperature.max())

# The values are suspcious since they are +/-999...this is a common error code with some sensors, so we can assume that they can be removed from the dataset. We can easily remove these erroneous values of temperature, but this time we will use a different method than before. The exclamation mark before an equal sign, `!=`, denotes "not equal to." We can use this as a logic operator to directly eliminate the values in one line. For example:
# ```
# array_1 = array_1[array_2 != -999]
# ```

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.1:</b>   
# Use the "not equal to" operator to re-define temperature and distance such that all the temperatures with -999 are removed (don't do the +999 values yet!). Keep in mind that the order of the arrays matters: if you reassign temperature, you won't have the information any more to fix distance!!!
# </p>
# </div>

distance = distance[temperature!=-999]
temperature = temperature[temperature!=-999]

# Are the arrays the same size still? If you did it correctly, they should be.

print(distance.size == temperature.size)
temperature.size

# For the +999 values we will use yet another method, a combination of the previous two.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.2:</b>   
#     Use the not equal to operator <b>and</b> a boolean array to define an array "mask" that will help you remove the data corresponding to temperatures with +999.
# </p>
# </div>

# We can also do it with a boolean for data_y.

mask = temperature!=999
distance = distance[mask]
temperature = temperature[mask]

# The array is named "mask" because this process utilizes **masked arrays**...you can read more about it [here](https://python.plainenglish.io/numpy-masks-in-python-d8c13509fbc8).

# Anyway, now that we have removed the annoying +/-999 values, we can finally start to see our dataset more clearly:

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# Looks good! But wait—there also appear to be some values in the array that are not physically possible! We know for sure that there was nothing cold in the greenhouse during the measurements; also it's very likely that a "0" value could have come from an error in the sensor.
#
# See if you can apply the `numpy` method `nonzero` to remove zeros from the array. Hint: it works in a very similar way to `isnan`, which we used above.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.3:</b>   
#     Use <code>nonzero</code> to remove the zeros.
# </p>
# </div>

distance = distance[np.nonzero(temperature)]
temperature = temperature[np.nonzero(temperature)]

# It also seems quite obvious that the values above 50 degrees are also not physically possible (or perhaps Jonathan was standing near an oven?!). In any case, they aren't consistent with the rest of the data, so we should remove them.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.4:</b>   
#     Use an inequality, <code><</code> to keep all values less than 50.
# </p>
# </div>

distance = distance[temperature<50]
temperature = temperature[temperature<50]

# Now let's take another look at our data:

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# Let's pretend that there is a systematic error in our measurement device because it was not calibrated properly. As a result, all observations below 15 degrees need to be corrected by multiplying the measurement by 1.5. Numpy actually makes it very easy to replace the contents of an array based on a condition using the `where` method!

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.5:</b>   
#     Play with the cell below to understand what the <code>where</code> method does (i.e., replacement)—it's very useful to know about!
# </p>
# </div>

temperature = np.where(temperature > 15, temperature, temperature * 1.5)

# Remember you can investigate the `where` function in a notebook easily by executing `np.where?`. Try it and read the documentation!
#
# Let's plot the array again to see what happened (you'll have to compare the two plots carefully to see the difference). Remember, that if you rerun the cell above many times, it will over-write `temperature`, so you will probably need to restart the kernel a few times to reset the values.

plt.plot(distance, temperature, "ok", label="Temperature")
plt.title("Super duper greenhouse")
plt.xlabel("Distance")
plt.ylabel("Temperature")
plt.show()

# Now that we are done cleaning the data, let's learn about it. 

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 2.6:</b>   
#     Calculate the mean and variance of temperature. Use built-in numpy functions.
# </p>
# </div>

# +
mean_temperature = temperature.mean()
print(f"{mean_temperature = :.3f}")

variance_temperature = temperature.var()
print(f"{variance_temperature = :.3f}")
# -

# ## Boosting productivity

# Have you ever gotten frustrated by sharing code with your friends? How nice would it be to do it the Google-Docs style and work on the same document simultaneously! Maybe you know Deepnote, which does this in an online interface. But there's a better solution: [Visual Studio Live Share](https://visualstudio.microsoft.com/services/live-share/)!
#
# In this PA you'll install the required extension in VS Code. You'll use this extension on Wednesday in class, and can freely use it in your future career!

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.1:</b>   
#     Download, install and login in the Visual Studio Live Share Extension from the Visual Studio Marketplace as explained in the <a href="https://mude.citg.tudelft.nl/2024/book/external/learn-programming/book/install/ide/vsc/vs_live_share.html">book</a>.
# </p>
# </div>

# After installing and signing into Visual Studio Live Share, you'll share a project with yourself to test the collaboration session.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.2:</b>   
#     Use your normal workflow to open the folder where this notebook is situated in VS Code.
# </p>
# </div>

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.3:</b>   
#     Start a collaboration session: select <strong>Live Share</strong> on the status bar or select <strong>Ctrl+Shift+P</strong> or <strong>Cmd+Shift+P</strong> and then select <strong>Live Share: Start collaboration session (Share)</strong>.
# </p>
# <img src="https://learn.microsoft.com/en-us/visualstudio/liveshare/media/install-live-share-visual-studio-code/live-share-button-status-bar.png" alt="Live Share Button">
# <p>
# The first time you share, your desktop firewall software might prompt you to allow the Live Share agent to open a port. Opening a port is optional. It enables a secured direct mode to improve performance when the person you're working with is on the same network as you. For more information, see <a href="https://learn.microsoft.com/en-us/visualstudio/liveshare/reference/connectivity#changing-the-connection-mode">changing the connection mode</a>.
# </p>
# </div>

# An invitation link will be automatically copied to your clipboard. You'll use this link to interact with yourself in this assignment. If you want to collaborate with others, you can share this link with them to open up the project in their browser or own VS Code.
#
# You'll also see the **Live Share** status bar item change to represent the session state.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.4:</b>   
#     Copy the invitation link in your web browser.
# </p>
# </div>

# A web version of Visual Studio Code will open in your browser. Continue there.

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.5:</b>   
#     Login using the same steps as in task 3.2.
# </p>
# </div>

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.6:</b>   
#     Go back to your desktop participant of VS code and try typing a few lines of correct code in the cell below. Do you see the same change in the browser happening live?
# </p>
# </div>

print('hello world')

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.7:</b>   
#     Go back to your browser version of VS code and try running the cell. Note that this requires the browser participant to request access. Request that access and approve it in the desktop participant of VS code. In the desktop participant you now need to select your Python environment. Does the cell run? Do you see the output in both participants? Make sure the output doesn't show an error!
# </p>
# </div>

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.8:</b>   
#     Let's try to make sense of what's happening. Where is the code executed? Run the following code cell from the browser participant. On which computer does this collaborative session run?
# </p>
# </div>

import platform
print("Running on a", platform.system(),"machine named:", platform.node())

# That's cool right? Imagine what you could do with this...

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.9:</b>   
#     Explore some of the functionalities of the Live Share (session chat, following). More information can be found <a href="https://learn.microsoft.com/en-us/visualstudio/liveshare/">here</a>.
# </p>
# </div>

# <div style="background-color:#AABAB2; color: black; width:95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
# <p>
# <b>Task 3.10:</b>   
#     Stop the collaboration session on the desktop-participant by opening the Live Share view on the <strong>Explorer</strong> tab or the <strong>VS Live Share tab</strong> and select the <strong>Stop collaboration session</strong> button:
# </p>
# <img src="https://learn.microsoft.com/en-us/visualstudio/liveshare/media/vscode-end-collaboration-viewlet.png">
# <p>
# Don't forget to save your work! You could have done that both on the browser-participant as on the desktop-participant!
# </p>
# <p>
# After stopping the collaboration session, your web-participant will be notified that the session is over. It won't be able to access the content and any temp files will automatically be cleaned up.
# </p>
# </div>

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
