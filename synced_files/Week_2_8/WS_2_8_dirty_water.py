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
# # Workshop 16: Dirty Water
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
# *[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.8. Wednesday January 16, 2024.*

# %% [markdown]
# ## Introduction
#
# In this exercise we apply a few simple concepts from the textbook to derive a safety standard to regulate spills in a chemical factory. You are asked to consider an economic and societal standard, then make a recommendation to the city council.
#
# ## Case Study
#
# A city with population 10,000 uses an aquifer for its water supply, as illustrated in the figure. The city owns a factory in the region that manufactures hazardous chemicals, and recently a chemical spill occurred that resulted in 10 residents getting sick and total damages of  €7,000M*. The city is going to enforce stricter regulations on the factory, and _you have been hired to advise the city council on the maximum allowable probability of a spill occurring (per year)_. You will make a recommendation based on the more stringent criteria between economic and societal risk limits.
#
# ![](./sketch.png)
#
# Experts have been consulted and it appears under the current plan the probability of a spill is 1/100 per year. The city council is considering two strategies to upgrade the spill prevention system. A small upgrade would cost €25M and can reduce spill probability by a factor 10; a large upgrade with investment costs of €50M would reduce the probability by factor 100.
#
# The city has also considered the regulations in a nearby region which uses a maximum allowable probability of 1 person getting sick as p_f=0.01. The city agrees with this, however, they are very much _risk averse_ (that's a hint!), regarding spills with more significant consequences.
#
# _*M = million, so 7,000M is 7e9, or 7 billion. All costs in this exercise are expressed in units €M._

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 1:</b>  
#
# What is optimal strategy in terms of total cost of the system?
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 2:</b>  
#
# Assuming that the number of people in a future spill would also be 10 people, what is the maximum allowable spill probability based on societal risk standards?
#
# Make a plot of the societal risk limit and add a point for the case of the city factory.
#     
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 3:</b>  
#
# Provide your advice to the polder’s authorities for the safety standard based on the outcomes of the economic analysis (Task 1) and societal risk (Task 2).
#
# </p>
# </div>

# %% [markdown]
# ## Spill Protection System
#
# The factory has two areas where the hazardous chemical is produced, and each eara has a direct route to the groundwater table. The large spill prevention system design (mentioned above) consists of the same spill containment structure that is built twice: one at each area.
#
# During your review, you find out that a material essential to the containment structure is made in large batches, and sometimes an entire batch turns out to be faulty. The failure probability calculation for the design of the structure was made under the assumption of independent events.
#
# _Note for the exam: we won't ask you to identify or justify positive or negative correlations, but given information like correlation coefficient, you should be able to provide quantitative results regarding the impact on simple series and parallel systems, relative to the independent case._
#

# %% [markdown]
# <div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Task 4:</b>  
#
# Given the new information, advise the city whether the current estimate of failure probability is over- or under-conservative. Explain why.
#
# </p>
# </div>

# %% [markdown]
# <div style="background-color:#C8FFFF; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
# <p>
# <b>Thirsty for more?</b>   
#
# We added <a href="https://mude.citg.tudelft.nl/book/pd/reliability-component/contamination.html" target="_blank">a new interactive page</a> to the book that illustrates component reliability analysis for another part of this contaminant transport case study.
# </p>
# </div>

# %% [markdown]
# **End of notebook.**
# <h2 style="height: 60px">
# </h2>
# <h3 style="position: absolute; display: flex; flex-grow: 0; flex-shrink: 0; flex-direction: row-reverse; bottom: 60px; right: 50px; margin: 0; border: 0">
#     <style>
#         .markdown {width:100%; position: relative}
#         article { position: relative }
#     </style>
#     <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
#       <img alt="Creative Commons License" style="border-width:; width:88px; height:auto; padding-top:10px" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
#     </a>
#     <a rel="TU Delft" href="https://www.tudelft.nl/en/ceg">
#       <img alt="TU Delft" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/tu-logo/TU_P1_full-color.png"/>
#     </a>
#     <a rel="MUDE" href="http://mude.citg.tudelft.nl/">
#       <img alt="MUDE" style="border-width:0; width:100px; height:auto; padding-bottom:0px" src="https://gitlab.tudelft.nl/mude/public/-/raw/main/mude-logo/MUDE_Logo-small.png"/>
#     </a>
#     
# </h3>
# <span style="font-size: 75%">
# &copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a> TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
