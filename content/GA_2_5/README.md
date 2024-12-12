# README for Group Assignment 2.5

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.5, Friday, Dec 13th, 2024.*

The focus of this assignment is on applying two different methods for optimization on a road network design problem. Part of the background material in the notebooks was already available in Chapter 5.11 in the textbook.

## Overview of material

- `README.md` (this file)
- `Report.md`, primary file in which you write your answers and copy plots from the notebooks. Typically, a short answer is sufficient, but please include a reasoning, justification or argumentation. Remember to use Markdown features to clearly indicate your answers for each question.
- `Analysis_LP.ipynb`, the notebook in which you apply a mixed integer linear program (MILP) to a road network design problem (NDP).
- `Analysis_GA.ipynb`, the notebook in which you apply a genetic algorithm (GA) to the same road network design problem (NDP).
- a subdirectory `./utils` containing some functions for visualization and data processing functions (which you don't need to open).
- a subdirectory `./figs` containing some figures included in the notebooks (which you don't need to open).
- a subdirectory `./input` containing data files 

## Python Environments

You can run all of the notebooks for today in the environment `mude-week-2-5` which you've been using during the workshop this week. In particular, it includes the optimization packages `pymoo` and `gurobipy`. The non-Python part of the software Gurobi was installed as part of PA 2.4.

## Task Overview
While there are two notebook this week, the programming portion is quite limited. The important thing is to interpret the output. As the models need to run for 5 minutes each, time management is very important. We suggest the following:

- Start with `Analysis_LP`. Run through it once without changing any of the code. This is needed to answer question 1 and 2 in the report. 
- Question 2 can be done independently while the model is running.
- Question 3 should be done after the first run and you need to complete question 2 first.
- `Analysis_GA` should also be run once without any changes. This is needed to answer question 5. 
- As you may want compare the solving times for the LP and GA, we recommend assigning one person to run both notebooks to ensure a consistent computing environment. 


**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
