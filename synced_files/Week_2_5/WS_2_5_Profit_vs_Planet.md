---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: mude-week-2-5
    language: python
    name: python3
---

# WS 2.5: Profit vs Planet

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.5, Optimization. For: December 11, 2024*


## Part 1: Overview and Mathematical Formulation

A civil engineering company wants to decide on the projects that they should do. Their objective is to minimize the environmental impact of their projects while making enough profit to keep the company running.

They have a portfolio of 6 possible projects to invest in, where A, B , and C are new infrastructure projects (so-called type 1), and D, E, F are refurbishment projects (so-called type 2).

The environmental impact of each project is given by $I_i$ where $i \in [1,(...),6]$ is the index of the project. $I_i=[90,45,78,123,48,60]$

The profit of each project is given by $P_i$ where $i\in [1,(...),6]$ is the index of the project: $P_i=[120,65,99,110,33,99]$

The company is operating with the following constraints, please formulate the mathematical program that allows solving the problem:

- The company wants to do 3 out of the 6 projects
- the projects of type 2 must be at least as many as the ones of type 1 
- the profit of all projects together must be greater or equal than $250$ ($\beta$)


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 1: Writing the mathematical formulation</b>   

Write down every formulation and constraint that is relevant to solve this optimization problem.
</p>
</div>


_Your answer here._


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2: Setting up the software</b>   

We'll continue using Gurobi this week, which you've set up in last week's PA. We'll use some other special packages as well. **Therefore, be sure to use the special conda environment created for this week.**

</p>
</div>

```python
import gurobipy as gp
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 3: Setting up the problem</b>   

Define any variables you might need to setup your model.
</p>
</div>

```python
# Project data
YOUR_CODE_HERE

# Minimum required profit
YOUR_CODE_HERE

# Number of projects and types
YOUR_CODE_HERE
```

## Part 2: Create model with Gurobi

**Remember that examples of using Gurobi to create and optimize a model are provided in the online textbook**, and generally consist of the following steps (the first instantiates a class and the rest are executed as methods of the class):

1. Define the model (instantiate the class)
2. Define variables
3. Define objective function
4. Add constraints
5. Optimize the model

Remember, you can always ask for help to understand a function of gurobi
```
help(gurobipy.model.addVars)
```



<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 4: Create the Gurobi model</b>   

Create the Gurobi model, set your decision variables, your function and your constrains. Take a look at the book for an example implementation in Python if you don't know where to start.
</p>
</div>

```python
YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 5: Display your results</b>   

Display the model in a good way to interpret and print the solution of the optimization.
</p>
</div>

```python
YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 6: Additional constraint</b>   

Solve the model with an additional constraint: if project 1 is done then the impact of all projects together should be lower than $\gamma$ with $\gamma=130$.

In the first cell you should add the constraint, then in a second cell optimize the model.

</p>
</div>

```python
YOUR_CODE_HERE
```

```python
YOUR_CODE_HERE
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
