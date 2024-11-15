# README for Group Assignment 2.1

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1, Friday, Nov 15, 2024.*

_You can access this assignment with the following link: [classroom.github.com/a/OCrfG9X1](https://classroom.github.com/a/OCrfG9X1)._

GA 2.1 applies the finite volume method (FVM) to a 2D problem with a fun geometry.

There are several files to be aware of:

1. `README.md`
2. A notebook `Analysis.ipynb` and `Report.md` files for carrying out analysis and answering questions as done in Q1.
3. A notebook `mesh_tips.ipynb` which provides explanation and examples for using the class `Mesh`, an understanding of which is critical for carrying out the necessary analysis.
4. A file `utilities.py` that contains code to solve the FVM problem.

The file `Analysis.ipynb` contains all instructions for completing the tasks for this assignment, divided into three Parts. Only the second part uses the files `utilities.py` and `mesh_tips.ipynb`.

## Task Overview

Here are a few tips to help you get through this assignment efficiently with your group members:
1. _Everyone_ should do Part 1, which carries out the discretization and formulation of algebraic equations; skills that are essential for the exam!
2. One subgroup can look at `mesh_tips.ipynb` to help with Part 2 of the notebook; select someone that got through the PA smoothly, as it is similar.
3. Part 3 can be done independently and does not require any programming.
4. For the Report, we recommend you wait until finishing Part 2 before starting to answer questions. 

## A note about the code

Note that you are not expected to understand the entire code like the syntax of the if statements and loops, or the way in which the boundaries or sides are defined and stored; however, you should be able to recognize high level steps of the algorithm, such as looping over triangles, sides, etc, and especially the time and "space" integration loops.

## Using code blocks in your Report

Some of the questions in the Report ask you to reproduce code. This is easy to do with a Markdown "code block" which uses three "backticks"

```
code block
```

You can also use single backticks for `inline code`. The backticks are usually found at the top left of your keyboard, along with the tilde. View this file in raw text mode to see what they look like.

When including code in your answers you should adjust the tabs/margins and shorten things in order to focus on the key parts of the algorithm.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.