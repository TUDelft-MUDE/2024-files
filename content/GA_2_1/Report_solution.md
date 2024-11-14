# Report for Group Assignment 2.1

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1, Friday, Nov 15, 2024.*

## Questions

**Question 1**

Which algebraic expression did you reach for the interior volumes? Write here your answer for Task 1.5  

_Write your answer here._

ANSWER: SEE SOLUTION NOTEBOOK

**Question 2**

Consider the `solve` method of the class `Mesh` and write an explanation about how the FVM is solved. Mention specifically how the time and space discretization is executed; mention specifically how the algorithm is _different_ compared to the finite difference approach (for the spatial integration). Use excerpts from the code to illustrate your answer here (example syntax is provided below).

_Write your answer here._

Example code block:

```
include relevant code in your answer!
```
_You should adjust the tabs/margins and shorten things in order to focus on the key parts of the algorithm._

**Question 3**

Given that in FVM the diffusion is computed as fluxes at the surfaces, what do you need to do in order to implement a Neumann boundary condition?   

_For FVM, a Neumann Boundary Condition is implemented directly by identifying the side of the volume that is a boundary and replacing this contribution in the previous algebraic equation._



**Question 4**

SOMETHING ABOUT RUNNING THE ANALYSES AND FINDING STABILITY LIMITS. MAKE A MARKDOWN TABLE TO SUMMARIZE SEVERAL VALUES OF TRIANGLES SIZES AS WELL AS INITIAL CONDITIONS AND BOUNDARY CONDITIONS.

_Write your answer here_

**Question 5**

_Write your answer here_

**Question 6**


_Write your answer here_

**Question 7**

Computations in meshes with non-equilateral triangles have added error sources that would need to be corrected to have an accurate solution. What is the impact the approach you implemented for the cases of Task 3.2, 3.3 3.4?

_Write your answer here_
_Task 3.2_
_Task 3.3_
_Task 3.4_

ANSWER: SEE SOLUTION OF TASKS 3.2,3.3,3.4 IN THE NOTEBOOK

**Question 8**

What would you do to correct the errors in the case of task 3.3? Describe in words, there is no need to make computations.

Tip: suppose that you also know the values of $\phi$ at the vertices. 

_Write your answer here_

ANSWER: THE CROSS-DIFFUSION NEEDS TO BE CORRECTED BY ACCOUNTING FOR THE PHI FLUX ALONG THE SURFACE TO CORRECT THE DIFFUSION RATE OF CHANGE IN THE DIRECTION OF CONNECTING CENTROIDS. 


**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.