# Report for Group Assignment 2.1

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1, Friday, Nov 15, 2024.*

## Questions

**Question 1**

Which is the algebraic expression for computing $\phi_{i}^{n+1}$ for the case of three interior volumes? Write the expression here and reproduce the code that accomplishes this in the `Mesh` class. Be sure to indicate clearly which values of $\phi$ are for the volume of interest and the (arbitrary) "neighbor" volumes.

_Write your answer here._

```
reproduce code here
--> remember there are tips in the README!
```

**Question 2**

Given that in FVM the diffusion is computed as fluxes at the surfaces, what do you need to do in order to represent a Neumann boundary condition in the algebraic equation? Include the line of code that accomplishes in the `Mesh` class, along with a brief explanation of the context.

_Write your answer here._

```
reproduce the code here
```

**Question 3**

Consider the `solve` method of the class `Mesh` and write an explanation about how the FVM is solved. Mention specifically how the time and space discretization is executed; also mention specifically how the algorithm is _different_ compared to the finite difference approach (for the spatial integration). Use excerpts from the code to illustrate your answer here (example syntax is provided below). There is no need to repeat the code and explanations provided in Questions 1 and 2.

_Write your answer here._

Example code block:

```
include relevant code in your answer!
``` 

**Question 4**

At the end of Part 2 you refined the mesh and saw that the original values for the time integration resulted in an unstable solution. Run the analysis a few more times and see if you can properly describe the situation (specifically regarding stability). Use specific results from your experiments (summarized in a Markdown table), as well as the stability criteria presented in the book to provide your explanation.

_Write your answer here_

**Question 5**

Computations in meshes with non-equilateral triangles have added error sources that would need to be corrected to have an accurate solution. What is the impact the approach you implemented for the cases of Task 3.2, 3.3 3.4?

Briefly state the source of error introduced by the geometry of the volumes considered in each task. You only need to write a few words each.

_Write your answer here_

- Task 3.2: 
- Task 3.3: 
- Task 3.4: 

**Question 6**

What would you do to correct the errors in the case of task 3.3? Describe your answer qualitatively, there is no need to make computations.

_Tip: suppose that you also know the values of $\phi$ at the vertices. _

_Write your answer here_

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.