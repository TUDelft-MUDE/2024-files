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

**Solution**

The algebraic expression is:

$$
\phi^{n+1}_i = \phi^{n}_i 
+ \sum_{nv} D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{nv}-\phi^n_i}{\Delta d_c} \right)
$$

where subscript $nv$ indicates each of the three "neighbor volumes."

This is implemented in the code in the `solve` method by looping over each face of the triangle, identifying whether it is an interior face, finding the triangle that shares that face, then computing the flux as described by each term in the algebraic expression.

```
flux[i] = constant*(phi_neighbor - phi)/centroid_distance
```

After each face is computed, the value of $\phi_{i}^{n+1}$ is computed as follows:

```
unknowns[time_step+1, triangle_id] = phi + np.sum(flux)
```

 As the volumes are equal-sized equilateral triangles, the code could have been made more efficient by only computing the areas, lengths and centroid-to-centroid difference _once_, outside of the loop for each triangle, but Robert ran out of time to do this.

**Question 2**

Given that in FVM the diffusion is computed as fluxes at the surfaces, what do you need to do in order to represent a Neumann boundary condition in the algebraic equation? Include the line of code that accomplishes in the `Mesh` class, along with a brief explanation of the context.

_Write your answer here._

```
reproduce the code here
```

**Solution**

For FVM, a Neumann Boundary Condition is implemented directly by identifying the side of the volume that is a boundary and replacing the partial derivative term in the previous algebraic equation.

The code implements this in the `solve` method by looping over each face of the triangle, identifying whether it is an exterior face, then assigning the proper flux which was specified in the list of boundary conditions in the notebook.

```
flux[i] = constant*self.boundary_side_types[idx][1]
```

**Question 3**

Consider the `solve` method of the class `Mesh` and write an explanation about how the FVM is solved. Mention specifically how the time and space discretization is executed; also mention specifically how the algorithm is _different_ compared to the finite difference approach (for the spatial integration). Use excerpts from the code to illustrate your answer here (example syntax is provided below). There is no need to repeat the code and explanations provided in Questions 1 and 2.

_Write your answer here._

Example code block:

```
include relevant code in your answer!
```

**Solution**

The time integration is nearly identical for FVM and FDM, if a forward Euler scheme is applied. This method is explicit, so the unknowns for a given time step can be calculated using values from the previous time step. This is apparent in the indexing illustrated in Q1, where `unknowns[time_step+1]` is computed using only information from the previous step (e.g., `phi = unknowns[time_step, ...]`).

The spatial integration is very different between FVM and FDM, as instead of looping over points in $x$ and $y$ we loop over _triangles,_ which is this line of code:

```
for triangle_id, triangle in enumerate(self.triangles)
```

Luckily the computations for each triangle are straightforward; the terms are mostly constant because the triangles are equilateral and equal size (and orthogonal; this is the subject of later questions).

Note that for the workshop, which considered a 1D problem and a 2D problem with rectangular volumes, the algebraic formulation of FVM produces identical results to the FDM. However, with 2D triangles, the loop over the surfaces illustrates quite clearly that there is a difference between these numerical schemes. 

**Question 4**

At the end of Part 2 you refined the mesh and saw that the original values for the time integration resulted in an unstable solution. Run the analysis a few more times and see if you can properly describe the situation (specifically regarding stability). Use specific results from your experiments (summarized in a Markdown table), as well as the stability criteria presented in the book to provide your explanation.

_Write your answer here_

Something about size of the triangles, time step size and should also involve the diffusivity coefficient. Markdown table should show a clear break from stable to unstable.

Starting with 

The notebook starts with the following parameters for $t_{final}$, $N_t$ and $D$:

```
mesh.solve(20, 100, 50)
```

This works find for the original mesh, but the solution is unstable for the refined mesh. Changing $N_t$ to 1000 results in a stable solution.

**Question 5**

Computations in meshes with non-equilateral triangles have added error sources that would need to be corrected to have an accurate solution. What is the impact the approach you implemented for the cases of Task 3.2, 3.3 3.4?

Briefly state the source of error introduced by the geometry of the volumes considered in each task. You only need to write a few words each.

_Write your answer here_

- Task 3.2: 
- Task 3.3: 
- Task 3.4: 

**Solution**

Stated here; notebook also contains additional information.

- Task 3.2: error due to approximated gradient not being at midpoint of face
- Task 3.3: error due to approximated gradient not being normal to the surface
- Task 3.4: both of the previous types of error are present

**Question 6**

What would you do to correct the errors in the case of task 3.3? Describe your answer qualitatively, there is no need to make computations.

_Tip: suppose that you also know the values of $\phi$ at the vertices. _

_Write your answer here_

**Solution**

When the midpoint between centroids is not on the surface that divides the two volumes, the gradient calculated between the centroids needs to be corrected. When the gradient is calculated as done in the notebook, it can be corrected by breaking it into two components; the "cross-diffusion" (the component parallel to the surface) should be removed from the calculation of the term from that face. Note, however, that this cross-diffusion should be included in the _other faces!_ This obviously makes the code more complicated, however, it can be handled in a straightforward way by looping over each face, keeping track of the components of the gradient, then summing them up after each face has been calculated.

When the midpoint between centroids is on the surface, but not located at the midpoint, our solution is inherently less accurate. If we had values of $\phi$ at the vertices, we would be able to use a midpoint rule for calculated $\phi$ on the surface, which would provide second order accuracy. To implement this in practice, we would revise the numerical scheme to define the unknowns ($\phi(x,t)$) at the vertices instead of the centroids. This would increase the size of the problem (the linear system for each time step is larger), however, the accuracy of the method would be increased. Whether this is a good idea or not depends on the type of problem and way in which it is solved.

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.