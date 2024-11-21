

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

**YOUR GROUP NAME HERE**

***Maximum score: 10 points. Every time a (1) is indicated in the solution below this is the start of (a part of) an answer that is worth 1 point***

## Task 1: Run the analysis and inspect code and results to answer the following questions ##

Show your work using either LaTeX equations, like this:

$$
\phi_i^{n+1} = \dots
$$

or by including an image with your (clearly written) handwriting. 

### 1. Boundary conditions


*1a. What boundary conditions are actively enforced? On which part of the boundary are they enforced?*

***(1)*** Dirichlet boundary conditions: $u=10$ on the bottom left edge

*1b. On the remainder of the boundary, nothing is done in the implementation to enforce any boundary conditions. Give a mathematical expression for the boundary condition that is naturally applied. Also describe an observation about the obtained solution that confirms that this boundary condition is indeed satisfied.*

***(1)*** Homogeneous Neumann boundary conditions: $\nabla u\cdot\mathbf{n}=0$. 

***(1)*** It can be observed that the gradient of the solution in perpendicular to the boundary is equal to zero. This is best observed from the 2D visualization, where iso-lines separating the different colours are normal to the boundary. 


### 2. Integration scheme
*2a. Which integration schemes are used to compute the element contributions to the $\mathbf{K}$ and $\mathbf{M}$ matrices? Comment on the locations and weights of the integration points.*

***(1)*** For the $\mathbf{M}$-matrix: 3-points at the middle of the edges of the triangle, each with weight $A/3$ where $A$ is the element area

***(1)*** For the $\mathbf{K}$-matrix there is no location because $\mathbf{K}$ does not depend on position, there is a single integration point with weight $A$ (or the integrand is just multiplied with $A$)
 
*2b. Which changes would you make to the code to evaluate the $\mathbf{M}$-matrix with a single integration point at the center of gravity of each element (give your answer by indicating where you would replace existing code and typing out the new lines of code)*

***(1)***

In the function `get_element_M`, change the lines that define `integration_locations` and `integration_weights`:

   `integration_locations = [(n1+n2+n3)/3]`
   
   `integration_weights = [element_area]`


### 3. Time step size dependence
*3a. Try increasing the step size $\Delta t$. What is the reason this code does not suffer from instability for large time steps?*

***(1)***
The time-integration scheme is Euler backward.

*3b. Try decreasing the time step to very small numbers. If you make the time step small enough, some unphysical behavior can be observed in the solution, at least for initial time steps. What is the source of this behavior?*

***(1)***
For very small $t$, the solution initially has local high gradients. The mesh is not fine enough to resolve this accurately. With small $\Delta t$ the first time steps are in this domain. 

### 4. $\mathbf{B}$-matrix
    
*4a Shape functions in the triangular element each have the form $N_i=a_ix+b_iy+c_i$ with $i\in[1,3]$. For every $i$, the coefficients $a_i, b_i, c_i$ are computed in the code to form the B-matrix. Why does the $\mathbf{B}$-matrix inside the element not depend on $x$ and $y$?*

***(1)*** 
The $\mathbf{B}$-matrix contains derivatives with respect to $x$ and $y$. Because there are only linear terms in $N_i$, these derivatives are not a function of $x$. 

*4b Give an expression for the $\mathbf{B}$-matrix in terms of these nine coefficients ($a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3$).*

***(1)***
$$
\mathbf{B} = \left[\begin{matrix}
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{matrix}\right]
$$

## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in yout Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
