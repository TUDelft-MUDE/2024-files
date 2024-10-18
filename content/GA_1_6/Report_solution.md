# Report for Group Assignment 1.6

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6, Friday, Oct 11, 2024.*

## Questions

## Part 1: Solving non-linear ODEs

**Question 1**

How close is your approximation to the exact solution $x=3$ when your initial guess is 0.01? Explain why it takes more iterations to converge when you use this value instead of a value much farther away than the solution. 

_Write your answer here._

Really close. The solution found is  3.000000000008298  it took  11  iterations to converge. When the initial guess is close to 0, the slope will also be close to 0, and the first "improved" guess is 450.005, very far from the solution! Then, the slope at that point is more reasonable and approaches rapidly the solution.  


**Question 2**

Include a figure of your solution for dt=0.25 s (task 2.3). 

_Your figure here._

_See the plot, task 2.3, of the analysis solution notebook._

See notebook.

**Question 3**

By trial and error, find the dt limit of stability for the explicit scheme.

_Note that an unstable condition is one that increases/decreases unbounded; an inaccurate solution that has not converged close to the "true" value is not necessarily an unstable condition._ 

_Sate the stability limit here._

The stability limit seemed to be between 0.35s and 0.4s when looking at the beginning of the plot. However, the error remains bounded. So, under the strict definition of stability given above, there does not seem to be a limit if we extend the plot to include larger values of time in the x axis. The fact that the function is nonlinear, dependent on harmonic functions, makes it quite complicated to confidently state a limit.    

Note also that the _implicit_ scheme also has issues when the time step becomes too big. This is due to the Newton-Raphson scheme not converging; it is not a stability issue. In this case, the solution does not converge starting with dt=0.35s. The solution is unconditionally stable, but it is stoped at the very beginning. 

## Part 2: Diffusion equation in 1D

**Question 4**

Add an image of the stencils and the algebraic expression of the differential equations for both solution methods: central difference in space with forward and backward difference in time. 

_Insert image here._

The algebraic expression and stencil using Forward Difference in time and Central Difference in space:

$$ 
T^{j+1}_{i} = T^j_i + \frac{\nu \Delta t}{\Delta x^2} \left(T^j_{i+1}-2T^j_i+T^j_{i-1}\right)
$$

_____o_____
__o__o__o__ 

The algebraic expression and stencil of Backward Difference in time and Central Difference in space:

$$ 
T^{j+1}_{i} = T^j_i + \frac{\nu \Delta t}{\Delta x^2} \left(T^{j+1}_{i+1}-2T^{j+1}_i+T^{j+1}_{i-1}\right)
$$

and rewritten for convenience, unknowns from one side and knowns from the other:

$$ 
T^{j+1}_{i} - \frac{\nu \Delta t}{\Delta x^2} \left(T^{j+1}_{i+1}-2T^{j+1}_i+T^{j+1}_{i-1}\right)  = T^j_i 
$$

__o__o__o__ 
_____o_____

**Question 5**

Add an image (or Latex equation) of your matrices $AT=b$ for both solution methods. Describe the differences in a few short sentences.  

_Your answer here._

For the explicit scheme:

For the explicit scheme:

$$
AT = b\\

T = \begin{bmatrix}
T_1^{j+1} \\
T_2^{j+1} \\
\vdots \\
T_{n-1}^{j+1}
\end{bmatrix}\\
\newline
A = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix}\\
\newline
b = \begin{bmatrix}
T^j_1 + \frac{\nu \Delta t}{\Delta x^2} \left(T^j_{0}-2T^j_1+T^j_{2}\right) \\
T^j_2 + \frac{\nu \Delta t}{\Delta x^2} \left(T^j_{1}-2T^j_2+T^j_{3}\right) \\
\vdots \\
T^j_{n-1} + \frac{\nu \Delta t}{\Delta x^2} \left(T^j_{n-2}-2T^j_{n-1}+T^j_{n}\right)
\end{bmatrix}


$$

For the implicit scheme:

$$
AT = b\\

T = \begin{bmatrix}
T_1^{j+1} \\
T_2^{j+1} \\
\vdots \\
T_{n-1}^{j+1}
\end{bmatrix}\\
\newline
A = \begin{bmatrix}
1+2\frac{\nu \Delta t}{\Delta x^2} & -\frac{\nu \Delta t}{\Delta x^2} & 0 & \cdots & 0 \\
-\frac{\nu \Delta t}{\Delta x^2} & 1+2\frac{\nu \Delta t}{\Delta x^2} & -\frac{\nu \Delta t}{\Delta x^2} & \cdots & 0 \\
0 & -\frac{\nu \Delta t}{\Delta x^2} & 1+2\frac{\nu \Delta t}{\Delta x^2} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1+2\frac{\nu \Delta t}{\Delta x^2}
\end{bmatrix}\\
\newline
b = \begin{bmatrix}
T_1^j + \frac{\nu \Delta t}{\Delta x^2}T_0^j \\
T_2^j \\
\vdots \\
T_{n-1}^j + \frac{\nu \Delta t}{\Delta x^2}T_n^j
\end{bmatrix}

$$

The A matrix is an identity matrix for the CDS-FDT case. The b vector is elaborated where the boundaries are implemented intrinsically. In the latter case, CDS-BDT, the matrix A is tridiagonal and the boundaries in the b vector are plainly implemented. No iteration is required in the implicit scheme because the dependence on time and space is linear, a.k.a., the power of the unknowns is 1.  

**Question 6**

Add an image of the results corresponding to Task 3.8 at t=1500 sec and at t=10000 sec.

_Insert image here._

See figure in the analysis solution notebook. At t=1500 sec a parabola connecting the Dirichlet BC are observed with a minimum value around x=0.15m. At t=10000 sec an almost straight line connecting the Dirichlet BC are observed, the steady state solution is almost reached around this moment. 

**Question 7**

From your results of task 3.4 you can observe a dependency on the parameter $\nu \Delta t / \Delta x^2$. Vary $\Delta t$ until you find the stability limit of the Explicit scheme (also print the parameter $\Delta t / \Delta x^2$). What is its value? Now, define $\Delta x$ by half (0.01 instead of 0.02) and vary $\Delta t$ until you find its stability limit and print the parameter $\Delta t / \Delta x^2$. Are the values similar? What is the implication for the computational time?

_Your answer should include a couple sentences as an explanation, as well as the values of $\Delta t$ at the limit of stability and the computation time for each approach (see last task of WS 1.6 solution for an example of tracking computation time in Python)._

_Write your answer here_

The dx could have been interpreted as dx=0.3/15=0.02m or dx=0.3/14=0.0214..m. If the former was used, the stability limit was at 50-51 seconds and for dx/2 it was about 13 seconds. If the later was used, the stability limit was 58-59 seconds and for dx/2 it was about 15 seconds. The computational time increases a lot since not only the grid contains more points when refining dx but also dt has to be reduced by a factor 3 (for this case). If the parameter $\nu \Delta t / \Delta x^2$ was printed, then a value of about 0.5 should have been found for both cases, if $\nu$ was not included, then a value about 127000 should have been found.

**Question 8**

For the implicit scheme, try to find a $\Delta t$ value for which the solution is not reasonable. State your result and explain.

_Write your answer here_

There does not seem to be a limitation of $\Delta t$, it can be quite large, even 1000 and it does reach the stable state. Its limitation would be related to the desired accuracy of the solution, as it still has an error related to the time step. However, as this was not a constraint, results are reasonable for absurdly large time steps. 

**Question 9**

Considering the non-linear ODE and the PDE results, would you say that Implicit methods are always better than Explicit methods? State "yes" or "no" and provide a brief explanation (2-3 sentences).

_Insert image here_

Explicit methods are not better than Implicit ones and viceversa, they both have advantages and disadvantages. The former is easier to schematize and to program but it normally requires smaller time steps to retain its stability and a reasonable solution. The latter is more complex for non-linear problems but gives the flexibility of using larger time steps. However, under some cases, the iteration method may not converge neither posing also a limit to the usable time step.

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.