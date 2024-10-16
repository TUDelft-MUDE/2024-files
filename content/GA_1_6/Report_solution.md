# Report for Group Assignment 1.6

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6, Friday, Oct 11, 2024.*

## Questions

## Part 1: Solving non-linear ODEs

**Question 1**

How close is your approximation to the exact solution $x=3$ when your initial guess is 0.01? Explain why it takes more iterations to converge when you use this value instead of a value much farther away than the solution. 

_Write your answer here._

See notebook.

**Question 2**

Include a figure of your solution for dt=0.25 s (task 2.3). 

_Your figure here._

See notebook.

**Question 3**

By trial and error, find the dt limit of stability for the explicit scheme.

_Note that an unstable condition is one that increases/decreases unbounded; an inaccurate solution that has not converged close to the "true" value is not necessarily an unstable condition._ 

_Sate the stability limit here._

The limit was between 0.3 and 0.4.

Note also that the _implicit_ scheme also has issues when the time step becomes too big, however, this is due to the Newton-Raphson scheme not converging; it is not a stability issue. The solution is stable, but it is also _terrible._

## Part 2: Diffusion equation in 1D

**Question 4**

Add an image of the stencils and the algebraic expression of the differential equations for both solution methods: central difference in space with forward and backward difference in time. 

_Insert image here._

**Question 5**

Add an image (or Latex equation) of your matrices $AT=b$ for both solution methods. Describe the differences in a few short sentences.  

_Your answer here._

See notebook.

**Question 6**

Add an image of the results corresponding to Task 3.8 at t=1500 sec and at t=10000 sec.

_Insert image here._

See notebook.

**Question 7**

From your results of task 3.4 you can observe a dependency on the parameter $\nu \Delta t / \Delta x^2$. Vary $\Delta t$ until you find the stability limit of the Explicit scheme (also print the parameter $\Delta t / \Delta x^2$). What is its value? Now, define $\Delta x$ by half (0.01 instead of 0.02) and vary $\Delta t$ until you find its stability limit and print the parameter $\Delta t / \Delta x^2$. Are the values similar? What is the implication for the computational time?

_Your answer should include a couple sentences as an explanation, as well as the values of $\Delta t$ at the limit of stability and the computation time for each approach (see last task of WS 1.6 solution for an example of tracking computation time in Python)._

_Write your answer here_

probably between 20 and 200 time steps (with original values of t0 and t_end)

stability threshold for diffusion should be nu*\Delta t/\Delta x^2 = 0.5 (but we forgot the nu in the equation above, so students will report different ratios)

a good answer would have been:
- find a bad dt
- change dx
- find another bad dt
- find that the ratio dt/dx^2 is about the same for both cases
- (we point out in solution that if you *nu it is around 0.5)

Isabel: grade based on the good answer and we will adjust.

**Question 8**

For the implicit scheme, try to find a $\Delta t$ value for which the solution is not reasonable. State your result and explain.

_Write your answer here_

**Question 9**

Considering the non-linear ODE and the PDE results, would you say that Implicit methods are always better than Explicit methods? State "yes" or "no" and provide a brief explanation (2-3 sentences).

_Insert image here_

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.