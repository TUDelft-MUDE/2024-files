# GA 2.2 Report: M is for Modelling

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.2. Due: November 22, 2024.*

## Report Instructions

Use Markdown features to clearly indicate your answers for each question below. For example, headings (`##` and `###`), **bold** or _italic_ text, or ordered lists. Show your work using either LaTeX equations, like this:

$$
u(x=?) = \dots
$$

or by including an image with your (clearly written) handwriting. 

**Tip:** _most IDE's have live Markdown previewers. In Jupyter Lab, right-click on the Markdown file and select "Show Markdown Preview." If you are using VS Code, `ctrl + shift + v` should work (you may need the extension "Markdown All in One"). We are not sure how to do this in Deepnote._

Please keep your solutions as **concise** as possible, and, where possible, answer in **bullet points**!

## Questions

**Question 1: Boundary conditions**
    
- What boundary conditions are actively enforced? On which part of the boundary are they enforced?
    
On the remainder of the boundary, nothing is done in the implementation to enforce any boundary conditions. 
    
- Give a mathematical expression for the boundary condition that is naturally applied. **Also** describe an observation about the obtained solution that confirms that this boundary condition is indeed satisfied.
    
**Question 2: Integration scheme**

-  Which integration schemes are used to compute the element contributions to the $\mathbf{K}$ and $\mathbf{M}$ matrices? Comment on the locations and weights of the integration points. 
    
- Which changes would you make to the code to evaluate the $\mathbf{M}$-matrix with a single integration point at the center of gravity of each element? Give your answer by indicating where you would replace existing code and typing out the new lines of code as **part of your answer in this report**.

**Question 3: Time step size dependence**

Try increasing the step size $\Delta t$ and observing what happens when you rerun the code. 
- What is the reason this code does not suffer from instability for large time steps? 

Try decreasing the time step to very small numbers. If you make the time step small enough, some unphysical behavior can be observed in the solution, at least for initial time steps. 
- What is the source of this behavior? 

**Question 4: $\mathbf{B}$-matrix**
    
Shape functions in the triangular element each have the form $N_i=a_ix+b_iy+c_i$ with $i\in[1,3]$. For every $i$, the coefficients $a_i, b_i, c_i$ are computed in the code to form the B-matrix. 
 - Why does the $\mathbf{B}$-matrix inside the element not depend on $x$ and $y$?

- Give an expression for the $\mathbf{B}$-matrix in terms of these nine coefficients ($a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3$). 

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.