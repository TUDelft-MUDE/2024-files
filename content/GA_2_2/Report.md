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

**Question 1: Derivation**

Follow the steps from strong form to discretized form to derive the expression $\mathbf{M}=\int_\Omega\mathbf{N}^T\mathbf{N}\,\mathrm{d}\Omega$ in the term $\mathbf{M}\dot{\mathbf{u}}$. You will only be assessed on how you deal with the term that contains the time derivative. The other terms exactly following the recipe outlined for the [Poisson equation in 2D](https://mude.citg.tudelft.nl/2024/book/fem/poisson2d.html) in the book. 

**Question 2: Problem definition**

Investigate the code and results to find out which problem is being solved. 

- Give a mathematical description of the problem in terms of governing equation and boundary conditions. Be as specific as possible, indicating the values that are used as input to the calculation. 

- In the final visualization contour lines are visible, connecting points that have the same temperature. As the solution evolves, these contour lines remain approximately perpendicular to the boundary. Which boundary condition does this observation relate to?


**Question 3: Integration scheme**

- In the `get_element_M` function, how many integration points are used and where in the triangles are they positioned? 
    
- In the `get_element_K` a simpler implementation is used. What is the essential difference between $\mathbf{K}_e$ and $\mathbf{M}_e$ that is the reason why this simpler implementation is valid for $\mathbf{K}_e$? (The subscript $_e$ is used to indicate the contribution to the matrix from a single element, or the *element matrix*). 

**Question 4: Shape functions**
    
Investigate the shape functions for the element with index 10 in the mesh. Use the `get_shape_functions_T3` function defined in the notebook to find expressions for the shape functions in that element and check that they satisfy the shape function properties. 

- What are the coordinates of the nodes of element 10? 

- What are the shape functions of the element? 

- Assert that the shape functions satisfy the partition of unity property:

$$
\sum_i N_i(\mathbf{x}) = 1
$$

- Assert for one of the shape functions that it satisfies the Kronecker delta property

$$
N_i(\mathbf{x}_j) = \begin{cases}
  1, & i = j \\
  0, & i\neq j
\end{cases}
$$

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
