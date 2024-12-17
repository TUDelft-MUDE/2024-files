---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.5
---

# WS 2.2: More support

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.2. For: 20 November, 2024*


In the book, the finite element derivation and implementation of rod extension (the 1D Poisson equation) is presented. In this workshop, you are asked to do the same for a slightly different problem.

## A modification to the PDE: continuous elastic support

<p align="center">
<img src="https://raw.githubusercontent.com/fmeer/public-files/main/barDefinition-2.png" width="400"/>
</p>

For this exercise we still consider a 1D rod. However, now the rod is elastically supported. An example of this would be a foundation pile in soil. 

The problem of an elastically supported rod can be described with the following differential equation:

$$ -EA \frac{\partial^2 u}{\partial x^2} + ku = f $$

with:

$$
u = 0, \quad \text{at} \quad x = 0 \\
EA\frac{\partial u}{{\partial x}} = F, \quad \text{at} \quad x = L
$$

This differential equation is the inhomogeneous Helmholtz equation, which also has applications in dynamics and electromagnetics. The additional term with respect to the case without elastic support is the second term on the left hand side: $ku$. 

The finite element discretized version of this PDE can be obtained following the same steps as shown for the unsupported rod in the book. Note that there are no derivatives in the $ku$ which means that integration by parts does not need to be applied on this term. Using Neumann boundary condition (i.e. an applied load) at $x=L$ and a constant distributed load $f(x)=q$, the following expression is found for the discretized form:

$$\left[\int \mathbf{B}^T EA \mathbf{B} + \mathbf{N}^T k \mathbf{N} \,dx\right]\mathbf{u} = \int \mathbf{N}^T q \,d x + \mathbf{N}^T F \Bigg|_{x=L} $$


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1: Derive the discrete form</b>   

Derive the discrete form of the PDE given above. You can follow the same steps as in the book for the term with $EA$ and the right hand side, but now carrying along the additional term $ku$ from the PDE. Show that this term leads to the $\int\mathbf{N}^Tk\mathbf{N}\,dx$ term in the $\mathbf{K}$-matrix. 
</p>
</div>


***Your derivation here***


## Modification to the FE implementation

The only change with respect to the procedure as implemented in the book is the formulation of the $\mathbf{K}$-matrix, which now consists of two terms:

$$ \mathbf{K} = \int \mathbf{B}^TEA\mathbf{B} + \mathbf{N}^Tk\mathbf{N}\,dx $$

To calculate the integral exactly we must use two integration points.

$$ \mathbf{K_e} = \sum_{i=1}^{n_\mathrm{ip}} \left(\mathbf{B}^T(x_i)EA\mathbf{B}(x_i) + \mathbf{N}^T(x_i) k\mathbf{N}(x_i) \right) w_i$$


<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">
<p>
<b>Task 2: Code implementation</b>   

The only change needed with respect to the implementation of the book is in the calculation of the element stiffness matrix. Copy the code from the book and add the term related to the distributed support in the right position. 
    
Use the following parameters: $L=3$ m, $EA=1000$ N, $F=10$ N, $q=0$ N/m (all values are the same as in the book, except for $q$). Additionally, use $k=1000$ N/m$^2$.

Remarks:

- The function <code>evaluate_N</code> is already present in the code in the book
- The <code>get_element_matrix</code> function already included a loop over two integration points
- You need to define $k$ somewhere. To allow for varying $k$ as required below, it is convenient to make $k$ a second argument of the <code>simulate</code> function and pass it on to lower level functions from there (cf. how $EA$ is passed on)

Check the influence of the distributed support on the solution:

- First use $q=0$ N/m and $k=1000$ N/$mm^2$
- Then set $k$ to zero and compare the results
- Does the influence of the supported spring on the solution make sense?
</p>

</div>



```python
# YOUR_CODE_HERE
```

<div style="background-color:#AABAB2; color: black; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px; width: 95%">

<p>

<b>Task 3: Investigate the influence of discretization on the quality of the solution</b>

- How many elements do you need to get a good solution?
- How about when the stiffness of the distributed support is increased to $k=10^6$ N/$m^2$
</p>

</div>

```python
# YOUR_CODE_HERE
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
