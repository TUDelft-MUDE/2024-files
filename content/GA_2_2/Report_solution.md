# GA 2.2 Report: M is for Modelling

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.2. Due: November 22, 2024.*

***Maximum score: 10 points. Every time a (1) is indicated in the solution below this is the start of (a part of) an answer that is worth 1 point***

## Task 1: Run the analysis and inspect code and results to answer the following questions ##

Show your work using either LaTeX equations, like this:

$$
\phi_i^{n+1} = \dots
$$

or by including an image with your (clearly written) handwriting. 


**Question 1: Derivation**

Follow the steps from strong form to discretized form to derive the expression $\mathbf{M}=\int_\Omega\mathbf{N}^T\mathbf{N}\,\mathrm{d}\Omega$ in the term $\mathbf{M}\dot{\mathbf{u}}$. You will only be assessed on how you deal with the term that contains the time derivative. The other terms exactly following the recipe outlined for the [Poisson equation in 2D](https://mude.citg.tudelft.nl/2024/book/fem/poisson2d.html) in the book. 




**solution**

Time dependent 2D Poisson equation:

$$
\frac{\partial u}{\partial t} = \nu \left(\frac{\partial^{2} u}{{\partial x}^{2}} + \frac{\partial^{2} u}{{\partial y}^{2}}\right) + q
$$

$$
\frac{\partial u}{\partial t} = \nu \nabla^{2} u   + q
$$

1. Integrate and introduce weight functions

$$
\int_{\Omega} w \frac{\partial u}{\partial t} d\Omega = \int_{\Omega} w \nu  \nabla^2 u d\Omega + \int_{\Omega} wq d\Omega
$$

2. Integration by parts: changing the second derivative, changing sign and introducing boundary condition (not necessary for the term with time-derivative)

$$
\int_{\Omega} w \frac{\partial u}{\partial t} d\Omega = - \int_{\Omega} \nu \nabla w \cdot \nabla u  d\Omega + \int_{\Gamma}  w \mu \nabla u \cdot \bar{n} d\Gamma + \int_{\Omega} w q d\Omega
$$

3. Substitute boundary condition (w= 0 on $\Gamma_D$ and $\mu \nabla u \cdot \mathbf{n}$ on $\Gamma_N$) (not necessary for the term with time-derivative):

$$\int_{\Omega} w \frac{\partial u}{\partial t} d\Omega + \int_{\Omega} \nu \nabla w \cdot \nabla u  d\Omega = \int_{\Gamma N}  w h d\Gamma_N + \int_{\Omega} w q d\Omega$$

4. Discretization:

$$
u^h = \mathbf{N}\mathbf{u}
$$ 
$$w^h = \mathbf{N}\mathbf{w}$$


$$\nabla u^h  = \mathbf{B}\mathbf{u}$$
$$\nabla w^h = \mathbf{B}\mathbf{w}$$

5. Substitute and take $\mathbf{u,w}$ out of the integral as they don't depend on $x$ and $y$

$$
\mathbf{w^T} \int_{\Omega} \mathbf{N}^T \mathbf{N} d\Omega \frac{\partial \mathbf{u}}{\partial t} + \mathbf{w^T} \int_{\Omega} \nu \mathbf{B}^T \mathbf{B} d\Omega \mathbf{u} =  \mathbf{w^T} \int_{\Gamma \mathbf{N}}  \mathbf{N}^T h d\Gamma_\mathbf{N} + \mathbf{w}^T \int_{\Omega} \mathbf{N}^T q d\Omega
$$

6. Eliminate $\mathbf{w}^T$

$$ \int_{\Omega} \mathbf{N}^T \mathbf{N}  d\Omega  \frac{\partial \mathbf{u}}{\partial t}+  \int_{\Omega} \nu \mathbf{B}^T \mathbf{B} d\Omega \mathbf{u} =   \int_{\Gamma \mathbf{N}}  \mathbf{N}^T h d\Gamma_\mathbf{N} + \int_{\Omega} \mathbf{N}^T q d\Omega
$$


7. write as a system of equations:

$$ 
\mathbf{M} \frac{\partial \mathbf{u}}{\partial t} + \mathbf{K u} = \mathbf{q}$$ 

$$ \mathbf{M \dot{u}} + \mathbf{K u} = \mathbf{q}$$


**Question 2: Problem definition**

Investigate the code and results to find out which problem is being solved. 

- Give a mathematical description of the problem in terms of governing equation and boundary conditions. Be as specific as possible, indicating the values that are used as input to the calculation. 

- In the final visualization contour lines are visible, connecting points that have the same temperature. As the solution evolves, these contour lines remain approximately perpendicular to the boundary. Which boundary condition does this observation relate to?

**solution**

1. Governing equations and boundary conditions:


The governing equation is:

$$
\frac{\partial u}{\partial t} = \nu \Delta u + q,
$$

The input values are:

- $\nu = 1$,
- $q = 15$,

Boundary conditions:

Dirichlet boundary conditions: $u=10$ on the bottom right edge
$$
u(x, y, t) = 10, \quad \text{on } \Gamma_D,
$$

Homogeneous Neumann boundary conditions on remainder of the bondary
$$
\nabla u \cdot \mathbf{n} = 0, \quad \text{on } \Gamma_N,
$$

Initial conditions:
$$
u(x, y, 0) = 30, \quad \in \Omega.
$$

2. Which boundary condition does this observation relate to?

- The homogeneous Neumman boundary condition

**Question 3: Integration scheme**

- In the `get_element_M` function, how many integration points are used and where in the triangles are they positioned? 
    
- In the `get_element_K` a simpler implementation is used. What is the essential difference between $\mathbf{K}_e$ and $\mathbf{M}_e$ that is the reason why this simpler implementation is valid for $\mathbf{K}_e$? (The subscript $_e$ is used to indicate the contribution to the matrix from a single element, or the *element matrix*). 




**solution**

1. 
- 3 integration points
- location at the midpoints of the triangle edges
in the code:
integration_locations = [(n1 + n2) / 2, (n2 + n3) / 2, (n3 + n1) / 2]

2. 

$$ \mathbf{M} = \int_{\Omega} \mathbf{N}^T \mathbf{N} \,d \Omega$$

$$ \mathbf{K} = \int_{\Omega} \mathbf{B}^T \nu \mathbf{B} \,d \Omega$$


$$
N_i = a_i + b_i x + c_i y, \quad i \in [1, 3],
$$

where $a_i, b_i, c_i$ are constants determined by the geometry of the triangle. The shape functions $N_i$ vary linearly across the element because they depend on $x$ and $y$.


Taking the derivative of $N_i$ with respect to $x$ and $y$ gives:

$$
\frac{\partial N_i}{\partial x} = b_i, \quad \frac{\partial N_i}{\partial y} = c_i.
$$

The derivatives $b_i$ and $c_i$ are constants because $N_i$ is linear, and the derivative removes the dependence on $x$ and $y$.

The $\mathbf{B}$-matrix is therefore constant within a single element and does not vary with $x$ or $y$. This simplifies the computation of the stiffness matrix $\mathbf{K}$ because $\mathbf{B}^T \nu \mathbf{B}$ remains constant and only needs to be multiplied by the area of the triangle.


**Question 4: Shape functions**
Investigate the shape functions for the element with index 10 in the mesh. Use the `get_shape_functions_T3` function defined in the notebook to find expressions for the shape functions in that element and check that they satisfy the shape function properties. 

1. What are the coordinates of the nodes of element 10? 

2. What are the shape functions of the element? 

3. Assert that the shape functions satisfy the partition of unity property:

$$
\sum_i N_i(\mathbf{x}) = 1
$$

4. Assert for one of the shape functions that it satisfies the Kronecker delta property

$$
N_i(\mathbf{x}_j) = \begin{cases}
  1, & i = j \\
  0, & i\neq j
\end{cases}
$$

**solution**

*1. Find coordinates of element 10*

- `nodes, connectivity = readGmsh('bigM.msh')` 

`nodes` is an array of the coordinates of the nodes. 

`connectivity` is an array of the nodes belonging to each element.


Node indices of element 10: [ 55,  56, 265]

Coordinates of the nodes of element 10: 
| $N_i$ | x | y | z |
|------------|--------------|--------------|--------------|
| 1         | 0.1          | 1.0          | 0.0          |
| 2          | 0.05         | 1.0          | 0.0          |
| 3          | 0.07628505   | 0.94825032   | 0.0          |

*2. shape functions*

 The `get_shape_functions_T3` outputs the coefficients for each shape function in the form:

 $$
 N_i = a_i +b_i x+ c_iy
 $$

 These coeffients are stored in an 3x3 mtrix coeffs
 These coeffients define the shape functions

For the 3 nodes of element 10 we get the following coeffients:

| Node  | $a_i$            | $b_i$            | $c_i$            |
|------------|--------------|--------------|--------------|
| 1          | -11.15853885 | 20.0         | 10.15853885  |
| 2          | -7.16525352  | -20.0        | 9.16525352   |
| 3          | 19.32379237  | 0.0          | -19.32379237 |




*3. Assert that the shape functions satisfy the partition of unity property*:

$$
\sum_{i=1}^3 N_i(\mathbf{x}) = 1
$$

   - Each shape function is of the form:
     $$
     N_i(x, y) = a_i + b_i x + c_i y, \quad i \in [1, 3],
     $$
     where $a_i, b_i, c_i$ are the coefficients computed using the `get_shape_functions_T3` function.

 Choose a Point Inside the Element:
   - for example centroid $\frac{N_1 +N_2+N_3}{3}$

Evaluate the Shape Functions at the this Point.
   - For each shape function $N_i$, compute its value at the test point

 Compute the Sum of Shape Functions:
   - Add the values of all shape functions at the test point:
 $$\text{Sum} = \sum_{i=1}^3 N_i(\mathbf{x}).$$

` sum(coeff[0] + coeff[1] * centroid[0] + coeff[2] * centroid[1] for coeff in coeffs)`


 Assert the Partition of Unity:
   - Check if the sum equals $1$:
$$
\text{Assert: } \sum_{i=1}^3 N_i(\mathbf{x}) = 1.
$$


```python
# Get the coordinates of the nodes for element 10
element_nodes = nodes[connectivity[9]]

# Compute the shape function coefficients for element 10
coeffs = get_shape_functions_T3(element_nodes[0], element_nodes[1], element_nodes[2])

# Compute the centroid of the element
centroid = element_nodes.mean(axis=0)

# Evaluate the sum of the shape functions at the centroid
partition_of_unity = sum(
    coeff[0] + coeff[1] * centroid[0] + coeff[2] * centroid[1] for coeff in coeffs
)

# Print the result
print(f"Sum of shape functions at the centroid: {partition_of_unity}")
assert abs(partition_of_unity-1)< 1e-5,, "Partition of unity property not satisfied!"
```

*4. Assert for one of the shape functions that it satisfies the Kronecker delta property*

For each shape function $N_i(x, y,z)$, calculate its value at the coordinates of each node $(x_j, y_j, z_j)$ then check if the shapefuction = 1

```python
# Extract coordinates of the nodes for element 10
element_coords = nodes[connectivity[9]]

# Compute shape function coefficients for element 10
coeffs = get_shape_functions_T3(element_coords[0], element_coords[1], element_coords[2])

# Verify Kronecker delta property
for i in range(3):  
    for j in range(3):  
        x, y, z = element_coords[j]  
        N_i_at_xj = coeffs[i][0] + coeffs[i][1] * x + coefffs[i][2] * y  
        expected = 1.0 if i == j else 0.0  
        print(f"N_{i+1}(x_{j+1}) = {N_i_at_xj}, Expected = {expected}")
        assert np.isclose(N_i_at_xj, expected), f"Kronecker delta property failed for N_{i+1} at node {j+1}"
        
```



## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in yout Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
