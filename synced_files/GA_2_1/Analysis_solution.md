---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: MUDE
    language: python
    name: python3
---

<!-- #region id="9adbf457-797f-45b7-8f8b-0e46e0e2f5ff" -->
# GA 2.1: FVM with an Unstructured  Mesh (diffusion)

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

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.1. For: 15 November, 2024.*
<!-- #endregion -->

# Overview

This assignment considers an interesting 2D shape: the "U" of MUDE! We will use the diffusion equation to compute the distribution of temperature in the U subject to specific boundary conditions and initial conditions.

This assignment contains three parts:

1. expressing the diffusion equation for triangle volumes and formulating algebraic equations with the finite volume method,
2. implementing and solving this method in an unstructured orthogonal mesh,
3. analyzing potential downsides to the discretized approach used if applied to non-orthogonal meshes, and considering how they might be corrected.

Remember, even though the problem is 2D, the numerical scheme will be fromulated by treating _volumes_ (as opposed to dimensions $x$ and $y$ directly).


## Part 1: Using Just Your Hands

The diffusion equation expressed in its reduced form is:

$$
\frac{\partial \phi}{\partial t} = \nabla \cdot D \nabla \phi
$$

Note that for triangle shape volumes, the fluxes are not necessarily directed in the $x$ direction. Rather, in the normal direction to the surfaces of each volume, as the flux direction is given by the $\phi$ gradients and the information propagates in all directions.

Over the next tasks, you will write step by step its derivation.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.1</b>

Integrate the PDE over triangle volumes of interest. 

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

$$
\int_{V} \frac{\partial\phi}{\partial t} dV
= \int_{V} \nabla \cdot (D \nabla \phi)dV 
$$                                                                                         

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.2</b>

Transform the corresponding volume integral into a surface integral using Gauss's theorem.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

$$
\int_{V} \frac{\partial\phi}{\partial t} dV
= \int_{S} \mathbf{n} \cdot (D \nabla \phi)dS 
$$                                                                                         

$$
\int_{V} \frac{\partial\phi}{\partial t} dV
= \int_{S} D \frac{\partial \phi}{\partial n}dS 
$$

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.3</b>

Approximate the integrals using numerical integration, using the midpoint rule. Write explicitly the three surface fluxes.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

$$
\frac{\partial\phi}{\partial t} \Delta V
= D\frac{\partial \phi}{\partial n} \Delta S \bigg\rvert_{S_1}
+ D\frac{\partial \phi}{\partial n} \Delta S \bigg\rvert_{S_2}
+ D\frac{\partial \phi}{\partial n} \Delta S \bigg\rvert_{S_3}
$$

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.4</b>

Divide both sides of the equation by $\Delta V$. The "depth" of the volume is common with the surface flux area. Write down the resulting equation.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

$$
\frac{\partial\phi}{\partial t} 
= D\frac{\partial \phi}{\partial n} \frac{\Delta L}{\Delta A} \bigg\rvert_{S_1}
+ D\frac{\partial \phi}{\partial n} \frac{\Delta L}{\Delta A} \bigg\rvert_{S_2}
+ D\frac{\partial \phi}{\partial n} \frac{\Delta L}{\Delta A} \bigg\rvert_{S_3}
$$

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.5</b>

Approximate the derivatives! Use central differences in space and Forward Euler in time. The subindices can be defined by arbitrarily naming the volume of interest and the adjacent volumes.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

Here we use $nv$ to denote the three neighbor volumes:

$$
\phi^{n+1}_i = \phi^{n}_i 
+ D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{nv1}-\phi^n_i}{\Delta d_c} \right)
+ D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{nv2}-\phi^n_i}{\Delta d_c} \right)
+ D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{nv3}-\phi^n_i}{\Delta d_c} \right)
$$

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 1.5</b>

The equation you derived above describes volumes with <b>three</b> interior sides. For a volume with <b>one</b> side being an exterior one, modify the equation above to implement the following Neumann condition (which replaces the discretized gradient term):

$$
\frac{\partial \phi }{\partial n} = 10 [C^o/m]
$$

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

$$
\phi^{n+1}_i = \phi^{n}_i 
+ D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{v1}-\phi^n_i}{\Delta d_c} \right)
+ D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{v2}-\phi^n_i}{\Delta d_c} \right)
+ D\frac{\Delta t \Delta L}{\Delta A} \cdot 10
$$

</p>
</div>


## Part 2: Implementation!

Below, the coordinates of the triangle vertices that cover the domain of interest are defined. The boundary conditions are also specified and incorporated in the resulting object `mesh` once the class `Mesh` is instantiated. The class `Mesh` is defined in `utilities.py` and has an identical structure and similar functions as provided in your PA for this week, which define key characteristics of the mesh and volumes, as well as providing useful plotting and solving methods.

Note: the `Mesh` class and it's usage is illustrated extensively in the companion notebook `mesh_tips.ipynb`.

```python
import matplotlib.pyplot as plt
import numpy as np
from utilities_solution import *
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.0</b>

Execute the cell below and study the resulting geometry until you are comfortable recognizing the problem that will be solved. In particular, check that you can identify which boundary conditions are applied, and where, before moving on.

If you are not sure how to interpret the code or figures, refer to <code>mesh_tips.ipynb</code>.

</p>
</div>

```python
length = 10

coordinates = [[  0.,    0.   ],
               [ 10.,    0.   ],
               [  5.,   -8.660],
               [ 15.,   -8.660],
               [ 10.,  -17.320],
               [ 20.,  -17.320],
               [ 25.,   -8.660],
               [ 30.,  -17.320],
               [ 35.,   -8.660],
               [ 30.,    0.   ],
               [ 40.,    0.   ]]

coordinates = np.array(coordinates)

boundaries = [[['Neumann', +1], [0, 1]],
              [['Neumann',  0], [1, 3, 6, 9]],
              [['Neumann', -1], [9, 10]],
              [['Neumann',  0], [10, 7, 4, 0]]]

mesh = Mesh(coordinates, length, boundaries)
mesh.plot_triangles();
mesh.plot_boundaries();
```

Before we continue to finding the solution, let's explore the relationship of today's mesh with the bars and kapsalon shops we considered in the PA for this week.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.1</b>

Write down the similarities between the housing-bar-kapsalon problem (PA 2.1) and the FVM representation of the diffusion equation here. Specifically, the locations and quantities associated with each of the three things.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

- **Houses**: these are the vertices of the volumes; we don't actually compute anything at these locations, they are only used to define the triangle/mesh geometry.
- **Bars**: these are the centroids of the triangles and are the locations where the unknowns $\phi(x, y, t)$ are computed. Note however, we compute something more like $\phi(i,t)$ where $i$ refers to the triangle number.
- **Kapsalon shops**: not only are these places where we can get tasty food, but also they are the centers of the side of each triangle (but only when the triangles are equal size and equilateral: orthogonal). This is the midpoint of the surface integral, an important observation that you probably would not have recognized in this task, but is considered in the third part of this notebook.

</p>
</div>


Write your answer here.


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.2</b>

Suppose **just for this single task** that the FDM would be used, sketch the potential grid points and draw the boundaries that would represent the same domain in $x$ and $y$ as defined by <code>coordinates</code> in the previous task. 

This is to illustrate one of the contrasts between FDM and FVM.

</p>
</div>


Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

Sketch several sets of grid points to illustrate the shape of the U; you will need several. Depending on how "wide" the U is (number of grid points), the geometry has trouble capturing the angles. Especially the boundary is not a smooth line, as "steps" are introduced to capture the boundary.

It is also possible to reshape the U into a rectangular shape; this would be much easier to solve with a simple FDM scheme, however, the problem is very different and may not be representative of the situation in reality that must be modelled.

Main takeaway is to see that FDM has challenges modelling non-rectangular geometry.

</p>
</div>


Now we will **continue with code implementation** by adding some information to the object that defines our problem, `mesh`. Remember to refer to `mesh_tips.ipynb` if you are not sure how to use it!


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.3</b>

First, use the method <code>set_initial_temperature()</code> to define the initial conditions for the volumes. The initial temperature should be 20 degrees everywhere except at the volume in the middle, where the temperature is 40 degrees.  

</p>
</div>

```python
# mesh.plot_triangles(YOUR_CODE_HERE) # useful for identifying the triangle id
# mesh.set_initial_conditions(YOUR_CODE_HERE)
# mesh.plot_triangles(fill_color='initial_conditions');

# SOLUTION
mesh.plot_triangles(triangle_id=4) # useful for identifying the triangle id
mesh.set_initial_conditions(default=20,
                            special_triangles=[[4, 40]])
mesh.plot_triangles(fill_color='initial_conditions');
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.4</b>

**Solve**.

Using the <code>solve</code> method, solve the problem for conditions where $t_{final}=20$, $N_t=100$ and $D=50$.

Use the resulting plot to see if you reach the solution you expect. Remember to consider the boundary conditions that were defined for you as well, not only the initial conditions.

</p>
</div>

```python
# mesh.solve(YOUR_CODE_HERE, YOUR_CODE_HERE, YOUR_CODE_HERE)
# mesh.plot_triangles(fill_color='unknowns',
#                     show_labels=False);

# SOLUTION
mesh.solve(20, 100, 50)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.5</b>

**What's going on?!**

In the previous task, you should have realized that the solution is _not_ what we expect.

It turns out we screwed up! In the massive file <code>utilities.py</code>, there is </b>one</b> line of code that needs to be fixed in order to solve this problem. Can you find it and fix it?

The first group to find the solution will win a prize!

<em>Hint: it's not in the plotting method, and is related to something you derived in the tasks above.</em>

<b>Note:</b> you should expect to spend some time reading the code (at least one very specific part of the code); not only will this help you fix the problem, but it will be very useful for answering the first few questions in the Report.

</p>
</div>



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

The process of reading the code and solving this problem is directly related to the first few questions of the Report.

The first thing to recognize is that the "solution" using the incorrect code looks exactly like the initial conditions. So this means that we are probably calculating something incorrectly in the solution algorithm, which is contained in the method <code>solve</code>.

It is challenging to read through the code in this method because there are a lot of loops and conditional statements. However, one should be able to recognize the time and space discretization steps, as well as the conditional statements which identify neighbor triangles to perform the surface flux calculations and apply the boundary conditions, when necessary. Note in particular that the "space" integration actually is a loop over all triangles (the volumes), and for each triangle there is a loop over each face.

Note that Python method <code>enumerate</code> returns the index of a list/array, as well as the item at that index; this is useful to more easily define values in the loop. 

Once the structure of the code is recognized, it should be straightforward to check that the algebraic equations defined above are implemented correctly (note variables like <code>unknowns</code>, <code>phi</code>, <code>constant</code> and <code>flux</code>).

At the end of this method, we can see how the Euler scheme is applied:

<pre>
<code>
unknowns[time_step+1, triangle_id] = unknowns[time_step, triangle_id]
</code>
</pre>

After close inspection, it should be obvious that this simply takes the solution from the previous time step for the triangle with given id, $\phi_{i}^{n}$ and assigns the value to the next time step, $\phi_{i}^{n+1}$. Because this happens for all time steps and all triangles, this results in a "solution" that simply repeats the initial conditions at every time step.

To fix the issue, we need to implement the algebraic expression

$$
\phi^{n+1}_i = \phi^{n}_i 
+ \sum_{nv} D\frac{\Delta t \Delta L}{\Delta A} \left( \frac{\phi^n_{nv}-\phi^n_i}{\Delta d_c} \right)
$$

which we can do as follows:

<pre>
<code>
unknowns[time_step+1, triangle_id] = phi + np.sum(flux)
</code>
</pre>

Note that the expressions provided here are the same as those requested for Question 1 in the report.

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.6</b>

**Refine**

Refine the mesh and create a plot to check visually how the geometry has changed.

</p>
</div>

```python
# YOUR_CODE_HERE
# mesh.plot_triangles();

# SOLUTION
mesh.refine_mesh();
mesh.plot_triangles();
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.7</b>

**Solve**.

Solve the problem for the new mesh using the same time and diffusion parameters as before.

<em>Hint: copy/paste the same code from when you solved it the first time!</em>

</p>
</div>

```python
mesh.solve(20, 100, 50)
mesh.plot_triangles(fill_color='unknowns',
                    show_labels=False);
```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 2.8</b>

**Broken again?!**

The solution for the refined mesh did not work---this time it is not a problem with the code, but another issue. See if you can make some adjustments and fix the solution!

Note that you will be asked about stability in the Report, so you might as well calculate this and record the values now.

<em>Tip: even though the mesh has smaller volumes (triangles), the solution should look similar to the previous (unrefined) mesh.</em>

</p>
</div>

<!-- #region -->

<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

First, note that the unstable solution is recognized by the large values of the solution indicated by the color bar in the figure: order 10 to the 87th power! Clearly the solution is on its way to infinity (and beyond?!).

Instead of solving with these parameters:

<pre>
<code>
mesh.solve(20, 100, 50)
</code>
</pre>

Increasing the number of time steps by a factor of 10 works:

<pre>
<code>
mesh.solve(20, 100, 50)
</code>
</pre>

This reduces the time step size $\Delta t$ by a factor of 10, and the reason it results in a stable solution can be explained using the stability criteria...


</p>
</div>
<!-- #endregion -->

```python
side_length= 5.0
t_final=20
D=50
Nt_vec=np.arange(200, 650, 50, dtype=int)
Stability = []
max_temp = []
for Nt in Nt_vec:
    _,stability,temp=mesh.solve(20, Nt=Nt, D=D)
    Stability.append(stability)
    max_temp.append(temp.max())
    print("==================================")
plt.figure(figsize=(10, 5))
plt.plot(Nt_vec, Stability, 'o-')
for x, y, temp in zip(Nt_vec, Stability, max_temp):
    plt.text(x, y,f'$T=${temp:.1e} C', fontsize=6, ha='center', va='top')
plt.xlabel('Number of time steps')
plt.ylabel(r'D*$\Delta t$ Surface/Volume/centroid_distance)')
plt.title('Stability criteria vs Number of time steps ')
plt.axhline(0.5, color='r', linestyle='--')
plt.grid()
plt.show()
```

## Part 3: Evaluating the implementation for non-equilateral triangles 

Computations in meshes with non-equilateral triangles have added error sources that would need to be corrected to have an accurate solution. In this section you will analyse and reflect on the potential downsides of your implementation for non-equilateral triangle volumes by looking only at the fluxes between two volumes. The file `utilities.py` and rest of the code in this notebook is completely irrelevant to this Part (except the `numpy` and `matplotlib` import).


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.1:</b>

The vertex coordinates of the first triangle to be analyzed are: [0,0] , [0,1] , [1,0]. The second triangle share the first two vertices and have coordinates [0,0] , [0,1] , [-1,0]. 

Plot the vertices and the triangles edges.

</p>
</div>

```python
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 0 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

plt.scatter(x_v1, y_v1)
plt.scatter(x_v2, y_v2)

def plotting_volumes(coordinates):
    for i in range(3):
        start = coordinates[i]
        end = coordinates[(i+1)%3]
        plt.plot([start[0], end[0]], [start[1], end[1]],
            color='black', linestyle='--', linewidth=2)

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    


```

<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.2:</b>

Plot the centroid of each triangle, its connecting line and the middle distance. For the scheme you implemented in the previous task, write down what this implies for the **numerical integration approximation of the flux at the surface**.

Specifically, consider the location where the flux is calculated, as implied by the numerical scheme, as well as the integration that is implied by the analytic surface integral over the surface of the shared triangle side.
</p>
</div>

```python
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 0 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    
```

Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

When you approximate the integral with a numerical one, it is assumed that one value of the flux is representative of the fluxes at the entire surface (recall numerical integration of Q1). For equilateral triangles, using the centroids to approximate the gradient implies that we are using a midpoint rule that is second order accurate. However, as the figure shows, for this situation the evaluation is _not_ done at the middle point; rather, it is representative of another location on the surface; thus the accuracy is not second order accurate.   

</p>
</div>


<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.3:</b>

Now lets analyze a different situation and change the coordinates of the second triangle to [0,0] , [0,1] , [-1,1].

Plot the centroid of each triangle, its connecting line and the middle distance. For the numerical scheme you implemented, write the implication for the **numerical derivative approximation of the flux at the surface**.

Specifically, consider the _direction_ in which the flux is calculated, as implied by the numerical scheme, as well as the quantity demanded by the analytic expression (i.e., the surface integral).

</p>
</div>

```python
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1 ]
y_v2 = [ 0 , 1 , 1 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    
```

Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

Now the evaluation of the flux is done at the midpoint of the surface (in contrast to the previous task) but its direction evaluation is not normal to the surface!

Thus, the gradient and flux are computed incorrectly if we applied the scheme implemented in our code.

</p>
</div>

<!-- #region -->
<div style="background-color:#AABAB2; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Task 3.4:</b>

Now lets analyze one last situation and change the coordinates of the second triangle to [0,0] , [0,1] , [-1.5,1].

Plot the centroid of each triangle, its connecting line and the middle distance. Write down what this implies for the **numerical derivative approximation of the flux normal to the surface**. 


</p>
</div>
<!-- #endregion -->

```python
x_v1 = [ 0 , 0 , 1 ]
y_v1 = [ 0 , 1 , 0 ]
coordinates_v1 = np.array([x_v1,y_v1]).T

x_v2 = [ 0 , 0 , -1.5 ]
y_v2 = [ 0 , 1 , 1 ]
coordinates_v2 = np.array([x_v2,y_v2]).T

centroid_1 = [np.sum(x_v1)/3,np.sum(y_v1)/3]
centroid_2 = [np.sum(x_v2)/3,np.sum(y_v2)/3]

plt.scatter(centroid_1[0], centroid_1[1])
plt.scatter(centroid_2[0], centroid_2[1])
plt.scatter( (centroid_1[0]+centroid_2[0])/2, (centroid_1[1]+centroid_2[1])/2)
plt.plot( [centroid_1[0] , centroid_2[0]] , [centroid_1[1],centroid_2[1]] )

plotting_volumes(coordinates_v1)    
plotting_volumes(coordinates_v2)    
```

Write your answer here.



<div style="background-color:#FAE99E; color: black; width: 95%; vertical-align: middle; padding:15px; margin: 10px; border-radius: 10px">
<p>
<b>Solution</b>

Now the evaluation of the flux (as implemented in task 2) is not done at the midpoint of the surface, not even at the surface, and its direction is not normal to the surface! 

Thus, there are two sources of error present, when it comes to the numerical scheme in the code of this assignment, applied to the volumes illustrated here.

</p>
</div>


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
