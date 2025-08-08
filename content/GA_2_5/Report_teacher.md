# GA 2.5 Report: Optimization

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.5. Due: December 13, 2024.*


## Questions

### Solving NDP problem with MILP

1.	**Run the model. Has the optimal solution been found? (yes/no)**

They need to check the Gap. If there is a Gap they cannot prove the optimal solution. If Gap is zero, yes they found the optimal solution.


2.	**If there is a gap, what can you tell about the value of the best solution that you are still theoretically able to obtain?**

If there is Gap 0, that’s it.
If the gap exists, they have to return the best bound form the results or simply calculate it: Objective * (1-Gap)


3.	**Consider now that the network operator does not want to surpass the capacity in the links. How do you write that constraint mathematically? Reminding you that in the current model it is possible to go beyond the capacity.**

They need to impose the capacity on each link.


4.	**Run the model with the new constraint. What changed in the network as a result of imposing the constraint?**

```
c_new = model.addConstrs(link_flow[i, j] <= cap_normal[i,j] + ((cap_extend[i,j]-cap_normal[i,j] ) * link_selected[i,j])  for (i, j) in links)
```

Note that we do: `cap_extend[i,j]-cap_normal[i,j]` because the cap extend has the normal capacity as well. There may be alternatives to do this.

Changes in the network is that now the links never supass their capacity. The flows on the road network change. This is enough. They can show that with the maps.

### Solving NDP problem with GA

5.**What very important solution performance indicator did you lose by using the GA metaheuristic in comparison with MILP?**

It’s the gap. With the genetic you never know if you are far or close if you don’t have any benchmark from the math programing. A good indicator that you are approaxing a better solution is the convergence in the curve.


6. **Change the parameters of the algorithm and see if it changes the performance.**

Change the population size and also the crossover operator (now you have `HalfUniformCrossover()`, change it to `PointCrossover()`) to see if it changes the performance.

No idea what they will get 😊 for sure that the population size changes things. Startign with lots of “individuals” means that the evaluation of the objective function needs to be done more times for each generation. But it also has more diversity which could help in the convergence. 


7. Is the GA a better algorithm to solve this problem than the math program?

For the duration of 300 seconds depends on the configuration of the algorithm. They should simply compare the objective function of both since the maximum time is the same. From our tests it seems that the linear programing was better.


## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a> &copy; 2024 TU Delft. <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. doi: <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515">10.5281/zenodo.16782515</a>.
