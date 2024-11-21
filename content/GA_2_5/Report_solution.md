# Project 9 Report: Optimization

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/), Week 2.6, Friday, Dec. 15, 2023.*

**SOLUTION**

_The solutions are provided below (and also in the notebooks). There are also additional explanations that try to explain what happened during the in-class session when some of the tasks could not be completed._

> Solutions that were expected to be in your report are formatted like this, albeit in a shorter version, and without figures.

_General note about the problem (MILP and GA)_: this assignment asks you to consider upgrading a road network with 76 links. The optimization algorithms are used to find the best solution _given a specific number of links to consider._ Thus, for a specific number of links, you would solve a "new" optimization problem. In reality, the number of links that can be upgraded is closely linked to your budget, and the budget itself might be adjusted depending on the results of several optimization analyses for different upgrade options. In other words, if the benefit (travel time reduction, based on optimal solution) is not linearly related to the costs (number of links considered), the solution is non-trivial. This is generally how real life works...in fact, the concept of excpected value (expected cost, expected benefit) could be used in this case to compare alternatives (we will learn how in Week 2.8). Unfortunately, the instructions in this assignment asked you to consider cases 1 or 76 links were considered, which _are_ trivial solutions:

- if you can only consider 1 link, you should choose that which gives the greatest decrease in travel time
- if you can consider upgrading all links, and all links reduce travel time, you should upgrade all of them!

The problem gets much more interesting if you must choose which specific combination of links to upgrade, because your budget allows for something between 1 and 76. Because of aspects such as the geometric connections between links and nodes, travel time changes in a complex way when only _part_ of the network is improved (this is why the solution is non-trivial). For example, it is obvious that you should choose links that are somehow connected to each other in order to improve the travel time in the entire network. If you only improve isolated parts of the network, you are simply moving around points of congestion (e.g., you would drive fast on an improved road, then hit terrible traffic when forced to drive slow on an unimproved stretch of road). This is the type of comparison the optimization algorithm is making as it searches through the possible solution space.

The time it takes to find an optimization solution is generally related to the size of the possible solution space. It can be shown with combinatorial mathematics that there are more possible solutions when 40 links are considered, compared to 20. However, during the in-class session you should have seen that it was more difficult to find a solution for 20 links. This is because the physics of the problem are also playing a role. It turns out that improving 40 links is enough to improve a large-enough part of the network to improve travel time for nearly all of the routes, and it is "easy" to find an optimal solution. However, with 20 links, you are only able to focus on smaller parts of the network; think of them as "corridors" through the total network. Finding the solution that chooses which of these corridors is the best (i.e., optimal) takes a longer time.

## Questions

### Solving NDP problem with MILP

Solution explanations here are based on results that have been computed with a budget of 40 road links (`extension_max_no` = 40) and I time limit of 100 seconds (the notebook given to students had 300), except where otherwise noted. Notice that all these results depend on the speed of your machine. You may have to extend the computation time to 300 seconds if you haven’t found a solution yet. Be aware that you can’t run a model only from the point onward of a change that you have introduced in a constraint. You need to run the whole model again. Also important to note when you run the optimization procedure again (the cell where this is written: `model.optimize()`) without restarting everything, Gurobi continues optimizing from the point where it stopped, meaning with the previous solutions tested.  Gurobi adds the sentence: `continuing optimization!`. But even if you restart Gurobi completely you may start at a more promising point and it is possible it will give you a better solution in the same 100 seconds. That’s because the process is not deterministic, it depends on the strategy of exploring the branch and bound tree as explained in the lectures. Check section “5.7. Integer problems and solving with Branch-and-Bound” to learn more.

1. **Has the optimal solution been found? (yes/no)**

> No (because gap is not zero).

No, the optimal solution has not been found because as you see in the table the last integer solution that has been found still has a GAP (65.1%). This GAP tells you the maximum error that you may be committing by choosing this solution, it does not mean actually that the last solution is not the optimal. It just tells you that you can't prove it yet because you have not explored all the nodes in the branch and bound tree. Check section “5.7. Integer problems and solving with Branch-and-Bound” to learn more.

2. **If there is a gap, what can you tell about the value of the best solution you are still theoretically able to obtain?**

> It's easy to check by just returning the best bound of the last best solution!

It's easy to check by just returning the best bound of the last best solution! Indeed you can’t reject the possibility that there isn’t a better solution that you haven’t found yet after you stopped the algorithm with a value of the objective function that is the same as the best bound. In this case, the bound is 3832504.14. Theoretically, it’s still possible to get an optimal solution with that objective function value.

3. **Consider now that the network operator does not want to surpass the capacity in every link. How do you write that constraint? Reminding you that in the current model it is possible to go beyond the capacity which makes traffic even slower (even higher travel time).**

> Constrain the vehicles to the capacity of the road (**this is the solution provided on the screen during the in-class session; see explanation for more details**):
> 
> ```python
> c_new = model.addConstrs(link_flow[i, j] <= cap_normal[i,j] + (cap_extend[i,j] * link_selected[i,j])  for (i, j) in links)
> ```

The answer to this question would have been: 
```
c_new2 = model.addConstrs(link_flow[i, j] <= cap_normal[i,j] + ((cap_extend[i,j]-cap_normal[i,j] ) * link_selected[i,j])  for (i, j) in links)
```

This constraint is a hard limit on the capacity of a road. You do not allow the flow to go over that physical limit, otherwise the problem is unfeasible. Notice that in this problem `cap_extend` has been defined as the final capacity after an extension, not just the added capacity. That was what created a confusion on Friday in class. In the constraint above when a link has not been selected for expansion only the normal capacity will be considered because `link_selected[i,j]` will be zero. But if the link is an upgraded one then you will add capacity to the capacity normal. That amount is the difference between the original capacity and the final extended capacity. 

Unfortunately, such a constraint creates an unfeasible problem. It’s not possible to pass all the flow with the capacity you have, even with the extensions. That is interesting because it shows that indeed there are many links where the flow is over the capacity creating great delays on the roads. 

When students changed the constraint to what was tested by the teachers of the optimization week (wrong constraint) where the capacity was double counted, `cap_normal + cap_extend`, the problem should be feasible: 

```
c_new1 = model.addConstrs(link_flow[i, j] <= cap_normal[i, j]+(link_selected[i, j]*cap_extend[i, j]) for (i, j) in links)
```

However, an additional issue occurred, which is that many students changed the constraint and started running the model from there (i.e., keep executing notebook cells without re-starting the Gurobi model). That is not possible since you are just adding more equations to your previous unfeasible problem. **Always run the models from beginning to end when making a change to the problem** (i.e., re-assign the `model` or create a new one).

With this “wrong” constraint that double counts the capacity, the model runs well.

With the original model without the constraint, you have an OF of 1.0976e+07 with a gap of 64.4% (best bound 3792722.49),  with the new one that limits hard the capacity you have 1.1062e+07  with a gap of 22.4% (best bound 8583881.27). 

The travel time seems to increase if you look at the best bound which is much higher but you can’t guarantee anything. An increase in travel time would make sense since you are restraining the flows even more, so cars need to find alternative paths. 

_Note that you can have different results from the numbers above depending on how the solving process goes. The point of this exercise was to be able to build the constraint which most groups were indeed able to do._

4. **What do you observe in the solution when you run this new model for 20 links and expanded capacity of 1.5 (default increase)?**

> The problem is unfeasible!

The point here was to show that if you indeed limit the capacity too much (only 20 links at expanded capacity) the cars do not even fit the network capacity.

5. **What can you do to fix that?**

> Increasing the budget and/or the capacity extension.

Often you find that you have created an unfeasible problem and you just need to relax it a bit more to find a solution. 

### Solving NDP problem with GA

In this workbook, by default, you have 76 links as a budget, and that is a very special solution because it means that you can upgrade all the links!  By mistake, there were two attributes with the budget in the notebook, but it was the parameter called `Budget` that should have been changed. It is used by the genetic algorithm to know how many binary variables can have the value of one. The maximum computation time in the results reported next is 300 seconds. No other stopping criteria were imposed that’s why the model never stops until 300 seconds. 

1. **What very important solution performance indicator did you lose by using the GA metaheuristic in comparison with MILP?**

> No GAP is produced therefore if you did not run the math programing model you would not know how far or close you are from the optimal solution.

Of course, seeing an almost flat line in the search progress in the GA helps build confidence that you have found the optimal solution. Moreover, since the solution is so trivial (76 links can be upgraded) you are sure of your optimal solution. Why didn't the genetic algorithm find an optimal solution in the first generation in some cases? This has to do with the large solution space, and the fact that the algorithm is initialized randombly, so we cannot guarantee that it always fings the global minimum, let alone find it in the fastest possible way. Additional insight is provided in the explanations below.

2. **Try to obtain a better solution in the same computation time. What parameter is best to change to get that improvement? write your results in the box.**

> Heere a description of anything you tried, whether it worked or not, was acceptable, as long as you were able to describe why it may or may not have improved computation time in a clear way.

Population size was the easiest parameter to control. You can notice that increasing the population size in this problem with 76 links does not improve the speed of convergence. Actually, it makes it slower because you need to combine in each generation many of your population genes. The search procedure with the population size of 10 gives a solution which is obviously the 76 links and an objective function of 9.873735E+06. If the population size is changed to 50 then the convergence seems to be slower. But you still find the same optimal solution. Running the model with a population size of 1 makes no sense, you can’t generate a new generation if you don’t have enough parents. 

![Figure 1](./figs_solution/fig1.png)

With just two individuals it even converges faster:

![Figure 2](./figs_solution/fig2.png)

When increasing population size from 10 to 200, and evaluating 36 links, you probably saw that the solution did not converge; this was due to the 5 minute time limit. Increasing to 15 minutes improves the solution, but still may be not enough. Clearly population size has a big influence on solution convergence! Because the population is initialized randomly, you should expect to run this several times to make sure you do not miss a suitable solution. There are also other “knobs” you can turn to change the computation time and process, for example, the parent and offspring settings.

Choosing an initial population size is highly dependent on the details of a particular problem. We do know, however, that with 1000’s of variables like we have in this problem, an initial population size of 10 is far too low! You would need to run this for a very long time to find a solution. In this project our intention was not to get a “great” solution, but to introduce you to the key characteristics of a GA approach (such as population size, crossover, etc) and how it is different than MILP, especially when it comes to how the solutions are assessed differently (e.g., gap versus objective function convergence).

In the assignment we asked you to try different parameter settings. We also included various methods from the pymoo package (for example, the `pymoo.operators.crossover.XXXXXX` at the top of the notebook):

- In the code we only used the half uniform crossover, which requires no additional parameter settings (i.e., no keyword arguments in the pymoo method)
- `PointCrossover` was covered in the online textbook; this method requires you to specify the number of points that are used to cross-over, which implies that one keyword argument should be provided. Here is an example from the documentation that actually includes a few examples, which includes the keyword argument `n_points=4`; this indicates 4 crossover points (although note that the textbook only includes examples illustrating this for 2 points).
- Note on the page linked in the previous point that there are two methods which were not included in the notebooks we gave you: `SinglePointCrossover` and `TwoPointCrossover`. These methods require no additional keyword arguments and would have worked “out of the box;” they were also illustrated with examples in the online textbook. This was unfortunately not included in the original assignment description.

3. **Run the model with a small number of links (=1) and a big number of links (max=76), what differences do you see in the solutions and computation time compared with the previous model? and why?**

> The performance changes a lot if you are locating just one link or 76. 76 is like no decision must be made, all are upgraded. With 1 you need to select the best one which is also fast. The problem is with intermediate solutions, like choose 30 out of the 76: it becomes a harder combinatorial problem to solve.

_The notebook was initialized with the value for budget being 76; we also asked you to consider 1 and 76 links. Both of these were not the best way to evaluate the effectiveness of the GA (or the MILP, for that matter)._

The performance changes a lot if you are locating just one link or 76. 76 is a trivial decision (in fact it is like no decision must be made, all links are upgraded; this is why the original code in the notebook converged very quickly. With 1 you need to select the best one which is also fast, but still the algorithm has a harder task. The problem is with intermediate network sizes like choosing 40 out of the 76: it becomes a harder combinatorial problem to solve. Note the comment in the previous question about population size, where the algorithm took a long time to converge when 36 links were selected.


### Comparison

1. **Compare the GA approach with MILP approach. For the same computation time which one is faster?**

> The GA is faster.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
