# GA 2.5 Report: Optimization

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.5. Due: December 13, 2024.*


## Questions

### Solving NDP problem with MILP

1. Run the model. Has the optimal solution been found? (yes/no)

2. If there is a gap, what can you tell about the value of the best solution that you are still theoretically able to obtain? (You may expand the truncated output from the optimization output to view the column names).

3. Consider now that the network operator does not want to surpass the capacity in the links. How do you write that constraint mathematically? Reminding you that in the current model it is possible to go beyond the capacity. 

4. Run the model with this new constraint. What changed in the network as a result of imposing the constraint?

### Solving NDP problem with GA

5. What very important solution performance indicator did you lose by using the GA metaheuristic in comparison with MILP?

6. Change the parameters of the algorithm and see if it changes the performance. (The population size and also the crossover operator - now you have `HalfUniformCrossover()`, change it to `PointCrossover()`) Comment on if it changes the performance.

7. Is the GA a better algorithm to solve this problem compared to the MILP?



## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
