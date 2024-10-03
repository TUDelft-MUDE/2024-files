# Report for Group Assignment 1.5

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.5, Friday, Oct 4, 2024.*

Remember there are "Tips for Writing the Report" in the [GA 1.3 README](https://mude.citg.tudelft.nl/2024/files/GA_1_3/README.html).

## Questions

## Numerical Derivatives

**Question 1**

Explain what the time derivative of the ice thickness represents and why it would be incorrect to compute it at once for the entire data set. 

_Write your answer here._

_The time derivative represents the growth rathe of the ice thickness. From a visual data inspection, it seems that the first measures of ice thickness occur at the beginning of the year. At a certain moment of the year measurements stop, which represents that the ice layer broke. At this moment, the time derivative does not have physical sense as the next measurement corresponds to a new/different ice layer._

**Question 2**

Summarize the number of ice thickness measurements, number of intervals and number of values calculated for each numerical derivative. Note any differences about the time value at which each derivative is computed. Then, explain why there are differences between each method. For each method, at which points are the time derivative missing? Why can't Central Differences be evaluated at the same time/location as the data points?

_Write your answer here._

_The time derivative is missing at the end of the data set using FE while using BE the information is missing at the first point. Central differences require that the distance between the evaluation point and those used to compute the derivative are equidistant, the measurements were not performed this way but CD can be evaluated between measurements (at the middle)._

**Question 3**

Insert the image of your results showing the measurements and the three numerical derivatives. What can you conclude of the accuracy of the growth rate estimation? In this case, what are the two reasons that make CD more (much more) accurate than FE/BE?

_Insert image here_

_Write your answer here._

_First, inherently CD is second order accurate while FE/BE are first order accurate. Second, defining CD between points reduces $\Delta x$ by half, making its evaluation even more accurate._

## Taylor Series Expansion

**Question 4**

Insert an image of your derivation of the first four derivatives of Task 2.1.

_Insert image here_

_See the solution in the notebook._

**Question 5**

Insert an image of your results corresponding to Task 2.5.

_Insert image here_

_See the solution in the notebook._

**Question 6**

How do the errors behave for the four TSE? Which one is more accurate near $x_0$? Farther away $x_0+5$? What is your opinion about using TSE for approximating harmonic functions (as the one in this exercise)?

_Write your answer here_

_Very close to $x_0$ every approximation seems accurate, as you increase the distance the larger orders retain a high accuracy. However, the farther you move, every solution diverge considerably. The higher orders show a faster acceleration. The approximation of harmonic functions using TSE is only reasonable if its done nearby the expansion point._

**Question 7**

Insert an image of your results corresponding to Task 3.1.

_Insert image here_

_See the solution of Task 3.1_

**Question 8**

Insert an image of your results corresponding to Task 3.3.

_Insert image here_

_See the solution of Task 3.3_

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.