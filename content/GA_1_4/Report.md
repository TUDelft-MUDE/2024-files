# Report for Group Assignment 1.4

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.4, Friday, Sep 27, 2024.*

Remember there are "Tips for Writing the Report" in the [GA 1.3 README](https://mude.citg.tudelft.nl/2024/files/GA_1_3/README.html).

## Questions

**Question 1**

Give a short explanation about the model in the assignment. How is it different from the last week's assignment? What is the reason to try a new model?

_We copy the LaTeX equation for the model here, in case you would like to refer to it in your Report._

$$
d = d_0 + R \ (1-\exp\left(\frac{-t}{a}\right)) + k \ \textrm{GW},
$$

_Write your answer here._

**Question 2**

Report the Gauss-Newton interation convergence criteria, describe how quickly convergence was realized. Also comment on the quality of the estimation with InSAR and GNSS. Give your interpretation for any discrepancy between observations and the model.

Include a Markdown table that summarizes the estimated parameters and their precision for both models.

_Write your answer here._

**Question 3**

Give an explanation of test statistic used to test which model (linear, non-linear) fits data better. What is the null hypothesis $H_0$ and alternative hypothesis $H_a$ in this test? What is the distribution of test statistic? Compare the test outcomes with InSAR and GNSS and interpret the results.

_Write your answer here._

**Question 4**

In order to get a better fit to the data (smaller residuals) for this case study, consider the following strategies and determine whether or not each one could help. Use the Markdown table provided to state "Yes" or "No", then elaborate on your answer with one or two sentences each in the last column.

1. better observations?
2. a more complicated geophysical model?
3. better initial values?
4. more observations?
5. combining observations?

In your answer, keep in mind that data acquisition and processing comes with a price. Note that in a real situation you would not look at a time series of only one point. For Sentinel-1 data you may have to pay, collecting GNSS data at different locations also costs money. How will you monitor the deformation if you have both GNSS and InSAR data at your disposal?

_Fill in your answer in this table:_

| No. | Answer | Elaboration |
| :---: | :---: | :----- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.