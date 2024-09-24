# Report for Group Assignment 1.4

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.4, Friday, Sep 27, 2024.*

## Questions

**Question 1**

Give a short explanation about the model in the assignment. How is it different from the last week's assignment? What is the reason to try a new model?

_We copy the LaTeX equation for the model here, in case you would like to refer to it in your Report._

$$
d = d_0 + R \ (1-\exp\left(\frac{-t}{a}\right)) + k \ \textrm{GW},
$$

**Solution**

Linear model in the assignment 1.3 and new non-linear model in the assignment 1.4 are both constructed to fit displacement of the ground surface.
The data sets of ground surface observations are the same and include InSAR and GNSS data. InSAR has 61 observations and GNSS 730. The standard deviation of each InSAR and GNSS measurement is assumed to be 2 mm and 15 mm, respectively. 

The fitted linear model from the assignment 1.3 followed the observations relatively well, but did not capture the signal completely. This could especially be seen in the residual plot with the confidence bounds where residuals were not completely centered around zero. These observations indicated that the model is generally good, but misses some important characteristics in the data. Non-linear model of the assignment 1.4 was introduced to add a bit of complexity and probably better fitting.

The non-linear model has four unknowns, two of which are the same as in assignment 1.3 and linear with respect to the computed displacement of the ground surface:
- initial displacement, $d_0$ mm
- displacement response due to the current groundwater level, $k$ 

Groundwater level (GW) data is provided in additional dataset and is used as a part of design matrix.

The second and third unknowns of non-linear model are introduced to replace linear rate of displacement by an exponential dependency with the following parameters:
- response of the soil layers to the extra weight of the road, $R$ mm. 
- a scaling parameter $a$, which represents the memory of the system

The redundancy of the non-linear model using InSAR data is 57, and using GNSS data is 726.

**Question 2**

In the assignment you chose initial values for the non-linear model parameters. Justify your decision.
Explain how and why you define the criterion to stop the iteration of a Gauss-Newton iteration algorithm.

**Solution**

- For $d_0$ and $k$ you could use the estimated values from the linear model
- For $R$: realize that it is the difference between displacement at start and end of the observation interval (look at plot with data).  
- For $a$: you could plot an exponential function $R \left(1-\exp\left(\frac{-t}{a}\right)\right)$ and try different values of $a$ to see which one would fit well here.

For the stop-criterion we will use the 'weighted squared norm' of $\Delta \hat{\mathrm{x}}_{[i]}$, with the inverse covariance matrix $\Sigma_{\hat{X}}^{-1}$ as the weight matrix. In this way we account for different precision of the parameters (high precision means we want the deviation to be smaller), as well as different order of magnitudes of the parameters.

**Question 3**

Report the convergence and quality of the estimation with InSAR and GNSS. 
Give your interpretation for any discrepancy between observations and the model.

**Solution**

Both models converged. With InSAR it took only 5, with GNSS 8 iterations. This can be declared by the difference in precision and number of observations.

The precision of the estimated offset $\hat{d}_0$ and $\hat{k}$ with InSAR is approximately 2 mm and 0.02, respectively. With GNSS the precision of those parameters is approximately a factor 2 worse due to the higher noise in the data. However, due to the high number of observations, the precision is still rather good. Note that the outliers do not have an impact on the precision, since the covariance matrix does not depend on the data (and outliers are non-random errors that are not accounted for in the covariance matrix). Precision of parameters is good if compared to estimated values, except for $a$ (std of 21, 79 days, while estimated value is 180, 224 days for InSAR and GNSS respectively). 

The InSAR discrepancies look as expected, no systematic effects: the residuals have a zero-mean and standard deviation close to the precision of the observables. This tells us that the precision that we used for the observations corresponds to the true noise level. 
    
For GNSS there are still many residuals at the start of the observation period which are outside the confidence bounds, also resulting in slightly larger empirical standard deviation of the residuals. The effect of the outliers is also visible when comparing the fitted model with GNSS and InSAR.

**Question 4**

Give an explanation of test statistic used to test which model (linear, non-linear) fits data better. What is the null hypothesis $H_0$ and alternative hypothesis $H_a$ in this test? What is the distribution of test statistic? Compare the test outcomes with InSAR and GNSS and interpret the results.

**Solution**

The null hypothesis is that we assume that the linear model is correct, the alternative hypothesis that the model is incorrect.

The test statistic is the difference of the weighted squared norms of residuals, and has a Chi squared distribution with 1 degree of freedom, since there is 1 extra parameter in the alternative hypothesis model as compared to the null hypothesis.

- For InSAR the test statistic is 95.6 which is significantly larger than the critical value 7.9. Therefore, the exponential model is accepted in favor of the linear one. 

- For GNSS the outcomes is different: test statistic is equal to 7.8, which is slightly smaller than the critical value 7.9, resulting in acceptance of the null hypothesis (i.e., linear model). The reason is that the GNSS data is much noisier and contains many outliers, such that an exponential trend cannot be distinguished.

**Question 5**

In order to get a better fit to the data (smaller residuals) for this case study, which of the following strategies could help? Elaborate on your answer with one or two sentences each.

1. better observations?
2. a more complicated geophysical model?
3. better initial values?
4. more observations?

In your answer, keep in mind that data acquisition and processing comes with a price. Note that in a real situation you would not look at a time series of only one point. For Sentinel-1 data you may have to pay, collecting GNSS data at different locations also costs money. How will you monitor the deformation if you have both GNSS and InSAR data at your disposal?

**Solution**

1. better observations will help, and should result in smaller residuals.
1. a more complicated geophysical model will help if it is able to capture the signal. However, since we don't see any systematic effects in the InSAR residuals, it is not expected that much gain can be expected. Including more parameters in the model will help to get smaller residuals, but is there still a geophysical explanation...?
1. better initial values won't help, since solution converged to valid results.
1. more observations generally helps, as long as they are not corrupted by outliers or systematic effects.

Use the observations together (i.e., estimate the unknown parameters using the GNSS and InSAR observations at the same time, which would result in 791 observations). With BLUE we would of course apply proper weights, taking into account the different precision.

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.