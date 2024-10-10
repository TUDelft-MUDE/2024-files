# Report for Group Assignment 1.3

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3, Friday, Sep 20, 2024.*

_Remember to read the grading and submission instructions in the `README.md` file thoroughly before finalizing your answers in this document!_

## Questions

**Question 1**

Give a short explanation about the data and the models we created. Include brief statements about the data, including an explanation of the groundwater interpolation step and why it was necessary.

_Your answer should only be a few sentences long, and be sure to use quantitative information! You do not need to reproduce the model, but it would be useful to do things like confirm the number of models, model parameters, observations, etc. You can use bullet lists to summarize and state interesting information._

_We copy the LaTeX equation for the model here, in case you would like to refer to it in your Report._

$$
d = d_0 + vt + k \ \textrm{GW},
$$

**Solution**

_Note that you were not expected to write this much in your own solution!_

Three data sets were used. We had InSAR and GNSS data (both satellite observations of the ground surface) and groundwater level data. InSAR and GNSS were used as observations and groundwater level was used as a (deterministic!) parameter. Two models were made using BLUE, one using InSAR and the other using GNSS as observations.

InSAR has 61 observations and GNSS 730, groundwater has 25. The standard deviation of each InSAR and GNSS measurement is assumed to be 2 mm and 15 mm, respectively. All three datasets are converted to mm when used in the Analysis notebook (satellite data is converted from m).

The model has three parameters, each of which are linear with respect to the computed displacement of the ground surface:
- initial displacement, $d_0$ mm
- rate of displacement as a function of time, $v$ mm/day
- displacement response due to the current groundwater level, $k$ 

$$
d = d_0 + v\ t + k \ \textrm{GW},
$$

Note in particular the differences between each term: $d_0$ and $k\ \textrm{GW}$ are constant in time, whereas $v\ t$ is not. The term $k\ \textrm{GW}$ is unique as it changes directly with the groundwater level; the relationship with time is thus non-linear, but deterministic.

To construct the model we need a parameter value for each observation (GNSS/InSAR); this is problematic as we only have 25 groundwater measurements. Luckily the groundwater level changes relatively slowly so we can interpolate (linearly) between the observations. This allows us to have 730 and 61 values that match the other datasets, which are used to construct the two models. 

**Question 2**

Two types of confidence intervals are used in this assignment. Describe what the confidence intervals are and mention specifically the type of information that is included in the algorithm to calculate them. Make observations about the CI's for each model. Explain why, even though the $\alpha$ value is small (i.e., the CI is close to 100%), in the time series plots many of the data are _not_ contained within the CI, whereas they _are_ contained within the CI for the residual plots.

_Hint: Are the intervals uniform for the entire time series? Which uncertainties are propagated into each type of CI?_

**Solution**

Confidence intervals are a useful way to report uncertainty and are based on an assumed theoretical distribution around a quantity of interest; in this case we assume a Normal distribution about each model prediction, $\hat{y}_i$, as well as each residual, $y_i-\hat{y}_i$. Using the first and second moments (i.e. mean and std. dev.), we can create confidence intervals based on a specific false alarm rate or significance level, $\alpha$. When our sample size is large enough and our assumptions that our random variable is distributed randomly are reasonably correct, we will see that the fraction of data points in our CI is (almost) equal to our critical value.

The two confidence intervals are different because of the way the covariance matrices $\Sigma_{\hat{Y}}$ and $\Sigma_{\hat{\epsilon}}$ are calculated, and the information that they represent. Even though both incorporate uncertainty of the observations, $\Sigma_{Y}$, the confidence interval of each model prediction $\hat{y}_i$ is based on the uncertainty of the model _parameters_, which is what the CI in the time series plot illustrates.

The reason why most of the data is not included in the first confidence interval is because it represents the confidence interval for the true but unknown change in deformation. As it is _not_ a confidence interval for the error in the observation data, the data points are not included in it.

**Question 3**

Do you see any systematic effects in the residuals for each model? Make a few relevant observations about the residuals. Give your interpretation for any discrepancy between observations and the model.

_It may be useful to include qualitative and quantitative observations about the residuals, for example, the histogram/PDF comparison and the moments of the residuals._

**Solution**

The mean value and standard deviation of the InSAR residuals is 0.0 mm and 3.115 mm.

The mean value and standard deviation of the GNSS residuals is 0.0 mm and 15.393 mm.

After examining the residual plots, for InSAR almost all residuals are within the confidence bounds, indicating that the quality that we assumed for the observations was good.
    
The fitted model seems to follow the observations relatively well, but does not capture the signal completely. This can especially be seen in the residual plot with the confidence bounds. You see that the residuals are not completely centered around zero but that we still see some 'signal' where the model underpredicts at the ends and overpredicts in the middle. Although the values are negative, we can see that the residual plot removes the trend described by the model and illustrates the "over" and "under" aspect quite clearly. 
    
Moreover, when reviewing the results for GNSS we see only a few outliers (residuals outside the confidence bounds), which is logical given the high limit (low $\alpha$). Furthermore, the left side of the plot has many more observations that are below the confidence bound; this can also be seen in the left tail of the GNSS histogram, which is slightly asymmetric.
    
All of these observations indicate that the model is generally good, but misses some important characteristics in the data. Perhaps we should consider adding a bit of complexity (we will do this next week!).

**Question 4**

Compare the results you found for the InSAR observations and the GNSS observations. Discuss the differences between the results. Be quantitative!

**Solution**

The estimated parameters, hence the fitted model, are different.

| Model | $d_0$ [mm] | $v$ [mm/day] | $k$ [$-$] |
| :---: | :---: | :---: | :---: |
| InSAR | 9.174 | -0.0243 | 0.202 |
| GNSS | 1.181 | -0.0209 | 0.16 |
    
Factors that have an impact are:
    
- precision of the observations
- number of observations
- outliers in the GNSS data

The precision (standard deviation) of each parameter is:

| Model | $\sigma_{d_0}$ [mm] | $\sigma_{v}$ [mm/day] | $\sigma_{k}$ [$-$] |
| :---: | :---: | :---: | :---: |
| InSAR | 2.128 | 0.0012 | 0.016 |
| GNSS | 4.467 | 0.0026 | 0.035 |
    
Although the quality of the GNSS data is lower compared to InSAR (15 mm vs 2 mm), the precision of the estimated parameters is only a factor 2 worse. Here we see the effect of "more" data points: the much lower precision of the observations is somewhat compensated by the much higher number of observations.

The GNSS data seems to have some outliers in the beginning and therefore the model fit is maybe not so good compared to InSAR. 

Also, when reviewing the residuals for both datasets, it seems that the model that we use is maybe too simple since we miss part of the signal. 

_Note: in the solution we plotted the true model as well, which you did not have. This makes it easier to see where the data have some outliers; in fact this was the "manually" adjusted values that deviate from the rest, which were generated randomly using a Monte Carlo Simulation with Normally distributed noise._

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.