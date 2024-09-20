# Report for Group Assignment 1.3

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3, Friday, Sep 20, 2024.*

_Remember to read the grading and submission instructions in the `README.md` file thoroughly before finalizing your answers in this document!_

## Questions

**Question 1**

Give a short explanation about the data and the models we created. Include brief statements about the data, including an explanation of the groundwater interpolation step and why it was necessary.

_Your answer should only be a few sentences long, and be sure to use quantitative information! You do not need to reproduce the model, but it would be useful to do things like confirm the number of models, model parameters, observations, etc. You can use bullet lists to summarize and state interesting information._

**Solution**

Three data sets were used. We had InSAR data, GNSS and waterlevel data. InSAR and GNSS were used as observations and waterlevel was used as parameter. Two models were made using BLUE, one using InSAR and the other using GNSS as observations. All three datasets are reported in mm. The standard deviation of InSAR and GNSS is assumed to be 2 mm and 15 mm respectively. 

InSAR has 730 observations and GNSS 61, groundwater has 25. To use the groundwater data it is linearly interpolated to get 730 and 61 values for both models. This is necessary since for each observation we need a value for each parameter.

**Question 2**

Describe what the confidence intervals are and mention specifically the type of information that is included in the algorithm to calculate. Make observations about the CI's for each model, and explain why all of the data points are not contained within the interval, even though the value is close to 100%.

_Hint: which uncertainties are propagated into the CI? Are the intervals uniform for the entire time series?_

**Solution**

Confidence intervals are a useful way to report uncertainty. They are based on theoretical distribution. Using the first and second moments (i.e. mean and std. dev.), we can create confidence intervals based on a specific critical value. When our sample size is large enough and our assumptions that our random variable is distributed randomly, we will see that the number of data points in our CI is (almost) equal to our critical value (96%).

However in this case our residuals are not distributed normally and our data sets (especially the GNSS) which will mean our CI's are not the truth. However, it can still be a useful tool to analyze our (propagation of ) uncertainty 

**Question 3**

Do you see any systematic effects in the residuals for each model? Make a few relevant observations about the residuals and the , do you see any systematic effect? Give your interpretation for any discrepancy between observations and the model.

_It may be useful to include qualitative and quantitative observations about the residuals, for example, the histogram/PDF comparison and the moments of the residuals._

**Solution**
    
_Note: here we plotted the true model as well, which you did not have._
    
- The mean value and standard deviation of the InSAR residuals is 0.0 mm and 3.115 mm. 
- The mean value and standard deviation of the GNSS residuals is 0.0 mm and 15.393 mm.
    
First of all, for InSAR almost all residuals are within the 99% confidence bounds, indicating that the quality that we assumed for the observations was good. 
    
The fitted model seems to follow the observations relatively well, but does not capture the signal completely. This can especially be seen in the residual plot with the confidence bounds. You see that the residuals are not completely centered around zero but that we still see some 'signal' where the model underpredicts at the ends and overpredicts in the middle. Although the values are negative, we can see that the residual plot removes the trend described by the model and illustrates the "over" and "under" aspect quite clearly. 
    
Moreover, when reviewing the results for GNSS we see only a few outliers (residuals outside the 99% confidence bounds), which is logical given the 99% limit. Furthermore, the left side of the plot have many more observations that are below the confidence bound; this can also be seen in the left tail of the GNSS histogram, which is slightly asymmetric.
    
All of these observations indicate that the model is generally good, but misses some important characteristics in the data. Perhaps we should consider adding a bit of complexity (Part 2!).

**Question 4**

Compare the results you found for the InSAR observations and the GNSS observations. Discuss the differences between the results. Be quantitative!

**Solution**

Estimated parameters, hence fitted model, is different. 
    
Factors that have an impact are
    
- precision of the observations
    
- number of observations
    
- outliers in the GNSS data
    
    
Although the quality of the GNSS data is lower compared to InSAR (15 mm vs 2 mm), the precision of the estimated parameters is only a factor 2 worse. Here we see the effect of 'more' data points: the much lower precision of the observations is somewhat compensated by the much higher number of observations.

The GNSS data seems to have some outliers in the beginning and therefore the model fit is maybe not so good compared to InSAR. 

Also, when reviewing the residuals for both datasets, it seems that the model that we use is maybe too simple since we miss part of the signal. 
    

**Last Question: How did things go? (Optional)**

_Use this space to let us know if you ran into any challenges while working on this GA, and if you have any feedback to report._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.