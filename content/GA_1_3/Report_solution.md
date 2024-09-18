# Report for Group Assignment 1.3

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3, Friday, Sep 20, 2024.*

_This assignment is to be turned in by uploading files to GitHub (as done with PA 1.3, but for the GA 1.3 repository)._

_Most of the questions require you to finish `Analysis.ipynb` first, but depending on how you allocate tasks between group members, it is possible to work on this in parallel. Make sure you save time for peer reviewing each others work before completing the assignment!_

_We don't expect long answers; be as concise as possible (just a few sentences max, usually); however, it is critical to support your arguments with qualitative observations (e.g., describe specific figures) and quantitative evidence (report results of calculations, values, etc) from `Analysis.ipynb` whenever possible._

## Questions

**Question 1**



**Question 2**


**Question 3**



**Question 4**


**Question 5**



**Question 6**

Interpretation questions:

- When you plot the residuals vs time, do you see any systematic effect? Give your interpretation for any discrepancy between observations and the model.
- What is the mean value of the residuals?
- What is the standard deviation of the residuals?
- What can you conclude when you compare the histogram of the data with the computed normal distribution? 
- Did you use quantitative results in your answers?

**Solution**
    
_Note: here we plotted the true model as well, which you did not have._
    
- The mean value and standard deviation of the InSAR residuals is 0.0 mm and 3.115 mm. 
- The mean value and standard deviation of the GNSS residuals is 0.0 mm and 15.393 mm.
    
First of all, for InSAR almost all residuals are within the 99% confidence bounds, indicating that the quality that we assumed for the observations was good. 
    
The fitted model seems to follow the observations relatively well, but does not capture the signal completely. This can especially be seen in the residual plot with the confidence bounds. You see that the residuals are not completely centered around zero but that we still see some 'signal' where the model underpredicts at the ends and overpredicts in the middle. Although the values are negative, we can see that the residual plot removes the trend described by the model and illustrates the "over" and "under" aspect quite clearly. 
    
Moreover, when reviewing the results for GNSS we see only a few outliers (residuals outside the 99% confidence bounds), which is logical given the 99% limit. Furthermore, the left side of the plot have many more observations that are below the confidence bound; this can also be seen in the left tail of the GNSS histogram, which is slightly asymmetric.
    
All of these observations indicate that the model is generally good, but misses some important characteristics in the data. Perhaps we should consider adding a bit of complexity (Part 2!).


**Question**

Compare the results you found for the InSAR observations and the GNSS observations in the questions above. Discuss the differences between the results. Be quantitative!

**Solution**

Estimated parameters, hence fitted model, is different. 
    
Factors that have an impact are
    
- precision of the observations
    
- number of observations
    
- outliers in the GNSS data
    
    
Although the quality of the GNSS data is lower compared to InSAR (15 mm vs 2 mm), the precision of the estimated parameters is only a factor 2 worse. Here we see the effect of 'more' data points: the much lower precision of the observations is somewhat compensated by the much higher number of observations.

The GNSS data seems to have some outliers in the beginning and therefore the model fit is maybe not so good compared to InSAR. 

Also, when reviewing the residuals for both datasets, it seems that the model that we use is maybe too simple since we miss part of the signal. 
    

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.