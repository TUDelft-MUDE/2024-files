# Report for Group Assignment 1.2

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.2, Friday, Sep 13, 2024.*

_This assignment does not need to be turned in._

_Most of the questions require you to finish `Analysis.ipynb` first, but depending on how you allocate tasks between group members, it is possible to work on this in parallel. Make sure you save time for peer reviewing each others work before completing the assignment!_

## Contents

**Question 1**

What is the expected value and standard deviation of the ice thickness after 3 days ($\mu_H$ and $\sigma_H$)? There should be two sets of results.__


Mean and standard deviation of linearized function:
- $\mu_h = 0.278$ m
- $\sigma_h = 0.034$ m


Mean and standard deviation of simulated distribution:
- $\mu_h = 0.278$ m
- $\sigma_h = 0.035$ m

**Question 2**

_Explain whether we should use the expected value for our prediction, or whether we should also account for the variability of the thickness estimate in the subsequent phases of our analysis?_

It depends! $\sigma_H$ = 0.034m is approximately 10% of $\mu_H=$0.278m; whether or not thisis a large variability depends on the sensitivity of our (unknown) model for predictingbreak-up day as a function of ice thickness. If that model is not very sensitive to ice thickness, it may not be necessary to account for the uncertainty in subsequent phases of the analysis and we can use the expected value directly. If it is very sensitive to this parameter,we should include the uncertainty, perhaps with another round of uncertainty propagation (or numerical simulation).

**Question 3**

_How do we obtain the "true" distribution of $H_{ice}$, and what does it look like?_

To find the "true" distribution of $H_{ice}$ we must propagate the uncertainty through simulations. That is, we know the distribution functions of the inputs $\Delta T$ and $H_{ice,0}$ and we know that they are independent, so we can generate random samples of them directly from the distribution functions, plug them into the function and use the resulting realizations of $H_{ice}$ to approximate the "true"distribution. We can see that the PDF of the Normal distribution matches the histogram quite closely, however, looking at the quantile plot indicates that the extreme low values of the distribution (the lower, or left tail) are not approximated well by the Normal distribution.

**Question 4**

_Are the propagated and simulated $\mu_H$ and $\sigma_H$ equivalent?_

They are not exactly equivalent, but within 1 mm of each other, which is close enough for our purposes of estimation.

**Question 5**

_Is the Normal distribution a reasonable model for $H_{ice}$?_

Yes, since the central moments seem to be properly estimated the Normal distribution would be a reasonable model, with one exception: we can see that the simulated values deviate from the line in the tails. Thus, if you are interested in estimating values with low non-exceedance probabilities this is a bad model: we can see that below 3 standard deviations the difference is a few cm, or around 10% of the expected value. This would be a concern if your break-up date model is sensitive to accurately predicting the likelihood of exceptionally small ice thicknesses given the uncertainty in our estimates of the future temperature and past measurements of the true ice thickness.

**Question 6**

_Why is the sampling distribution not the "true" distribution?_ **DRAFT**

In reality we would like to measure this, but we can't, so we assume distributions of input. This is an assumption, and not be a true representation of reality; it could be very far off. The function itself is also a model and introduces error. Thus the "sample" is only "true" if our model(s) are perfect. Jammer... **DRAFT**

**Question 7**

_Describe the values of the function of random variables (the output) for which we might expect the model to be inaccurate. Quantify this inaccuracy by comparing the probability calculated with the assumed distribution with the frequency of similar values observed. Use the empty cell in Task 3 in the `Analysis.ipynb` file for computations._ **DRAFT**

students should identify the q values where the dots deviate from the line, then find the probability of being outside those values (the tails). then they will need to count the number of times these values occur in the sample. (probably need to give a hint for how to do this-Berend!). I haven't checked the numbers, but I'd expect them to be significantly different. **DRAFT**

**Question 8**

_Test your learning for this week!_ **DRAFT**

Task is to do Exercise 2 on the [Q1 Exam from 2023](https://mude.citg.tudelft.nl/2024/files/Exams/23_Q1.html). **DRAFT**

**Question 9**

_Apply this to another situation!_ **DRAFT**

_WILL LEAVE OUT IF NOT READY IN TIME._ **DRAFT**


**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.