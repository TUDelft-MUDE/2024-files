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

It depends! $\sigma_H$ = 0.034m is approximately 10% of $\mu_H=$0.278m; whether or not this is a large variability depends on the sensitivity of our (unknown) model for predicting break-up day as a function of ice thickness. If that model is not very sensitive to ice thickness, it may not be necessary to account for the uncertainty in subsequent phases of the analysis and we can use the expected value directly. If it is very sensitive to this parameter,we should include the uncertainty, perhaps with another round of uncertainty propagation (or numerical simulation).

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

_Using the loop in Task 3.1, explore the effect of sample size on the results. Describe the observations you make and explain why they are happening._

The results are clearly dependent on sample size; this is logical given that the set of samples from a Monte Carlo method asymptotically converges to the true distribution as sample size increases. It is obvious that the empirical distribution is not good for $N=5$ and $N=50$, but it is interested how _skewed_ the distribution is for $N=5000$. Conclusion? Always be mindful that the number of samples matters! (We will learn a technique for quantifying the accuracy later in MUDE).

**Question 7**

_Why is the sampling distribution not the "true" distribution?_ 

There are two primary reasons. The first is captured by the previous question and answer: the result from a Monte Carlo Simulation (an empirical distribution) depends on the number of samples. As this increases, the "true" distribution is better approximated (asymptotically).

THe second reason is that the Monte Carlo Simulation is only as good as our assumptions for the input distributions and the function itself. If these are a poor representation of reality, then the MCS result will be poor as well. For example, if we assume Normal distributions for the input parameters, but the "true" distributions are skewed, the results may be inaccurate.In reality we would like to measure this, but we can't, so we assume distributions of input. This is an assumption, and not be a true representation of reality; it could be very far off. The function itself is also a model and introduces error.

In short, the "sample" (our model) is only "true" if our model(s) are perfect, which in reality is never the case. Jammer!

**Question 8**

_Describe the values of the function of random variables (the output) for which we might expect the model to be inaccurate. Quantify this inaccuracy by comparing the probability calculated with the assumed distribution with the frequency of similar values observed. Use the empty cell in Task 3 in the `Analysis.ipynb` file for computations._

_Hint: you can count the number of values in an array that conform to a specific boolean condition using_`sum(MY_ARRAY <= MY_VALUE)`. _It may also be useful to find the length of an array with `len(MY_ARRAY)`._

It looks like the Normal distribution and sampled distribution deviate for values of $q<2$, or $H_{ice}<0.2$ m (values where the dots deviate from the line). A short `for` loop in the code computes the probability of being in the lower tails of the theoretical and empirical distribution, which is around 1% and 2%, respectively. This is not a huge difference, however, we see that the estimated probability values differ _significantly_ as the value of ice thickness decreases. For example, at 0.13 m the theoretical distribution underpredicts the probability by a factor of 40! This means the uncertainty propagation law (combined with an assumption of the Normal distribution) will underpredict the chance that the ice is thin, which could have significant impact on the accuracy of our ice model and our bets. Conclusion: we better study more probability to account for the tails of the distributions better! (We will do this later in MUDE).

Note that in this case the use of the terms _probability_ and _frequency_ are terms used for the distribution and empirical sample, respectively; in both cases they quantify the uncertainty associated with observing specific values of ice thickness after a few days.

**Question 9**

_Test your learning for this week!_ **DRAFT**

Task is to do Exercise 2 on the [Q1 Exam from 2023](https://mude.citg.tudelft.nl/2024/files/Exams/23_Q1.html). **DRAFT**

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.