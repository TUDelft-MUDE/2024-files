# Feedback for Group Assignment 1.7

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.7, Friday, Oct 18, 2024.*

**This Feedback file contains a few observations and conclusions for all three data sets. See the solution notebooks for more details about a particular data set. A Report solution is not provided, as all relevant information is summarized in this file or the notebooks.**

A few particular notes for specific data sets are identified with italics (_disharge, emissions_ or _force_).

## Questions

**Question 1**

- Discharge best fit: h by Gaussian, u by Gumbel
- Emissions best fit: C by uniform, H by Gaussian
- Force best fit: H by exponential, T by Gumbel

Avoid words such as “clearly” or “obviously” when choosing your distribution. State your assumption and provide arguments.

_Emissions:_ Most groups went for a log-normal distribution for C. It can be argued that you use lognormal, however in the GoF tests, it should strike you that the right tail does not fit. Some groups went for Gumbel for H. Do not solely rely on your metrics, make sure you visually inspect the data, do you really see it skewed? Maybe try both and look at the fitted line through a histogram.

**Question 2**

p-values are
- Discharge: h --> 0.719, u --> 0.152 (both H_0 not rejected)
- Emissions:  H --> 0.001, C --> 0.0 (both rejected)
- Force: H --> 0.0, d --> 0.0 (both rejected)

Check your KS-tests! Some groups gave some wrong arguments in the function. Also check what it means and definitely do not “discard” it because it tells you something different than your QQ plot. It is possible to reject the null hypothesis, which is based on a strict mathematical criteria, but still accept the distribution for other reasons (e.g., the deviations in the QQ plot are sufficiently small for your purposes; or perhaps it's better than the other distribution). If you were given advice to "discard" the KS test from a teacher in class, you were still expected to justify the reasoning behind it.

Check your code to make sure that you computed the value correctly. Additionally, do not compare p-values, this results in mistakes in interpretation (the p-value should be used to assess pass/fail, it is not meant to be a relative comparison).

A function does not fit worse or better because the KS test P value is lower or higher respectively.

When you say “it does not fit well” ask yourself why. This is what we look for in argumentation. 

Lastly, do not be too quick and claim things “fit quite well” even if you see large deviations at the tails. Be precise in your wording. Example: “The distribution fits the data well below X meters, however the distribution over/underestimates the right tail.”

**Question 3**

Some groups highlight that “the simulations look much more like X distribution than the data”. Do not forget to mention why. This is of course expected as the simulations are sampled from these distributions.

Make sure to point out certain differences such as the spreading and where these came from.

Check if your values have meaning. Does a negative force or a negative emission have meaning? If you truncated your data or dealt with these values in another way, specify it in the report or add a plot.

**Question 4**

The correlation coefficients are:
- Discharge: 0.39
- Emissions: 0.27
- Force: 0.46

Many groups simply stated the value of the correlation coefficient and forgot to argue what effect correlation has on the final distribution. 

_Discharge:_ The distributions may fit well, however the right tail does not, a recommendation could be to choose another one.

**End of file.**

<span style="font-size: 75%">
By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a> &copy; 2024 TU Delft. <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. doi: <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515">10.5281/zenodo.16782515</a>.