# GA 2.8 Report: [Ice Ice Baby](https://www.youtube.com/watch?v=rog8ou-ZepE)

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

_Questions 3 and 4 are the most important for this assignment. They are weighted roughly 2-3 times more than the others._

## Questions

**Questions 1: Describe the probability associated with winning the Ice Classic when purchasing a single ticket. Describe the range of values possible and illustrate your description using values reported in a Markdown table.**

The probability is...low! Choosing the right day is max 0.06 and right minute is max 0.002. The product together is the product, 1e-4. This decreases by several orders of magnitude as you go far away (e.g., down to 1e-8).

**Question 2: Notebook 3 illustrates how you can change the probability model for breakup day and time. Using the information in the PDF from Week 1.1, make an adjustment to the probability model and report how this improves the probability of winning.**

_The PDF can be found [here](https://mude.citg.tudelft.nl/2024/files/Week_1_1/). This question simply quantifies the change you made lists the reasons why it may increase your chances of winning._

There is no right answer here. The main point was to recognize that the historical average gives a good idea for when breakup will occur, but extra information prior to the event can be used to adjust the distribution. This could be a weather forecast, or a change in the river flow rate, etc. For example, it is easy to imagine that if you can establish the weather is exceptionally cold or warm, the breakup will shift earlier or later; this can best be applied by reducing or increasing the mean value of the distribution of breakup day (as done in Notebook 03).

You could also have changed the standard deviation of the day or minute, but that would have had a different reasoning. Perhaps you thought that the probabilities estimated for the "unpopular" minutes were too low, in which case increasing the standard deviation would improve that.

**Question 3: Justify why you changed the distribution in Question 3 and indicate how MUDE methods would help you quantify this decision, if you had more time.**

_In this question you should try to indicate how the data informs your decision to change the distribution parameters. To explain how this could be quantified, list three examples of analyses that could be completed using MUDE methods. You should also provide a brief explanation of why each analysis would be useful and which data it would use._

This is also subjective. Perhaps an easy one might have been to use time series analysis to make a model of temperature each spring, then use that to stationarize the observations of temperature up to APril 5 (the betting date) to determine if the weather was warmer or colder than average. By analyzing the correlation between specific parameters associated with the observed pattern (for example, if you observe 5 days in a row with average temperature above 0 C), you might be able to estimate the best estimate of the breakup day (i.e., earlier than the historic average).

**Question 4: come up with a ticket combination that seems to maximize your chances of winning, without taking on an undesirable level of (financial) risk. Explain your thought process for selecting the chosen ticket combination. You are free to use the original distribution or an altered one (as long as you justify it).**

_Your answer here._

This question involved a lot of exploration. The essential thing to notice is that there is a tradeoff between finding the "likely" minutes (close to the joint mode of the distribution) and the "popularity" of those minutes (you must share the victory with many people, thus reducing your winnings). In the end, the most advantageous situation was one where you think the breakup will happen much earlier or later than average, thereby making the chance higher that your ticket will win, while simultaneously choosing minutes that have a smaller $N_w$ to divide the winnings over.

Other interesting things were:
- noting that there is a huge change in the distribution of $N_w$ from minute to minute
- there is strange behavior (due to human betting practices) in the pattern of $N_w$ for various days; for example, people all of a sudden choose far fewer tickets on day 41 compared to day 40!

**Question 5: Make a Bet! Fill out the file `ticket.md` and also submit a paper ticket.**

_This question can be completely unrelated to the more formal analysis above---have fun with it! Decide with your group what your bet will be with your real Ice Classic ticket. Remember that the closest MUDE group to the real minute will win a prize from Robert, and if one of us wins the real thing we will all split the payout!_

_You don't need to write anything in this file, only the `ticket.md` file._

No matter the justification, we hope you made a lucky guess - let's hope we win this year!

**Question 6: confirm that you did the following.**

_Did you know you can use checkboxes in Markdown? Try it out as you mark each task as complete by filling the empty `[ ]` with an `x` like this `[x]`._

- [ ] I submitted a paper ticket.
- [ ] I completed the file `ticket.md` with the correct information.
- [ ] I made sure that the information on both items is the same!
- [ ] We are going to win the Ice Classic!

## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
By <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE Team</a> &copy; 2024 TU Delft. <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. doi: <a rel="Zenodo DOI" href="https://doi.org/10.5281/zenodo.16782515">10.5281/zenodo.16782515</a>.

