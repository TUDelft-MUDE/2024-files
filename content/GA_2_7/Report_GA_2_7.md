# Group assignment 2.7 Report: Extreme Value Analysis

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

**YOUR GROUP NAME HERE**


## Primary Task

**Complete the notebook `NOTEBOOK_NAME.ipynb` and write your answers in this document, as requested in the questions below. Note that only part of the notebook results are required to be included in this report.** Typically a short, one-line answer is sufficient; though a simple 'yes' or 'no' is _not_ sufficient. Please include a short reasoning, justification or argumentation.

_You will be graded on the plots and answers provided in this file. You can delete the instructions and any other unnecessary text prior to submission._

**Are you using ChatGPT or similar tools?** _We don't mind, but please don't ask ChatGPT the questions written here and copy/paste whatever answer it gives you - this isn't fun to read and robs you of an opportunity to learn. Try to ask small simple questions to ChatGPT, then read the response carefully and edit it before using it here. It would also be good to ask one or more of your fellow group members to proof-read it before submitting!_ 

## Report Instructions

Remember to use Markdown features to clearly indicate your answers for each question below. 

**Importing figures into a Markdown file:**
1. Use relative referencing only, with the git repo (working directory) as the root (this is expressed with a single dot `.`)
2. Our grading systems are case-sensitive so match the names of folders exactly
3. Use linux-style path separators: `/` rather than `\`.
4. Do not include spaces in your file path or image name; if it is unavoidable replace the space with `%20`, for example: `![My image](./my%20image.png)`

Here are some examples:
- an image located in the working directory `![My image](./imagename.ext)` (where `ext` is any image extension).
- an image located in a sub-directory called "images": `![My image](./images/imagename.ext)`
- an image with a space in the file name: `![My image](./images/my%20image.png)`

When using Markdown to include an image, the square brackets is a text tag that is displayed in case the image does not load. Do not include a dot in the square brackets; i.e., do _not_ do this: `![my image.](./image.svg)`.

## Questions

**1. Provide a short description of your data set.**

_Your answer here. Include qualitative and quantitative information; do not duplicate the information from the README, but rather the observations you make in Task 1, and anything you think is relevant for the EVA. Don't include a figure of the time series, since that is requested in a later question_


**2. What type of distribution do you need to use in the Block Maxima method? Summarize the parameters of this distribution including the tail type. When looking at the fit, would you consider the block maxima method to be appropriate?**

_Your answer here. Summarize the distribution parameters in a Markdown table. Include a semi-log plot of exceedance probability that compares the empirical and fitted distribution. Don't include a figure of the time series, since that is requested in a later question._

_Here is a hint for a simple table; feel free to modify._

| Shape | Location | Scale |
| :---: | :---: | :---: |
| ? | ? | ? |


**3. What type of distribution do you need to use the Peak over Threshold method? Summarize the parameters of this distribution including the tail type. Do you need to add/subtract the threshold when using this method, and if so, at what point in the analysis do you do so? When looking at the fit, would you consider the peak over threshold method to be appropriate?**

_Your answer here. Summarize the distribution parameters in a Markdown table. Include a semi-log plot of exceedance probability that compares the empirical and fitted distribution. Don't include a figure of the time series, since that is requested in a later question._


**4. Comment on any differences you see between the distributions from the two EVA Methods (just one or two sentences, using the figures included above). In terms of information used to fit each distribution, what is the main difference between the two methods?**

_Your answer here. Include a figure of the time series that presents sampled block maxima as red dots and POT samples as blue dots, and refer to this in your answer to the second part of the question._


**5. Compare return periods of the event of October 2024 produced by the distributions of the two EVA Methods Explain the reason for the differences you observe.**

_Your answer here. To support your explanation, include the return period figure, as well as a Markdown table that summarizes (quantitatively) the return periods._


**6. Which return period would you pick forthe event of October 2024? Justify your answer. Comment also on how you can improve the analysis.**

_Your answer here._

## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
