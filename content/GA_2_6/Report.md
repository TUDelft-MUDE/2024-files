# Project 10 Report: Machine Learning

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

**YOUR GROUP NAME HERE**


## Primary Task

**Complete the notebook `P10.ipynb`and write your answers in this document as requested in the questions below. Note that only part of the notebook results are required to be included in this report.** Typically a short, one-line answer is sufficient; though a simple 'yes' or 'no' is _not_ sufficient, please include a short reasoning, justification or argumentation.

_You will be graded on the plots and answers provided in this file. You can delete the instructions and any other unnecessary text prior to submission._

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

## Answers to Questions

### Section 1

**1.1) What is the purpose of splitting a dataset into training, validation, and test sets in the context of machine learning?**

_Your answer here._


**1.2) What part of the pre-processing improves the representativity of the overall distribution of the data?**

_Your answer here._


**1.3) Why should the `MinMaxScaler` be fitted on the training data only, and then used to transform both the training and validation data?**

_Your answer here._



### Section 2

Plot the validation and training loss curves. Add this plot to your report.

_Your plot here._


**2.1) Based on the shape of the loss curves, what can you indicate about the fitting capabilities of the model? (Is it overfitting, underfitting, or neither?)**

_Your answer here._

**2.2) How do you explain the difference between the values of training and validation score?**

_Your answer here._

### Section 3
Plot the validation loss grid. Add this plot to your report.

_Your plot here._

**3.1) How does hyperparameter tuning in machine learning relate to the concept of model complexity?**

_Your answer here._

**3.2) From the graph, what is the impact of increasing the number of hidden layers on the model's ability to capture complex patterns in the data?**

_Your answer here._

### Section 4

_Your plot here._

**4.1) The score indicates a high fitting, is that reflected in the plot of the errors? Why?**

_Your answer here._


**4.2) Is the the plot of errors centered around zero? If not, what does that mean?**

_Your answer here._


**4.3) How diverse can the speed up values be if you run the cell multiple times? Why?**

_Your answer here._

**4.4) What would occur with the speed up if you increase the number of neurons in the hidden layers?**

_Your answer here._

### Section 5

**5.1) What is the minimum that your model predicts for this network?**

_Your answer here._


**5.2) How confident are you in the prediction of your model? Why?**

_Your answer here._


## General Comments on the Assignment [optional]

_Use this space to let us know if you encountered any issues completing this assignment (but please keep it short!). For example, if you encountered an error that could not be fixed in your Python code, or perhaps there was a problem submitting something via GitLab. You can also let us know if the instructions were unclear. You can delete this section if you don't use it._

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
