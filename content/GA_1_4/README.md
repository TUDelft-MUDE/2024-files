# README for Group Assignment 1.4

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.4, Friday, Sep 27, 2024.*

GA 1.4 continues from where GA 1.3 left off by using the non-linear least squares method to create a non-linear model of the InSAR and GNSS satellite data to evaluate road deformation.

There are several files to be aware of:

1. `README.md`: this file, which contains an introduction and instructions.
2. `Analysis.ipynb`: a Jupyter notebook which contains the primary analysis to complete during the in-class session.
3. `Report.md`: a Markdown file containing questions about the findings of Analysis, as well additional questions to check your understanding of the topic for this week.
4. `functions.py`: a Python file that defines various functions for use the notebook
5. `setup.py`: a Python file that setups repeats the analysis from GA 1.3 and saves the results to a pickle file for loading into `Analysis.ipynb`
6. `auxiliary_files`: a subdirectory that contains `csv` and `pickle` files.

The grade for this assignment will be determined as follows (described below):

- The notebook `Analysis.ipynb` is worth 20%
- Each question in the `Report.md` has equal weight.

Assignment submission, grading and working method are similar to last week with the following exceptions:

1. You should clone the repo as done in the PA this week. You can select one group member to commit and push the changes back to the GitHub repo at the end of the session.
2. If more than one team member wants to make a commmit, make sure you first commit you local changes, then fetch and pull from GitHub before pushing your own changes.
3. VS Code does not work well when all group members are editing the same file actively. When using live share we recommend that you use it primarily to follow each other and avoid frequently making edits simultaneously. For example, you can work on the Report questions separately and then have each member paste their work in one at a time to avoid glitches.

## Assignment Context

The introduction of the non-linear model is included in `Analysis.ipynb`. See the [GA 1.3 README](https://mude.citg.tudelft.nl/2024/files/GA_1_3/README.html) for background information.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.