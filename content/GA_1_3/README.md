# README for Group Assignment 1.3

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3, Friday, Sep 20, 2024.*

_A link to access this GA will be provided in class on Friday morning. It will work the same way as PA1.3(on GitHub)._

There are several files to be aware of:

1. `README.md`: this file, which contains and introduction and instructions.
2. `Analysis.ipynb`: a Jupyter notebook which contains the primary analysis to complete during the in-class session.
3. `Report.md`: a Markdown file containing questions about the findings of Analysis, as well additional questions to check your understanding of the topic for this week.
4. `functions.py`: a Python file that defines plotting functions for the last Task of the notebook
5. Auxiliary files: a `data` subdirectory that contains `csv` files.

The grade for this assignment will be determined as follows (described below):

- The notebook `Analysis.ipynb` is worth 20%
- Each question in the `Report.md` is worth 20%

### Submission

This assignment is to be turned in by uploading files to GitHub (as done with PA 1.3, but for the GA 1.3 repository). You only need to upload modified files (e.g., `Analysis.ipynb` and `Report.md`). Do not change the file names when uploading.

**The deadline to upload the assignment to GitHub is 12:30.** There is a one-hour grace period where you can upload the files without receiving a penalty; however, you can **not** use this grace period to improve your answers or analysis; it is only for fixing problems with uploading files. You should stay in the classroom to complete this task so that teachers can help you with any problems you may have. The 13:30 deadline is strict; no assignment will be accepted after this time without a grading penalty, and you must notify us via email that your assignment is late.

### Working Method

You are expected to read and work through `README`, `Analysis` and `Report` sequentially. Most of the questions require you to finish `Analysis.ipynb` first, but depending on how you allocate tasks between group members, it is possible to work on this in parallel. Make sure you save time for peer reviewing each others work before completing the assignment!

We highly recommend you use the VS Code Live Share feature to work on the assignment, especially for the `Report.md`. It may be useful to identify one group member at the beginning of the session to download the files and set up the Live Share session.

## Grading

Your GA will be graded on a 10 point scale and consists of notebooks (e.g., `Analysis.ipynb`) and a number of questions in the `Report.md`. Individual notebooks and questions will be graded on a 10 point scale, and the total grade for the GA is calculated as a weighted average of each. Each question will be graded on the relevance and correctness of your answers (**not** the length!). Unless otherwise stated, the notebook and other programming-related content is graded primarily for completeness.

### Individual Questions

Here is an example for how an individual question may be graded:

- 0-5 points: excessive use of AI in an unedited form. Answers are general and not relevant. No quantitative information included. Significant parts of the questions not answered.
- 6-8 points: good answers, captures the main points. Perhaps too long or misses important insights into the problem, or contains limited quantitative information. If analysis was not completed, or concept not well understood, the answers contain useful discussion, information or insight into the issue faced by the group.
- 9-10 points: exceptional answers that are quantitative, concise and insightful.

_Note that you can still pass the questions/assignments even if you do not complete the analysis! You are always welcome to report issues in the last question of the `Report.md`._

### Notebooks and Programming

Individual tasks in the notebook are not graded. If the notebook is completed successfully you will generally get full credit, unless it is clear that something was implemented incorrectly, or if the notebook was submitted in an unreadable state (i.e., without code cell outputs visible).

Here is an example for how your notebook may be graded:
- 0-5 points: incomplete without explanation. Unreadable results in GitHub.
- 6-9 points: incomplete but contains explanation justifying reason or evidence that work was done.
- 10 points: notebook completed as requested.

### Tips for Writing the Report

Keep these key points in mind:

1. Don't regurgitate information from the assignment or book---we are not interested in your ability to copy/paste our own material.
2. Shorter is better!
3. Use quantitative information: for example, instead of "data set 1 is bigger than data set 2" try "data set 1 has 446 values with a mean of 5.6 m and std dev of 1.2 m. Data set 2 has 25 values with a mean of 4.9 m and std dev of 0.1 m.
4. You are welcome to use AI tools, but please state whether or not you did so (e.g., "ChatGPT gave the bullet list of pros/cons, which we then modified.").
5. If you use AI, do **not** copy/paste questions and answers blindly! It is best to ask small questions and modify the answers using your knowledge of the topic and our activity. Answers that we suspect of containing a significant AI component will not recieve any points.
6. Group members should peer review answers before submission (especially if you are using AI tools).

## Assignment Context

There are numerous subsurface processes that give rise to observable subsidence at the surface. These processes can be categorized into two main categories: 'deep' subsidence and 'shallow' subsidence. 'Deep' subsidence stems from processes occurring in the deeper subsurface layers (e.g., deeper than 0.5 kilometers below the surface). For instance, extracting gas from a reservoir leads to compaction of the deeper layers, which then results in subsidence of the Earth's surface. On the other hand, 'shallow' subsidence arises from activities within the upper layers of the subsurface. When the groundwater table drops, it triggers shrinkage and oxidation of organic material above the ground water level. Additionally, processes like consolidation and creep contribute to shallow subsidence. Conversely, when the groundwater level rises, a portion of the subsidence becomes reversible, as the layers swell due to the increased water content.

In the Green Heart region in the Netherlands a lot of 'shallow' subsidence occurs. In the typical polder landscape the groundwater table is kept blow a certain level, causing peat layers to oxidize and shrink resulting in subsidence. Also, since the ground water level is highly variable over the year (due to changes in temperature and precipitation), this results in highly variable ground movements which can be quite significant. 

In the context of our assignment, we investigate the observed deformation of a recently constructed road in the <a href="https://www.groenehart.nl/the-green-heart-of-holland" target="_blank"> Green Heart</a> Region. It's reasonable to anticipate that when a heavy structure is built on top of a 'soft' soil layer, additional subsidence may occur due to compaction in the upper surface layers. Over time, as the sediment settles, this extra compaction will diminish. However, it is still expected to observe some up and down movement related due to changing ground water levels. 

### Data

The input data for this assignment are two different deformation time series for a (hypothetical) road in the Green heart in the Netherlands. We assume that the road was built in 2016. We will have a look at <a href="https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar" target="_blank"> InSAR</a> (Interferometric Synthetic Aperture Radar) data and <a href="https://en.wikipedia.org/wiki/Satellite_navigation" target="_blank"> GNSS</a> (Global Navigation Satellite System) data.

With InSAR we can retrieve displacements from time series of radar images. In this exercise we will consider displacement time series from Sentinel-1 from 2017 till 2019. More information on the Sentinel-1 mission can be found <a href="https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1" target="_blank">here</a>.

In the project repository on GitLab, you will find three data files for this assignment:
- `gnss_observations.csv`
- `insar_observations.csv`
- `groundwter_levels.csv`. 

Note that all files consist of two columns, one for the dates and one for the observations. 

In the GNSS and InSAR files the observations are observed vertical displacements (units of m). Groundwater levels are in units of mm.

**Once you have read everything above, continue with the `Analysis.ipynb`**


**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.