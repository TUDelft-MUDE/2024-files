# PA 1.5: MUDE-git-crossword and Ice prediction
*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.5. Due: before Friday, October 4th, 2024.*

This PA consists of 4 parts:

1. `PA_1_5_.useful_tricks.ipynb`: a notebook covering a few simple Python topics that are especially useful for the WS and GA assignments this week.
2. [Programming for Week 1.5](https://mude.citg.tudelft.nl/2024/book/programming/week_1_5.html) (Online Textbook): read this chapter, which covers more Git topics: branching, forking, pull requests and merge conflicts
3. `crossword.md`: file containing a crossword puzzle, which you will use to work on a separate branch and merge commits into the `main` branch.
4. `predictions.txt`: a file in a _separate_ repository [github.com/iceclassic/mude-pa-1-5](https://github.com/iceclassic/mude-pa-1-5). where you will contribute a commit from your own fork of the separate repository

Notes:
- The notebook can be completed independently from the other parts; try to finish complete this before the Wednesday in-class session.
- This PA actually involves 3 repositories: the one created for this assignment, the ice repository and your own personal fork of the ice repository, which you will create.

## Ice Classic Repo

For Part 4 you will be making a prediction for the 2025 ice classic! The repo is: [github.com/iceclassic/mude-pa-1-5](https://github.com/iceclassic/mude-pa-1-5).

**To add your prediction,** find your GitHub username in the list in `predictions.txt` and enter a prediction in the format `(YYYY-MM-DD HH:MM:SS)` (year,month, day, hour minute, seconds). That's it! 

- The repo already contains a figure that displays all of the historic breakup days and times, and the predictions of MUDE students will be added on top. Here is a direct link to the figure: [predictions.svg](https://github.com/iceclassic/test_pa15/blob/main/test.svg).
- The repo is set up to automatically update the figure every time a commit is made. Note, however that it may take some time before your change is visible, as the instructors must approve the pull requests.

## Grading Criteria

You will pass this PA if:
1. For the python and markdown topics of this PA.
  - Your notebook `PA_1_5_useful_tricks.ipynb` runs without errors.
  - Your repository contains a file `my_figure.svg` located in subdirectory `/figures`
  - You update the `myfigure.md` file to include a reference to the previous figure.
2. For the branching part of this PA: you successfully merged a pull request from your own branch.
3. For the forking part of this PA: You successfully created a pull request to the ice repo with your bet (it should be visible on the [Pull Requests tab of the repo](https://github.com/iceclassic/mude-pa-1-5/pulls)). After your pull request is accepted (this requires some manual actions from your teachers on your pull request) you can confirm that your bet is approved by seeing it in the figure `guesses.svg`.

You can verify that you passed checks 1 and 2 by looking for the green circle in this repository. It runs automatically just like last week when you uploaded your notebook. Check 3 will be done manually by your teachers.


**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.