# Project 10 Report: Machine Learning

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

The focus of this assignment is to use a neural network to create a surrogate model of a pipe network that predicts pressures as a function of pipe diameter. The context is that the model that is used to compute pressures in the system is time-consuming to run, so we would like to train a neural network to make calculations faster (i.e., use the surrogate model).

## Working Remotely

For this week, we recognize that many may be working remotely. As such, we've made the project files available in advance to allow you all the ability to work on it beforehand and make arrangements with your group-mates to work effectively and cooperatively. Teachers will still be present, in-person, on Friday and it is still expected that the project report be submitted at the end of the session Friday. If there are any announcements to be made, we will add them to presentation, which you can view remotely at [this link](https://tud365.sharepoint.com/:p:/s/MUDE/EcZ-L1gD2ABEhlOb0p83OGIBl4N-Jr4OqP2TRNRL9CEYiQ?e=3Uuueo).

## Overview of material

- `README.md` (this file)
- `Report.md`, primary file in which you write your answers to questions.
- `P10.ipynb`, the notebook in which you will build and evaluate the neural network.
- a subdirectory `./figs` containing some figures included in the notebooks (which you don't need to open).
- a `.gitignore` file preventing your data being pushed to Gitlab.

**You need to download the data for this project!** Two data files are needed: `features_BAK.pk` and `targets_BAK.pk`, which should added to your repository in a subdirectory `./data`, which is located in the same directory as `P10.ipynb`. **Download the data [using this link](https://surfdrive.surf.nl/files/index.php/s/UmjdZCAAlbvaKRO/download).** 


### Python Environments

The environment you used earlier in the week will be sufficient, as long as it has `sklearn` and the standard packages used elsewhere in MUDE.

## Submission and deadline

- Submit your answers, together with any relevant plots, in the Markdown file `Report.md`. This is the primary document that will be used to determine your grade; however, the auxiliary files (e.g., `*.ipynb` or `*.py` files) may be checked in case something is not clear.
- The deadline is to submit your work by making commits to your Group's GitLab repository by Friday at 12:30h.
- This project will be graded on interpretation, application, documentation and programming.

## Repository, Formatting and Static Check

There is no static check for this project. Be sure to leave the outputs from your code cells in your `*.ipynb` file so that they are readable.

You are always expected to provide well-formatted figures and Markdown text in your `Report.md` file, as well as logically organize any auxiliary files you may use (e.g., try to put your figures in a sub-directory, if there are a lot of them). If you run out of time it is OK if your `*ipynb` files do not run.

## Backup data links

Sometimes the download links reach a maximum limit. If the link above no longer works, try one of these:
- [Backup link 1](https://surfdrive.surf.nl/files/index.php/s/E2FOutaHes7gm6Z/download)
- [Backup link 2](https://surfdrive.surf.nl/files/index.php/s/OQb1kpbrND3NPqg/download)

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
