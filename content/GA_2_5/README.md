# Project 9 Report: Optimization

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

The focus of this assignment is on applying two different methods for optimization on a road network design problem. Part of the background material in the notebooks was already available in [Chapter 5.11 of the textbook](https://mude.citg.tudelft.nl/book/optimization/project.html).

## Overview of material

- `README.md` (this file)
- `Report.md`, primary file in which you write your answers to some questions and eventually copy plots from the notebooks. Typically, a short, one-line answer is sufficient, but please include a short reasoning, justification or argumentation. Remember to use Markdown features to clearly indicate your answers for each question below.
- `P09A-Dont_do_math_and_drive.ipynb`, the notebook in which you apply a mixed integer linear program (MILP) to a road network design problem (NDP).
- `P09B-Evolve_and_drive.ipynb`, the notebook in which you apply a genetic algorithm (GA) to the same road network design problem (NDP).
- `environment.yml`, for creating a Python environment
- a subdirectory `./utils` containing some functions for visualization (which you don't need to open).
- a subdirectory `./figs` containing some figures included in the notebooks (which you don't need to open).
- a not-yet existing subdirectory `./input`, which should include the datasets for this problem, which are not included in the Git repository. **Download the data [using this link](https://surfdrive.surf.nl/files/index.php/s/Rmw7BDnatHv2VYR/download)** and make sure you save those files inside a subdirectory `./input`. Once unzipped, you can copy the `input` directory so that files, for example is `./Project_9/input/TransportationNetworks/SiouxFalls/*.tntp`, where `Project_9` is the repository/directory where the notebooks are located.
- a `.gitignore` file preventing your imported data in the `./input` subdirectory being pushed to Gitlab.

### Python Environments

You can run all of the notebooks for today in the environment `mude-opt` which you've been using during the workshop this week (WS13). In particular, it includes the optimization packages `pymoo` and `gurobipy`. The non-Python part of the software Gurobi was installed as part of `PA12`.

The `*.yml` file included in this repository is the same as that used for WS13 this week, so you can re-use the same Conda environment (e.g., `conda activate mude-opt`). Here are a few tips to remember when using Anaconda prompt:

- Review your existing environments with `conda info --envs`
- Create an environment with command `conda env create -f environment.yml`
- For the activated environment, check the packages explicitly requested with `conda env export --from-history`
- If you want to create a new environment from the `*.yml` file, but the name already exists, simply change the name in the file using a text editor

## Submission and deadline

- Submit your answers, together with any relevant plots, in the Markdown file `Report.md`. This is the primary document that will be used to determine your grade; however, the auxiliary files (e.g., `*.ipynb` or `*.py` files) may be checked in case something is not clear.
- The deadline is to submit your work by making commits to your Group's GitLab repository by Friday at 12:30h.
- This project will be graded on interpretation, application, documentation and programming.

## Repository, Formatting and Static Check

There is no static check for this project. Be sure to leave the outputs from your code cells in your `*.ipynb` file so that they are readable.

You are always expected to provide well-formatted figures and Markdown text in your `Report.md` file, as well as logically organize any auxiliary files you may use (e.g., try to put your figures in a sub-directory, if there are a lot of them). If you run out of time it is OK if your `*ipynb` files do not run.

## Backup data links

Sometimes the download links reach a maximum limit. If the link above no longer works, try one of these:
- [Backup link 1](https://surfdrive.surf.nl/files/index.php/s/StqaFtNDg6DNR4a/download)
- [Backup link 2](https://surfdrive.surf.nl/files/index.php/s/tC56Rpbhd7WpN9k/download)

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
