# Project 7: Signal Processing

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*


## Project 7 Info

The focus of this assignment is on signal processing, and specifically on spectral analysis (i.e. analysis of signals in the frequency domain).

**Your primary objective is to complete all tasks in the notebook `P7.ipynb` and summarize your answers in `Report.md`.**

### Overview of material

- this `README.md` with instructions
- `Report.md`, primary notebook in which you are supposed to copy your plots and include your answers to the questions (typically a short, one-line answer each time; though a simple 'yes' or 'no' is not sufficient, please include a short reasoning, justification or argumentation).
- `P7.ipynb`, the secondary Jupyter notebook with description and tasks, to be used for actual coding
- `cantileverbeam_acc50Hz.csv`, data file with acceleration measurements from the cantilever beam (tasks 7-9)
- `CSIRO_Alt_seas_inc.txt`, data file with Global Mean Sea Level measurements (task 10; optional)
- [Cantilever Beam Experiment Video](https://youtu.be/o4moRwvlBLU?si=aKelBMWm3HB2Of26): a short supplementary one minute movie illustrating the cantilever-beam experiment.

We will use `pandas` to import the dataset, so make sure this is included in your `mude` environment before starting Jupyter Lab.

## Grading

This project will be graded with the following weights for each assessment category:

* 0.40 for **Interpretation** and 0.40 for **Application**
* 0.10 for **Documentation** and 0.10 for **Programming**.

All 10 tasks are weighted equally (1 point each) and the maximum grade is 10 (points). Task 10 is optional; each group will receive 1 point by default for this one.

### Submission and deadline

- Submit your answers, together with any relevant plots, in the Markdown file `Report.md`. This is the primary document that will be used to determine your grade; however, the auxiliary files (e.g., `*.ipynb` or `*.py` files) may be checked in case something is not clear.
- The deadline is to submit your work by making commits to your Group's GitLab repository by Friday at 12:30h.

### Repository, Formatting and Static Check

There is no static check for this project. Be sure to leave the outputs from your code cells in your `*.ipynb` file so that they are readable.

You are always expected to provide well-formatted figures and Markdown text in your `Report.md` file, as well as logically organize any auxiliary files you may use (e.g., try to put your figures in a sub-directory, if there are a lot of them). If you run out of time it is OK if your `*ipynb` files do not run.

**Importing figures into a Markdown file:** in Project 5 and 6 we noticed a lot of figures not rendering properly in the Markdown file (i.e., broken image links); this was typically due to spaces in the file name. To avoid rendering issues in your Markdown reports, we recommend *not* including spaces in the filename as a general rule (e.g., `my_image.png` instead of `my image.png`). If you need to include an image in Markdown with a space in it, replace the space with `%20`. For example, to show file `my image.png` you should use: `![My image](my%20image.png)`

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
