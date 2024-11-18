# Project 8: Time Series

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/).*

The focus of this assignment is to use time series analysis techniques to evaluate a data set of global mean sea level, with a special focus on capturing auto-correlation and a periodic signal in the model.

### Overview of material

- `Report.md`, primary notebook in which you are supposed to copy your plots and include your answers to the questions (typically a short, one-line answer each time; though a simple 'yes' or 'no' is not sufficient, please include a short reasoning, justification or argumentation).
- `P8.ipynb`, the Jupyter notebook with tasks, to be used for actual coding
- `CSIRO_Alt_seas_inc.txt`, data file with Global Mean Sea Level measurements

You can complete this assignment using your `mude` environment.

### Submission and deadline

- Submit your answers, together with any relevant plots, in the Markdown file `Report.md`. This is the primary document that will be used to determine your grade; however, the auxiliary files (e.g., `*.ipynb` or `*.py` files) may be checked in case something is not clear.
- The deadline is to submit your work by making commits to your Group's GitLab repository by Friday at 12:30h.
- This project will be graded on interpretation, application, documentation and programming. Each task is worth roughly one point each.

### Repository, Formatting and Static Check

There is no static check for this project. Be sure to leave the outputs from your code cells in your `*.ipynb` file so that they are readable.

You are always expected to provide well-formatted figures and Markdown text in your `Report.md` file, as well as logically organize any auxiliary files you may use (e.g., try to put your figures in a sub-directory, if there are a lot of them). If you run out of time it is OK if your `*ipynb` files do not run.

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

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.
