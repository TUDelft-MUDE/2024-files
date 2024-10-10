# PA 1.6: Boxes and Bugs
*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.6. Due: before Friday, October 11th, 2024.*


_You can access this assignment with the following link: [classroom.github.com/a/WT3kcHGt](https://classroom.github.com/a/WT3kcHGt)._

This PA consists of 4 parts:

1. [Programming for Week 1.6](https://mude.citg.tudelft.nl/2024/book/programming/week_1_6.html) (Online Textbook): read this chapter, which covers errors and error handling in Python.
2. `PA_1_6_Boxes_and_Bugs.ipynb`: a notebook covering a few simple Python topics that are especially useful for the WS and GA assignments this week.
3. Python file `script_test.py`: prints a simple statement to your CLI to confirm you have VS Code set up properly for executing Python scripts (instructions below).
4. Two additional `*.py` files (beginning with `script_`), each of which contains some code with a few bugs that you must find and solve using the Python traceback that is generated in the CLI after running them.
5. An `auxiliary_files/` subdirectory containing figure and a `*.csv` file to accompany the assignment file.

## Running Python Scripts in VS Code

So far we have mostly been using Jupyter notebooks, with a few examples of importing functions using `*.py` files. However, it is important to recognize that **Jupyter notebooks are not the only way to run Python code.** With your MUDE setup of conda and VS Code it is very easy to execute the contents of a `*.py` file directly, with output being generated in the command line interface. This workflow is called _scripting_ and the contents of the `*.py` files are referred to as scripts. 

Try running a script by opening `script_test.py` in the editor and clicking the triangular "Run Python files" button in the top right corner. You should see a simple message printed in the CLI. If this works, you are ready to read the Python traceback in the CLI and debug the other `*.py` files in the repo and complete the PA.

Note that if you get stuck in the Python interpreter in your CLI, you can type `exit()` to get back to the native CLI prompt (e.g., `cmd` if you are using Windows).

If this does not work, ask an instructor for help during question hours.

## Grading Criteria

You will pass this PA if:
1. Your notebook `PA_1_6_Boxes_and_Bugs.ipynb` runs without errors.
2. All of the Python scripts in your repository run without errors.

You can verify that you passed both checks by looking for the green circle in this repository (the last workflow run).

If your check is failing, view the Python traceback by going to the Actions tab, click the most recent workflow run, click the job (the box diagram) and expand and read the command line interface output.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.