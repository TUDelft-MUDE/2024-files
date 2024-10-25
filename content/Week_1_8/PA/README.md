# PA 1.8: Sympy---your new best friend?

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.8. Due: before Friday, October 25th, 2024.*

_You can access this assignment with the following link: [classroom.github.com/a/SK4BFppe](https://classroom.github.com/a/SK4BFppe)._


This PA consists of 3 parts:

1. Creating a new `conda` environment `mude-week-8` to use for the PA and Week 1.8 activities using the file `environment.yml` (instructions below).
2. Reading a [page in the textbook about the Python package Sympy](https://mude.citg.tudelft.nl/2024/book/external/learn-python/book/08/sympy.html).
3. `PA_1_8_Equations_Done_Symply.ipynb`: a notebook to practice using Sympy by repeating calculations from WS 1.7.

## Grading Criteria

You will pass this PA if:
1. Your notebook `PA_1_8_Equations_Done_Symply.ipynb` runs without errors.

You can verify that you passed by looking for the green circle in this repository (the last workflow run).

If your check is failing, view the Python traceback by going to the Actions tab, click the most recent workflow run, click the job (the box diagram) and expand and read the command line interface output.

## Creating a (new) `conda` environment

For the past weeks you have been using the `mude-base` environment that we made in Week 1.1...remember that?! However, now we need _new_ packages for next week. In addition, 

If you forget what an environment is, [**read this page in the book from week 1!**](https://mude.citg.tudelft.nl/2024/book/external/learn-programming/book/environments.html).

_Why create a new environment and not just add the necessary packages to `mude-base`?_

This is not a good practice, especially if we are using the environments for specific purposes. In this case, the probability libraries we need next week rely on the C programming language to run efficiently, and when this happens it sometimes conflicts with other exiting packages in the environment. In addition, it is usually better to create an environment all at once, rather than adding packages sequentially over time.

Convinced? Let's make the environment, it's quite simple. Complete these steps and then you can test out the environment by using importing the Sympy package in the PA notebook for this week.

Step 1.

- open up a CLI on your computer (it is easiest to just do this in VS Code)
- execute the following command and read the output to make sure the process completes.

```
conda env create -f environment.yml
```

If you get an error, you may need to specify the location of `environment.yml`.

Step 2.

Once the environment is installed, check that it was completed by running this command:

```
conda env list
```

You should see `mude-week-8` in the list.

Step 3.

Use it! Select this environment when you open the notebook for PA 1.8.

The first cell should execute without error if you correctly installed the environment:

```
import sympy as sym
```

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.