# README for Group Assignment 2.2

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.2, Friday, Nov 22, 2024.*

GA 2.2 applies the finite element method (FEM) to a 2D problem with a fun geometry.

There are several files to be aware of:

1. `README.md`
2. A notebook `the_big_M.ipynb` (equivalent to the previous `Analysis.ipynb` files)
3. A `Report.md` file for answering questions
4. File `big_M.msh` and `big_M_fine.msh`: two FEM meshes to use in the analysis. Only the first one is needed, but the second one can be tried as well by modifying a single line in the notebook. 

## Primary Task

Run the analysis in `the_big_M.ipynb`, inspect the code (really **really** read it!) and results, then answer the questions in `Report.md`.

Note that unlike other GA's, besides reading and understanding the contents, you don't have to do anything with the code in the notebook before reading the Report questions. The questions mostly focus on what is happening in the analysis. The provided finite element code is complete, but you will need to look at the details of the implementation to answer the questions. For the final question you will have to perform some additional operations in the notebook.

## Python environment

To run the code for this assignment **add a new Python package** to your `mude-base` environment:

```
conda activate mude-base
conda install ipympl
```
Remember to install _before_ running your notebook. If you did not do this, you may have to restart the kernel in your notebook.

## Task Overview

You may be able to divide work on the last two questions once the derivation and determination of boundary conditions is complete. Note that the first question in particular is essential exam practice and every group member should be able to complete it individually.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
