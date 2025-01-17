# README for GA 2.8: [Ice Ice Baby](https://www.youtube.com/watch?v=rog8ou-ZepE)

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 2.8, Friday, Jan 16th, 2025.*

The focus of this assignment is risk analysis.

You will evaluate the distribution of profit for the Ice Classic, generate a risk curve, then determine what an appropriate bet could be.

Remember that the Ice Classic is betting game where you buy a ticket and guess the **minute** at which the ice in the Nenana River in Alaska breaks. Everyone with the winning ticket splits the prize money, which is typically around 300,000 USD! Remember that each ticket costs $3.

## Overview of material

There are a number of files in this assignment:

- `README.md` (this file)
- `Analysis_01.ipynb`
- `Analysis_02.ipynb`
- `Analysis_03.ipynb`
- `Report.md`
- `ticket.md`: the file to specify the official Ice Classic Bet for your group (this file will be automatically processed, so follow the instructions carefully!).
- `tools.py`
- `tickets_per_minute.pkl` (in subdirectory `pickles/`)
- `.gitignore` (to avoid committing large `*pkl` files to your repo)
- The [online Solution from GA 1.1](https://mude.citg.tudelft.nl/2024/files/Week_1_1/), which is a PDF that contains a **lot** of good summary information about the Ice Classic that will be useful for the end of the GA, especially where you need to consider a "future" bet.

Note in particular that the file `tools.py` contains a _lot_ of poorly documented code. It can do a lot, but only Robert and Gabriel have any idea how to use it, so find them if you want to do something special. Otherwise, everything you need for the GA is illustrated somewhere in the notebooks. 

## Python Environments

There is a good chance that you will get a "package not installed" error. Don't worry, you are more than capable of fixing this by running a notebook cell with `!pip install <package-name>`. Or by opening up a CLI and running `pip install...`, or `conda install...` ... you get the idea.

## Task Overview

Notebooks 1 and 2 have some calculations that you need to complete, but the majority of it is illustrating to you how to use the tools available for making the visualizations and calculations easy. Notebook 3 is a short one, but it is coupled with the report writing.

It's probably best if you work with each other to recap what the key details of the Ice Classic are about, and study the PDF of the solution from Week 1.1 together to remember some key points.

Once you look at the PDF, you can have some group members work on Questions 2 and 3 in the report while the others learn to use the Python tools in Notebooks 1 and 2.

Questions 3 and 4 in the report are the most important for this assignment. They are weighted roughly 2-3 times more than the others.

And don't forget to submit a bet for your ticket!

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.
