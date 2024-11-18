# PA13 Information, Week 2.5

_[CEGM1000 MUDE](http://mude.citg.tudelft.nl/), Optimization, Week 5 of Quarter 2._



updates for 2024:
- environment stuff may be reduced
- need to address optimizing the model and adding constraints
- should probably also address the "running notebook cells out of order" issue directly
- can remove pandas part if the rest is too big











This week the programming assignment will illustrate some good practics for managing data sets in a git repostiroy. It is based on two files:
- `README.md`: environment deletion and combining datasets with Git (this document).
- `PA13_data_process.ipynb`: a brief intro to pandas to help us process and manage our data.

## Instructions

Read through this file (`README.md`) on how to delete last week's environment. Then complete the tasks in notebook `PA13`.

## Deleting `conda` python environment

Last week, we looked at how to create environments and added `mude-PA12` next to your existing `mude` environment. Although the environments do not interfere with each other, you might want to delete environment you don't use, for example `mude-PA12` from last week. Why? Try running the following in a Terminal or Anaconda prompt, which will list the file locations of your environment. Then find the folder and check how big they are:

```
conda info --envs
```
You probably found that the environments can get very large: the environment for `PA12` is around 2 GB!!!! It is a good idea to remove environments once you know they are no longer useful (for example `mude-PA12`).

To remove an environment, in your Anaconda Prompt, run:

```
conda remove --name MYENV --all
```
With your environment name for `MYENV`. You will be asked to confirm the deletion, then it may take a while to remove all of the files. To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run:

```
conda info --envs
```

The environments list that displays should not show the removed environment.

This information has been taken from the [Anaconda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment).

Now you can delete any other useless environments you have lying around on your computer!

## Using data sets with Git

As you may recall, git is able to track changes to text-based files; it does this by literally tracking every single character and every line in the file. This is generally not a problem for most software and code-based projects, as the file size does not get too large. For example, the really cool "big M" finite element mesh, as well as all of the code to implement the finite element method took less than 29 KB of hard disk space. That's over 1000 times smaller than a photo on your cell phone!

However, as you may have noticed, the data sets that we use in some of our projects can become large (say, 10's of MB). If we make small changes to these files, git will store a record of the file in the changed state, and the disk space required for the repository starts to grow. For very large datasets (say, 100's of MB or GB and TB) it is impractical to use git to track these files.

In fact, if your data is truly a set of observations that was made once, under a very specific set of conditions, it should have no need to be tracked at all by git (since the data will not, and _should_ not, change). In this case it makes sense to use other platforms to save and preserve data. For example, cloud storage systems, backup hard disks, etc.

Although it may seem like you don't edit the data file directly, sometimes when a piece of software accesses a file (for example, importing it into your Juypter notebook), the file system or git _thinks_ that the file has been changed. This often causes git to keep a new snapshot of the file, taking up valuable disk space. As you make more and more commits to the repository, you are saving an unnecessary number of extra "snapshots" of the file, even though none of the information stored inside it has changed.

In summary, the disadvantages of including large files in your git repository are:
- it takes a long time to push/pull from origin
- it takes a long time to switch branches or move to different commits
- disk space is used unnecessarily to track duplicate versions of the same file

This programming assignment will walk you through one recommended workflow for managing data in your projects and preventing your repository from getting too large. In particular, we will:
- add a dataset to our working directory
- ignore the dataset with a `.gitignore` file
- process the dataset into a more usable form
- save and ignore the new data file
- commit the code we used to create the file, but not the file itself

We will use (and learn about) the package `pandas`. To accomplish this, proceed with the instructions in the notebook file for `PA13`.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.


