# PA 2.4

_[CEGM1000 MUDE](http://mude.citg.tudelft.nl/), Time Series Analysis, Week 4 of Quarter 2._



- This week we will install the software Gurobi, which is required for optimization topic next week. This is to give you enough time to solve any problems with the installation. Next week we will learn how to use it.


This week the programming assignment is based on three files:
- `README.md`: instructions for PA12 and environment creation (this document)
- `PA_2_4_A_gurobilicious.ipynb`: set-up of Gurobi software for Week 2.5 (optimization) with a new conda environment
- `PA_2_4_B_axes_of_awesome.ipynb`: Python-related tools used in Week 2.4 for Time Series Analysis


## Instructions

Read through this file (`README.md`). This `README.md` also includes a task on creating a new conda environment. Afterwards complete the tasks in notebooks `PA_2_4_A` and `PA_2_4_B`. Note that `PA_2_4_A` will refer you to the MUDE website, where you will find [instructions for setting up the Gurobi license file UPDATE LINK](https://mude.citg.tudelft.nl/software/gurobi/).

## Grading Criteria

You will pass this PA if:
1. Your notebook `PA_2_4_B_axis_of_awesome.ipynb` runs without errors.
2. A file `license.lic` is committed to your repository
3. The license file confirms you have installed the academic license of Gurobi version 12

Note that we won't be checking your notebook A for errors, as this will not run on the autograder webserver as it does not have Gurobi installed.

## Python environments revisited

Until now, we have been able to complete our work in MUDE with a few packages like `numpy` and `scipy` in our `mude` environment, which we create and manage with `conda`. In the previous quarter you also created a new environment, 'mude-week-8'. This week, we will once again create a new environment in preparation for the week on Optimisation ahead. You may have noticed that the environments are always generated using a file called 'environment.yml'. This is just a text-based format (see below) where we can list the packages that we want to be included in our environment. We can then tell `conda` to create the environment based on the contents of the file! 

All we need to do to create an environment from a file is to write a list of what we want and then tell `conda` to read it. That's it!

#### List requirements in `*.yml` file

To write our list of requirements, we will use a file with a new (to us) file extension: the `*.yml` file (pronounced "yah-mul"). It is a text-readable file, that stands for "Yet another Markup Language." You don't need to worry about this, except to recognize that this is one of _many_ types of files that use a particular type of text formatting to give a computer specific instructions. It is very similar to the way Markdown formatting works.

Take a look at the contents of the file `environment.yml` in this repository. Can you understand what is being described? For each section (`name` and `dependencies`) you should see that it uses a colon `:` to list the information. This will be processed by `conda` when creating the new environment.

There is another special type of formatting with two colons `::`. This is how we tell `conda` to look on a specific _channel_ for the particular package. Conda channels are the locations where packages are stored; you can think of them as a specific URL web address. This is where the creator of the package can manage and maintain its distribution (e.g., publishing new versions, installation information, etc). Conda packages are downloaded from these URL's, and if you know where a particular package is stored, you can give `conda` explicit instructions. For example, we can see that Gurobi is stored on the `gurobi` channel, because the URL is `https://anaconda.org/gurobi/gurobi` (note that Anaconda is an organization that provides a wide variety of software; the website anaconda.com is used to provide documentation and information about the organization, whereas anaconda.**org** is explicitly used for package distribution).  This is specified in the environment file using the `channel::package` notation. In the `*.yml` file, `gurobi::gurobi` is equivalent to using the command `conda install -c gurobi gurobi` in Anaconda prompt.

In summary, as you can see from reading the file, we will set up an environment specifically for this assignment,  along with a number of dependency packages, two of which are installed from special conda channels.

#### Create environment from `*.yml` file

The command for creating the environment is simple. Do the following:

1. Open Anaconda Prompt (Windows) / your default terminal app (Mac)
2. Navigate to your working directory (where this file and `environment.yml` is located)
3. Execute this command: `conda env create -f environment.yml`
4. Keep reading this assignment as you wait (this may take several minutes)

Do you know why this takes so long? Because we are installing many packages at once! Keep an eye on the terminal window as this process is completed. First `conda` is collecting information about the dependencies, then it will _solve_ the environment; in other words, figure out which version of each package it should use. Once it is ready, it will present the list of packages and proceed with the "installation" (really just downloading `*.py` files and putting them in a folder on your computer) Note that the prompt may ask you to confirm that the installation should proceed, depending on your system settings. 

Once the environment is created, we can activate it, and also check that everything was installed properly. Try `conda env export -n ENV_NAME` to see what was installed by "default." The list is very long, even though we only asked for a few packages!

It is also interesting to try `conda env export --from-history` (make sure you activated it already), which shows the specific packages requested. Do you notice anything in particular when looking at the output? That's right, it's exactly the same as our file `environment.yml`! The only thing extra is that it identifies `default` as the conda channel (since we didn't specify anything else in the `*.yml` file).


## Next steps

Once you have successfully installed our new environment and you have activated it, you are ready to proceed with the rest of this assignment.

We simply presented the instructions for creating an environment in this README; if you would like to read more about this, you should refer to the [Anaconda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). You can also read about creating an environment file on the same page [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually), if interested.


**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2023 <a rel="MUDE Team" href="https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=65595">MUDE Teaching Team</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 License</a>.


