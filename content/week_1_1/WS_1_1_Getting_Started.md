# Workshop 1.1: Getting Started!

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/2024): Week 1.1. September 4, 2024.*

This workshop guides you through the setup of your first MUDE `conda` environment, which we will call `mude-base`. You will use this environment for programming activities for much of Q1, so it is important to get it set up and be comfortable with it before the end of the Wednesday session.


Once your software is installed (as described in the book and `README.md` file) and you have confirmed that you can run `conda` from your Command Line Interface (CLI), do the following:

1. Create a **working directory** for your Week 1.1 files on your computer.
2. Download this repository as a zip file. Unzip it and copy the files to your **working directory** (ask a teacher for help if you can't do this).
3. Open a CLI and use it to navigate to your working directory (see note below).
4. Create a `conda` environment by executing the command `conda env create -f environment.yml` (see note below)
5. Activate environment via the CLI by executing `conda activate mude-base`
6. Confirm that Python was installed correctly by checking that `python --version` returns 3.12 (you may need to try `python3 --version`)

Once you have completed these steps, you are ready to proceed with the Programming Assignment, PA 1.1 in file `PA

## Creating a `conda` Environment

For Windows OS users you are advised to use the Anaconda Prompt. For Mac OS users, you should be able to use `conda` in the Terminal application.

You should follow the CLI as `conda` resolves the environment, which will take a few minutes. After it has selected the proper set of packages, it will ask if you want to proceed: choose "yes"!

## Navigating to your Working Directory

The essential command is `cd FILE_PATH`, which will navigate the terminal window to the directory at `FILE_PATH`. Note that although `cd` works on both Windows and Mac, but many CLI commands are different between the two OS's; visit the [CLI page in the textbook](https://mude.citg.tudelft.nl/2024/book/external/learn-programming/book/install/common/cli.html) for tips.

On Windows:
1. Open the folder you wish to use as your working directory in the File Explorer
2. In the bar near the top you will see the folder address; you may need to click in the window to see the file path (i.e., the tree of folders separated by _backward_ slashes, `\`)
3. Select the entire path and copy it (`CTRL+C` or right-click, "copy address as text")
4. Open a CLI: Anaconda Prompt (or Windows Command Prompt)
5. You can now paste the file path into the CLI after the `cd ` command (don't forget the space; it should look like this: `cd C:\...`)
6. Confirm you are in the right place by inspecting the path listed in your CLI prompt (it should typically start with be `C:\ ...` and end with `>`)
7. You can also inspect the contents of the directory by executing `dir`

On a Mac :
1. Find the folder you wish to use as your working directory
2. Right-click and select **Get Info**
3. Hold down the **Alt (Option)** key; the menu items will change\
4. Select **Copy ... As Pathname**
5. You can now paste this path into the Terminal after the `cd ` command
6. Confirm you are in the right place by executing the command `pwd`
7. You can also inspect the contents of the directory by executing `ls` (that's a lower case "L")
8. Go [here](https://www.groovypost.com/howto/copy-a-file-path-on-a-mac/) to see the source of this tip, along with screenshots.

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.