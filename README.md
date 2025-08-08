# MUDE Files 2024

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16782515.svg)](https://doi.org/10.5281/zenodo.16782515)

This repository primarily contains MUDE Files from the 2024-25 academic year. MUDE stands for Modelling, Uncertainty and Data for Engineers, a required module in the MSc programs from the faculty of Civil Engineering and Geosciences at Delft University of Technology in the Netherlands. Note that the MUDE Textbook is published separately: for the 2024-25 academic year, see [mude.citg.tudelft.nl/book/2024/credits](https://mude.citg.tudelft.nl/book/2024/credits).

The primary purpose of this repository is to share the weekly assignments and solutions: programming assignments (PA), workshops (WS) and group assignments (GA). This is almost entirely Jupyter Notebooks. We have really shared everything, work in progress and all, so while there is a tremendous amount of material here, it will certainly take some time to sort through it if you are not intimately familiar with MUDE. To help with this matter, you might want to have a look at the presentation the presentation _Enhancing Student Experience While Modernizing the Curriculum,_ available at [doi.org/10.5281/zenodo.10879193](https://doi.org/10.5281/zenodo.10879193): the target audience was TU Delft staff in other faculties, but there is a general overview of MUDE that may be helpful.

The assignments mentioned above (PA, WS, GA) are located in directory `content/`. In addition, this repository contains several other types of files:

1. Files uploaded to TU Delft webserver for student access (copies of assignment notebooks/source code, as well as readable versions of assignments converted to HTML). These are located in directory `src/` and include a number of files released under a CC BY license, but for which the source code is not available, for example: PDF copies of PowerPoint Slides and PDF copies of Exams and Solutions.
2. A directory `synced_files/` that contains automatically generated files that are Git-friendly (see below).
3. Various auxiliary files for carrying out assignment creation and release to students on TU Delft webservers (for example, the GitLab CI/CD pipeline configuration file `.gitlab-ci.yml`).

The explanation in this file is written by Robert Lanzafame, the MUDE Module Manager for the 2023-24 and 2024-25 academic years. For questions or additional information about MUDE, please get in touch with the current MUDE Team at MUDE-CEG@tudelft.nl.

## License and Referencing

Unless otherwise specified herein, all files in this repository are released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This means that you can share and adapt the material in any medium or format, even for commercial purposes, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. We ask that you use the following citation when referencing these materials:

> MUDE Team (2024) Modelling, Uncertainty and Data for Engineers (MUDE) Files. https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

A complete list of contributors is maintained in the file `CITATION.cff`, and can be used in place of "MUDE Team" in the citation above, as desired. If you wish to refer to a subset of files, please refer to a specific _topic_ in its entirety, using the following format:

> `<AUTHORS>` (2024) `<TOPIC>` Files from Modelling, Uncertainty and Data for Engineers (MUDE). https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

where fields `<AUTHORS>` and `<TOPIC>` are replaced with the appropriate values as described in the sections herein. For example, if you wish to refer to the files for Week 2.8, you would use:

> Lanzafame, Robert (2024) Risk Analysis Files from Modelling, Uncertainty and Data for Engineers (MUDE). https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

Due to the complex nature of MUDE and the large size of the MUDE Team, the metadata associated with this release of files is likely to require correction; therefore versioning is used to document updates to the source code. The version number has format `v2024.B.C` where minor corections will advance C and more significant updates (e.g., adding missing authors, updating licensing information, etc.) will advance B. The "2024" part of the version number will never change, as this will help distinguish releases in future years, as well as to be comparable with the MUDE Textbook, which uses a similar versioning scheme (e.g., MUDE Textbook and MUDE Files from the 2024-25 academic year are both versioned as `v2024.B.C`). A complete list of versions can be found in the GitHub Releases page or the Zenodo page. As we are not editing the contents of the files, only the metadata associated with them, we do not anticipate that it is necessary to include the version number in the citation recommended above.

### How to View and Use the Files Herein

The easiest way to vew the contents of this repository is to clone it and use an IDE that is capable of rendering Jupyter notebooks, HTML, Markdown and Python files. For example, Visual Studio Code or Jupyter Lab.

There are also several ways to view the files without cloning the repository, using a conventional web browser:

1. GitLab/GitHub will automatically render Jupyter notebooks; however, note that the assignments use a number of Markdown and HTML formats that were meant for viewing on VS Code or Jupyter Lab, and may not render correctly in a web browser.
2. The original "Files Page" shared with students during the 2024-25 academic year is available at [mude.citg.tudelft.nl/files/2024](https://mude.citg.tudelft.nl/files/2024) (though it may not be maintained indefinitely).
3. A GitHub Pages site may be set up to view the files in a similar way to the original Files Page _(this is not yet implemented)._

The notebooks were intended to be executed using a Conda environment created during week 1 of the module; a specification can be found at `./content/Week_1_1/environment.yml`. Note that packages were added sequentially in each week on an as-needed basis, so you may need to add packages manually. In some cases (e.g., Week 2.5 Optimization or Week 1.8), a new package was created for that particular week, and instructions can typically be found in the various assignment files.

### Acknowledgements

In addition to the generous support of the faculty of Civil Engineering and Geosciences at Delft University of Technology (as also described on the [Credits page of the MUDE Textbook](https://mude.citg.tudelft.nl/book/2024/credits)), a tremendous thank you goes out to our MUDE students for their role in the constant refinement of MUDE materials. In particular: the first year students whose constant feedback helped influence the redesign of MUDE for year 2 (2023-24); as well as the many MUDE teaching assistants: for those in the first year (2022-23) who, like the MUDE teachers themselves had no idea what MUDE was (yet!); and the brave students from the first and second years who were motivated to help teachers make MUDE even better for the next generation of students. Finally, a special thank you goes to the colleagues who overcame fears and challenges associated with the "new" (to us) way of working with Jupyter ecosystem: your willingness to learn and adapt to improve your teaching is truly inspiring!

## Overview of MUDE Topics

This is a list of topics, each of which has its own subsection herein containing lists of authors, contributors and materials that are _not_ included in the CC BY license of this repository, along with any additional information that may be relevant. The topics are listed here in weekly order as they were covered during the 2024-25 academic year. The numbering "Week Q.W," where "Q" refers to Quarter 1 or 2, each of which has 8 weeks of instruction ("W"):

- Week 1.1: Modelling Concepts
- Week 1.2: Uncertainty Propagation
- Week 1.3-1.4: Observation Theory
- Week 1.5-1.6: Numerical Modelling
- Week 1.7-1.8: Univariate and Multivariate Continuous Distributions
- Week 2.1: Finite Volume Method
- Week 2.2: Finite Element Method
- Week 2.3: Signal Processing
- Week 2.4: Time Series Analysis
- Week 2.5: Optimization
- Week 2.6: Machine Learning
- Week 2.7: Extreme Value Analysis
- Week 2.8: Risk Analysis
- Programming Assignments (all weeks)

Note that a Programming Assignments (PA) is provided each week, but are collectively considered a unique topic. This has to do with the way programming is integrated into the other topics of MUDE; for a pedagogical discussion of this (as well as a general overview of MUDE), see the presentation _Enhancing Student Experience While Modernizing the Curriculum,_ available at [doi.org/10.5281/zenodo.10879193](https://doi.org/10.5281/zenodo.10879193).

## Synchronization Script

A synchronization script `sync_notebooks.py` was developed to overcome the challenges of using Jupyter notebooks in a Git repository. Jupyter notebooks stored in the `content/` directory are automatically processed to remove code outputs and the remaining Markdown and Python code cells can be converted to several formats: `*.HTML`; `*.md` files with code in formatted code blocks; `*.py` files with Markdown formatted as commented lines of code. These files were not intended for direct editing, as they were meant produce easy to read Git diffs for edited notebooks, as well as produce file formats that are easier to view and read when a notebook rendering software is not available (e.g., HTML in a web browser or downloading Markdown files to view in a text editor). 

The synchronization script is _not_ released as part of the CC BY license of this repository. It is, however, shared under an MIT License, and can be referenced as follows:

> Lee, Kwangjin and Lanzafame, Robert (2024) Synchronization script for MUDE Files (v1.0.0). https://github.com/TUDelft-MUDE/2024-files. MIT License.

This includes files: `README_sync.md`, `sync_notebooks.py` and `config_sync.yml` and associated output files.

See file `README_sync.md` for more information about the synchronization script (unfortunately it is not yet completely documented).

## Technical Details

This Git repository is stored on TU Delft GitLab ([gitlab.tudelft.nl/mude/2024-files](https://gitlab.tudelft.nl/mude/2024-files)) and GitHub ([github.com/TUDelft-MUDE/2024-files](https://github.com/TUDelft-MUDE/2024-files)).

The materials released herein were originally developed using the TU Delft instance of GitLab (`gitlab.tudelft.nl`), with a TU Delft Ubuntu server used for CI/CD pipelines and serving files to students (Files, as well as the MUDE Textbook and website). Beginning with the 2025-26 academic year, the MUDE Team will be using GitHub and GitHub pages for storing source code and serving files and website to students. Although the original GitLab repository and TU Delft server was operational at the time of writing (August, 2025), its continued maintenance is not guaranteed. Therefore, the files herein are also published on GitHub under the Organization TUDelft-MUDE.

A Zenodo record is linked to this repository to provide "permanent" storage of the source code, and is (nearly) automatically updated every time a new release is created on GitHub. Whenever a new release is made, the text description in the Zenodo record should be updated manually by copying and pasting the contents the of this README.

To make a correction to this repository, do the following:

1. Make a commit and push it to GitHub _and_ GitLab (two upstream remotes are configured; Google it, there are plenty of how-to's).
2. Use GitHub to create a new release and tag (note that the tag will not automatically go to GitLab, which is OK).

## MUDE Topics in Detail

Unless otherwise noted, authors typically were teachers, PhD's and postdocs that played a significant role in designing and developing the materials, whereas contributors played a valuable role, albeit a relatively minor one, for example: reviewing materials, suggesting case studies, formatting digital files and (especially for teaching assistants) giving valuable feedback as to the appropriate level of difficulty and quality of the assignments.

### Week 1.1: Modelling Concepts

Authors: 

Contributors: 

Materials not included in CC BY license:

_WORK IN PROGRESS_

### Week 2.3: Signal Processing

Authors:

Contributors:

Note that the source code for the solution for GA 2.3 is not included here, as the intention is to use the same analysis in future years; a PDF in the `src/` directory provides results from the solution without the source code.

Materials not included in CC BY license: