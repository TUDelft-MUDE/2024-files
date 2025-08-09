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

Unless otherwise specified herein, all files in this repository are released under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This means that you can share and adapt the material in any medium or format, even for commercial purposes, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made.

In general, we ask that you use the following citation when referencing these materials:

> MUDE Team (2024) Modelling, Uncertainty and Data for Engineers (MUDE) Files. https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

A complete list of contributors is maintained in the file `CITATION.cff`, and can be used in place of "MUDE Team" in the citation above, as desired. If you wish to refer to a subset of files, please refer to a specific _topic_ in its entirety, using the following format:

> `<AUTHORS>` (2024) `<TOPIC>` Files from Modelling, Uncertainty and Data for Engineers (MUDE). https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

where fields `<AUTHORS>` and `<TOPIC>` are replaced with the appropriate values as described in the sections herein. For example, if you wish to refer to the files for Week 2.8, you would use:

> Lanzafame, Robert (2024) Risk Analysis Files from Modelling, Uncertainty and Data for Engineers (MUDE). https://github.com/TUDelft-MUDE/2024-files. CC BY 4.0 License. [doi: 10.5281/zenodo.16782515](https://doi.org/10.5281/zenodo.16782515).

Due to the complex nature of MUDE and the large size of the MUDE Team, the metadata associated with this release of files is likely to require correction; therefore versioning is used to document updates to the source code. The version number has format `v2024.B.C` where minor corections will advance C and more significant updates (e.g., adding missing authors, updating licensing information, etc.) will advance B. The "2024" part of the version number will never change, as this will help distinguish releases in future years, as well as to be comparable with the MUDE Textbook, which uses a similar versioning scheme (e.g., MUDE Textbook and MUDE Files from the 2024-25 academic year are both versioned as `v2024.B.C`). A complete list of versions can be found in the GitHub Releases page or the Zenodo page. As we are not editing the contents of the files, only the metadata associated with them, we do not anticipate that it is necessary to include the version number in the citation recommended above.

### Using MUDE Materials with Your Students

We understand that it can be distracting to students to include information about licenses and references in the materials you share with them. To avoid such distractions, yet still conform to the terms of use described above, we recommend that you do the following:

1. On any file that is derived from MUDE Files, include a small note such as: "MUDE (2024)" or "Made with MUDE Materials (2024)," depending on the extent to which materials have been reused.
2. On the course syllabus, website or within other general course information, include the reference (as described above), as well as an explanation about which files were used and how they were modified (if at all). 

In any application, if you find it useful to include a hyperlink, use the doi, as it is the most stable long-term: `https://doi.org/10.5281/zenodo.16782515` (often formatted with this abbreviated text `doi: 10.5281/zenodo.16782515`).

Finally, if you use MUDE materials, please let us know! We are sharing them with an open license because we think they are useful, but we also hope that it will encourage others to do the same. Pull requests or emails (MUDE-CEG@tudelft.nl) are always welcome.

### How to View and Use the Files Herein

The easiest way to vew the contents of this repository is to clone it and use an IDE that is capable of rendering Jupyter notebooks, HTML, Markdown and Python files. For example, Visual Studio Code or Jupyter Lab. Final versions of most assignments and their solutions (Jupyter notebooks and instructions in `README.md` files) can be viewed using the "Files Page," which is how files were shared with students during the 2024-25 academic year. This is preserved in two places online:

1. The original "Files Page"  is available at [mude.citg.tudelft.nl/2024/files/2024](https://archive.mude.citg.tudelft.nl/2024/files/) (though it may not be maintained indefinitely).
2. A GitHub Pages site has been set up to view the files, similar to the original "Files Page". It is available at: [mude.citg.tudelft.nl/2024-files](https://mude.citg.tudelft.nl/2024-files) (the setup of this site is described in the "Technical Details" section below).

Many files in `content/` are not included in the "Files Pages" above. If you do not want to clone the repository, all source files can also be explored via the file system viewer of the GitHub repository at [github.com/TUDelft-MUDE/2024-files](https://github.com/TUDelft-MUDE/2024-files) or the TU Delft GitLab repository at [gitlab.tudelft.nl/mude/2024-files](https://gitlab.tudelft.nl/mude/2024-files).These websites will automatically render Jupyter notebook files; however, note that the assignments use a number of Markdown and HTML formats that were meant for viewing on VS Code or Jupyter Lab, and may not render correctly in the browser.

For all cases, note: source code for all files is in the `content/` directory; rendered files are in `src/`, in particular `src/students/` contains the files available on "Files Pages" listed above; subdirectory `src/teachers/` was used to share solutions with in-class teachers, protected by a password, prior to release to students (this subdirectory does not contain files like lecture slides or exams).

The Jupyter notebooks were designed to be executed by students on their personal computers using a Conda environment created during week 1 of the module and used regularly by students throughout the semester; a specification can be found at `./content/Week_1_1/environment.yml`. Note that packages were added sequentially in each week on an as-needed basis, so you may need to add packages manually for some assignments. In some cases (e.g., Week 2.5 Optimization or Week 1.8), a new environment was created for a particular week, and instructions can typically be found in the various assignment files.

### Acknowledgements

In addition to the generous support of the faculty of Civil Engineering and Geosciences at Delft University of Technology (as also described on the [Credits page of the MUDE Textbook](https://mude.citg.tudelft.nl/book/2024/credits)), a tremendous thank you goes out to our MUDE students for their role in the constant refinement of MUDE materials. In particular: the first year students whose constant feedback helped influence the redesign of MUDE for year 2 (2023-24); as well as the many MUDE teaching assistants: for those in the first year (2022-23) who, like the MUDE teachers themselves had no idea what MUDE was (yet!); and the brave students from the first and second years who were motivated to help teachers make MUDE even better for the next generation of students. Finally, a special thank you goes to the colleagues who overcame fears and challenges associated with the "new" (to us) way of working with Jupyter ecosystem: your willingness to learn and adapt to improve your teaching is truly inspiring!

Rok Stular developed the original web server and CI/CD pipeline concepts for the 2023-24 academic year. These were completely revised and set up for the 2024-25 academic year by Kwangjin Lee.

Tom van Woudenberg gets a special mention for his thorough review of every file in all of the weeks he was involved in MUDE as a teacher, which was most of them (far above average). Robert Lanzafame was responsible for implementing many of the changes in year 2 that gave MUDE its unique (albeit complicated) character: for example, a consistent notebook structure, assignment types and use of open source websites. Sandra Verhagen is a hero for leading the MUDE Team as Module Manager of year 1: keeping over 50 colleagues on track to deliver a brand new module based on Jupyter notebooks and Python when very few of us had any experience with it, during a year where the entire curriculum was new, all while keeping students satisfied and producing overall positive results---miraculous.

As it is impossible to indicate the contributions of all MUDE TA's individually, they are listed here (with some exceptional contributions listed below):

- Anastasios Stamou
- Antonio Magherini
- Berend Bouvy
- Caspar Jungbacker
- Daniel van der Hoorn
- Gabriel Follet
- Guilherme Ferreira Sêco de Alvarenga
- Isabel Slingerland
- Jelle Knibbe
- João Moura Pereira de Lucas Teixeira
- Kwangjin Lee
- Max Guichard
- Mona Devos
- Nora Kovacs
- Renat Piscorschi
- Rok Stular
- Sophie Keemink
- Thirza Feenstra

## Overview of MUDE Topics

This is a list of topics, each of which has its own subsection herein containing lists of authors, contributors and materials that are _not_ included in the CC BY license of this repository, along with any additional information that may be relevant. The topics are listed here in weekly order as they were covered during the 2024-25 academic year. The numbering "Week Q.W," where "Q" refers to Quarter 1 or 2, each of which has 8 weeks of instruction ("W"):

- Week 1.1: Introduction to MUDE and Modelling Concepts
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

## Technical Details for the Respository

This Git repository is stored on the TU Delft GitLab ([gitlab.tudelft.nl/mude/2024-files](https://gitlab.tudelft.nl/mude/2024-files)) _and_ GitHub ([github.com/TUDelft-MUDE/2024-files](https://github.com/TUDelft-MUDE/2024-files)).

The materials released herein were originally developed using the TU Delft instance of GitLab (`gitlab.tudelft.nl`), with a TU Delft Ubuntu server used for CI/CD pipelines and serving files to students (Files, as well as the MUDE Textbook and website), all set up and maintained by Kwangjin Lee. Beginning with the 2025-26 academic year, the MUDE Team will be using GitHub and GitHub pages for storing source code and serving files and website to students. Although the original GitLab repository and TU Delft server was operational at the time of writing (August, 2025), its continued maintenance is not guaranteed. Therefore, the files herein are also published on GitHub under the Organization TUDelft-MUDE.

A Zenodo record is linked to this repository to provide "permanent" storage of the source code, and is (nearly) automatically updated every time a new release is created on GitHub. Whenever a new release is made, the text description in the Zenodo record should be updated manually by copying and pasting the contents the of this README.

When making a correction to files in this repository, do the following:

1. Make a commit and push it to GitHub _and_ GitLab (two upstream remotes are configured; Google it, there are plenty of how-to's).
2. Use GitHub to create a new release and tag (note that the tag will not automatically go to GitLab, which is OK).

### Files Pages

The original "Files Page" consisted of an FTP-style webpage built using a custom CI/CD pipeline on the TU Delft GitLab amd MUDE web server. The CI/CD pipeline was set up to deploy files to the web server when pushed to branch `release`. It was left intact after the 2024-25 academic year, but it may not me maintained long-term, as the MUDE Team is shifting IT tooling to GitHub. The MUDE web server is being maintained simply to preserve files from before the 2024 academic year and maintain a NetID login for a small number of
sensitive files.

The new "Files Page" is built using GitHub Pages and the [php-directory-listing script](https://github.com/caendesilva/php-directory-listing) by Caen De Silva. Specifically: the Pages setting for the repository uses GitHub Actions; on branch `gh-pages`, all files from `main` are deleted except those in `src/students/`, which are moved to root; several new files are added in this branch only (e.g., `.nojekyll`, the PHP script and the workflow file). To build the website, the workflow file `demo.yml` is modified to recursively list all files in all subdirectories (thank you GitHub Copilot). To modify the files on this (reincarnated) "Files Page," manually replace any files in the `gh-pages` branch with those modified on `main` (and consider using `git checkout ...` from branch `gh-pages` to easily update specific files). 

Remember that any changes made to the repository should be pushed to both remote repositories (GitHub and GitLab); for the "Files Pages" an additional step is to merge the `main` branch into `release` (GitLab) and replace changed files on the `gh-pages` branch manually (GitHub).

## MUDE Topics in Detail

This section describes specific information about each topic, as well as identifying the key authors and contributors. Unless otherwise noted, authors typically were teachers, PhD's and postdocs that played a significant role in designing and developing the materials, whereas contributors played a valuable role, albeit a relatively minor one, for example: reviewing materials, suggesting case studies, formatting digital files and (especially for teaching assistants) giving valuable feedback as to the appropriate level of difficulty and quality of the assignments.

### Week 1.1: Introduction to MUDE and Modelling Concepts

GA 1.1 is included as PDF's in the students handout as it is based on a Power Point slide, except for the Task 2 notebook.

Authors: Robert Lanzafame

Contributors: Patricia Mares Nasarre, Antonio Magherini, Anastasios Stamou were essential for filling in details of some of the assignment tasks related to the Nenana Ice Classic case study.

Materials _not_ included in CC BY license:

- All images and data related to the Nenana Ice Classic, which were reused here and assembled from publicly available sources on the internet.
- Screenshots taken of BSc thesis projects included in the Nenana Ice Classic case study (see slides).

### Week 1.2: Uncertainty Propagation

Authors: Sandra Verhagen, Patricia Mares Nasarre, Robert Lanzafame

Contributors: Jeroen Hoving provided an excellent non-linear equation and some reference slides for ice thickness application.

### Week 1.3-1.4: Observation Theory

Authors: Sandra Verhagen

Contributors: Lotfi Massarweh, Chengyu Yin, Wietske Brower

### Week 1.5-1.6: Numerical Modelling

Authors: Jaime Arriaga Garcia, Justin Pittman, Robert Lanzafame

Contributors: Isabel Slingerland provided critical feedback and suggestions for improving the assignments. Dhruv Mehta provided an outline of some assignments and Ajay Jagadeesh provided feedback.

### Week 1.7-1.8: Univariate and Multivariate Continuous Distributions

Authors: Patricia Mares Nasarre, Robert Lanzafame

Contributors: (anonymized) data sets were provided by Miguel Mendoza Lugo, Gina Torres Alves, Patricia Mares Nasarre, Rieke Santjer. Oswaldo Morales Napoles provided critical feedback on the theoretical aspects.

### Week 2.1: Finite Volume Method

Authors: Jaime Arriaga Garcia, Robert Lanzafame

Contributors: Isabel Slingerland provided critical feedback and suggestions for improving the assignments. Dhruv Mehta provided an outline of some assignments and Ajay Jagadeesh provided feedback. The triangular meshing script was developed by Robert with significant contributions from GitHub Copilot.

### Week 2.2: Finite Element Method

Authors: Frans van der Meer

### Week 2.3: Signal Processing

The source code for the solution for GA 2.3 is not included here, as the intention is to use the same analysis in future years; a PDF in the `src/` directory provides results from the solution without the source code.

Authors: Christiaan Tiberius

Contributors: Serge Kaplev, Lucas Alvarez Navarro.

### Week 2.4: Time Series Analysis

Authors: Christiaan Tiberius, Sandra Verhagen, Berend Bouvy, Alireza Amiri-Simkooei

Contributors: Serge Kaplev.

### Week 2.5: Optimization

Authors: Bahman Ahmadi, Gonçalo Homem de Almeida Correia, Jie Gao

Contributors: Nadia Pourmohammadzia, Jialei Ding and Tom van Woudenberg.

### Week 2.6: Machine Learning

Authors: Iuri Rocha, Anne Poot, Joep Storm, Leon Riccius, Alexander Garzón Díaz

### Week 2.7: Extreme Value Analysis

Authors: Patricia Mares Nasarre

Contributors: Elisa Ragno, Oswaldo Morales Napoles and Robert Lanzafame.

### Week 2.8: Risk Analysis

Authors: Robert Lanzafame

Contributors: Gabriel Follet was essential for processing the ice classic data. The scripts for the Ice Classic part were developed by Robert with significant contributions from GitHub Copilot.

Materials _not_ included in CC BY license:

- All images and data related to the Nenana Ice Classic, which were reused here and assembled from publicly available sources on the internet.

### Programming Assignments (all weeks)

Authors: Robert Lanzafame, Tom van Woudenberg

Contributors: Patricia Mares Nasarre provided content for the data processing PA; Max Guichard for the Object Oriented Programming PA with `scipy.stats` and Riccardo Toarmina for the machine learning PA. Tom developed many of the Git and GitHub related assignments.

Materials _not_ included in CC BY license:

- some PA's include content reused from others under Creative Commons licenses, and is documented in the specific files.