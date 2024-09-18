# README for Group Assignment 1.3

*[CEGM1000 MUDE](http://mude.citg.tudelft.nl/): Week 1.3, Friday, Sep 20, 2024.*

_This assignment is to be turned in by uploading files to GitHub (as done with PA 1.3, but for the GA 1.3 repository)._

There are several files to be aware of in this GA:

1. `README.md`: this file, which contains and introduction and instructions.
2. `Analysis.ipynb`: a Jupyter notebook which contains the primary analysis to complete during the in-class session.
3. `Report.md`: a Markdown file containing questions about the findings of Analysis, as well additional questions to check your understanding of the topic for this week.
4. `functions.py`: a Python file that defines plotting functions for the last Task of the notebook
5. Auxiliary files: a `data` subdirectory that contains `csv` files.

You are expected to read and work through `README`, `Analysis` and `Report` sequentially. Complete all tasks in the Analysis notebook and answer all questions in the Report by 12:30. You should leave the cell outputs in the notebook.

### Submission

**COMING SOON**

### Working Method

**COMING SOON**

## Assignment Context

There are numerous subsurface processes that give rise to observable subsidence at the surface. These processes can be categorized into two main categories: 'deep' subsidence and 'shallow' subsidence. 'Deep' subsidence stems from processes occurring in the deeper subsurface layers (e.g., deeper than 0.5 kilometers below the surface). For instance, extracting gas from a reservoir leads to compaction of the deeper layers, which then results in subsidence of the Earth's surface. On the other hand, 'shallow' subsidence arises from activities within the upper layers of the subsurface. When the groundwater table drops, it triggers shrinkage and oxidation of organic material above the ground water level. Additionally, processes like consolidation and creep contribute to shallow subsidence. Conversely, when the groundwater level rises, a portion of the subsidence becomes reversible, as the layers swell due to the increased water content.

In the Green Heart region in the Netherlands a lot of 'shallow' subsidence occurs. In the typical polder landscape the groundwater table is kept blow a certain level, causing peat layers to oxidize and shrink resulting in subsidence. Also, since the ground water level is highly variable over the year (due to changes in temperature and precipitation), this results in highly variable ground movements which can be quite significant. 

In the context of our assignment, we investigate the observed deformation of a recently constructed road in the <a href="https://www.groenehart.nl/the-green-heart-of-holland" target="_blank"> Green Heart</a> Region. It's reasonable to anticipate that when a heavy structure is built on top of a 'soft' soil layer, additional subsidence may occur due to compaction in the upper surface layers. Over time, as the sediment settles, this extra compaction will diminish. However, it is still expected to observe some up and down movement related due to changing ground water levels. 

### Data

The input data for this assignment are two different deformation time series for a (hypothetical) road in the Green heart in the Netherlands. We assume that the road was built in 2016. We will have a look at <a href="https://en.wikipedia.org/wiki/Interferometric_synthetic-aperture_radar" target="_blank"> InSAR</a> (Interferometric Synthetic Aperture Radar) data and <a href="https://en.wikipedia.org/wiki/Satellite_navigation" target="_blank"> GNSS</a> (Global Navigation Satellite System) data.

With InSAR we can retrieve displacements from time series of radar images. In this exercise we will consider displacement time series from Sentinel-1 from 2017 till 2019. More information on the Sentinel-1 mission can be found <a href="https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1" target="_blank">here</a>.

In the project repository on GitLab, you will find three data files for this assignment:
- `gnss_observations.csv`
- `insar_observations.csv`
- `groundwter_levels.csv`. 

Note that all files consist of two columns, one for the dates and one for the observations. 

In the GNSS and InSAR files the observations are observed vertical displacements (units of m). Groundwater levels are in units of mm.

**Once you have read everything above, continue with the `Analysis.ipynb`**

**End of file.**

<span style="font-size: 75%">
&copy; Copyright 2024 <a rel="MUDE" href="http://mude.citg.tudelft.nl/">MUDE</a>, TU Delft. This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0 License</a>.