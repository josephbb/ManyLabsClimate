# A Manylabs Megastudy​​ Testing the Main Behavioral Climate Action Interventions in 60 Countries


# Abstract

The climate crisis is currently humanity’s most consequential and challenging problem. Given the solution to this global emergency depends on promoting climate change action, we empirically assessed the effectiveness of 11 expert-crowdsourced theoretically-driven behavioral interventions in stimulating difference facets of climate change mitigation (including belief, policy support, information-sharing, and tree planting efforts). On a sample of 60,000 participants from 60 countries, we found that the intervention effectiveness greatly differed depending on the outcome. For example, climate beliefs were strengthened most by decreasing psychological distance, whereas information sharing was stimulated most by negative emotion induction. These results not only speak to the utility of harnessing such large-scale experimental protocols, but also to the need to tailor interventions according to their targeted outcome. 

## Table of Contents

# Table of Contents
1. [Introduction](#introduction)
   1. [Directory Structure](#directory-structure)
2. [Running the Code](#running-the-code)
   1. [Requirements and Dependencies](#requirements-and-dependencies)
      1. [Downloading Data](#downloading-data)
      2. [Hardware](#hardware)
      3. [Software Requirements](#software-requirements)
   2. [Running the Analysis Code](#running-the-analysis-code)
   3. [Adjusting Parameters](#adjusting-parameters)
3. [Troubleshooting](#troubleshooting)

# Introduction

This repository contains code for the analylsis of a megastudy of interventions geared at improving climate change belief and action. The analysis in the paper consists of a series of Bayesian models for each of four key outcomes (Policy Support, Belief, Social Media Sharing, WEPT). The volume of data inherent to a megastudy introduces computational challenges, such that a GPU (or exceptional patience) is required to run these analyses. 


## Directory Structure
- `src/`
    - `__init__.py`: Allows importing from src
    - `distributions.py`: Distributions used in statistical models
    - `models.py`: Code for PYMC models
    - `plots.py`: Code for plotting results
- `out/` (generated)
    - `posteriors/`: Contains posteriors generated during model fitting
    - ` preregistered/`: Contains output from preregistered analysis
- `dat/` : Contains data
- `params.json`: Analysis parameters
- `environment.yaml`: Anaconda Environment Installation File
- `prereg.R`: Preregistered analysis
- `*.ipynb` : Analysis Ipython Notebooks





# Running the Code 

## Requirements and Dependencies 

### Downloading Data
The analysis code here relies on `data_55.xlsx`, which can be downloaded as described in the manuscript. This dataset comprises data from all countries as of March 20<sup>th</sup> 2023. The data location will need to be updated in `params.json` if its value differs from the default `./dat/data_55.xlsx`

### Hardware
Below describes the hardware used to run this analysis, which may exceed most laptops, particularly with regard to the presence of a CUDA compatible GPU and Memory. We recommend this code is run on a workstation, if one is available. As an alternative, the code can be uploaded to a cloud GPU computer provider (e.g., LambdaLabs). Note that these specifications are for the full Bayesian Analysis. The preregistered analysis could be ran on virtually any hardware produced in the last decade or so. 

- Used
    - Processor (CPU): AMD Ryzen 9 3900x 
    - Graphics Card (GPU): Nvidia 1080 Ti
    - Memory (RAM): 64GB DDR4
- Required minimum  (Estimated):
    - CPU: Modern, > 4 Cores
    - Memory (RAM): 32 GB
    - GPU: 1080 Ti or newer
    - Storage: 30GB


### Software Requirements

To run these analyses, you must first install __[CUDA](https://developer.nvidia.com/cuda-downloads)__ and __[CUDNN](https://developer.nvidia.com/CUDNN)__. As hardware varies, installation of these for your specific machine may vary and we encourage you to read the linked installation guides in depth. 

After confirming CUDA and CUDNN are working on your local machine, the easiest way to run this code is to install __[Anaconda](https://docs.anaconda.com/anaconda/install/index.html)__. Note that the present analysis used Anaconda v22.11.1. After installing anaconda, set up a virtual environment through the following command: 

```
conda env export --name manylabsclimate --file environment.yaml
```

If you have issues with GPU support, please the __[JAX installation guide](https://github.com/google/jax#installation)__. 

Running the pre-registered analysis will require installing __[RStudio](https://posit.co/download/rstudio-desktop/)__. 

### Running the Analysis Code

The analysis code consists of six distinct files: 

- **Model fits**
    - `belief.ipynb` 
    - `share.ipynb` 
    - `policy.ipynb` 
    - `wept.ipynb`
- **Output Generation**
    - `figures_and_tables.ipynb` 
- **Preregistered Analysis**
    - `prereg.R` 

You must first run `belief.ipynb` before running the additional model fitting scripts. This is necessary, as `belief.ipynb` generates adjusted estimates of belief used for subsequent models. Model fitting is a computationally intensive process and may take in excess of 24 hours. This process will generate up to 30GB of files in `./out`, so ensure you have sufficient hard drive space. 

Generated files include posterior estimates stored as netcdf files. These are necessary for generating figures and tables using `figures_and_tables.ipynb`. Figures for the preregistered analysis (at the end of `figures_and_tables.ipynb`) can only be generated if you have ran the pre-registered analysis in R. 

### Adjusting Parameters

Throughout the code, key variables have been added to `params.json`. Adjusting these is beyond the scope of this readme. 


### Note
The modeling here required trade-offs between depth of exploration of the model and feasibility given computational constraints imposed by the dataset. While the posterior predictive fits look reasonable, it is entirely possible that you find an alternate, preferable model specification. If so, please give it a try. Reach out to me if anything is qualitatively different. 


# Troubleshooting

If you encounter any issues while running this code, please try the following steps:

- Check that all the required dependencies are installed and versions are consistent with `environment.yaml` 
    - If you have issues running the first block of code, there are likely software that need to be installed. 
- Make sure you have the necessary hardware requirements as listed in the `Hardware Requirements` section.
    - Running first block of each jupyter notebook should indicate whether the GPU is detected.
    - Getting GPUs running with PYMC's GPU sampling can be tricky, check forums and cofirm your GPU is working with sample code. Ensure that you can sample a trivial model such as: 

    ```
    import pymc as pm
    import pymc.sampling.jax as pmjax
    import numpy as np
    import arviz as az

    y = np.random.normal(0, 1, 100)

    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1)
        sigma = pm.Normal('sigma', 0, 1)
        y_obs = pm.Normal('y_obs', mu, sigma, observed=y)
    
    with model:
        idata = pmjax.sample_numpyro_nuts(chain_method='vectorized', postprocessing_backend='cpu')

    az.summary(idata)
    ```  
- Check that the data are in the correct format and located in the appropriate directory.
- If your issue is with a file other than `belief.ipynb`:
    - Be sure that you have fully ran (without errors) `belief.ipynb` and it has generated an adjusted dataset. 
- If you encounter any errors, try searching for the error message online to see if others have encountered and solved similar issues.
- If you are still having trouble, please reach out to Joe Bak-Coleman. We note that our ability to respond to such requests may be limited. 
- If you identify a bug/issue, we'd be keen to know. 









