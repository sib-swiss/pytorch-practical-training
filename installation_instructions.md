# Environment setup for the Deep Learning Practical training

> :exclamation: if you encounter any error with the instructions given on this page, please create a [github issue](https://github.com/sib-swiss/pytorch-practical-training/issues/new) to explain your problem  and we will try to get back to you ASAP.


We detail in this page how to set up your environment with the different external modules you will need in order to be able to follow the course.

We recommend you create a new conda environment specifically for the course (if you are unfamiliar with conda environment see [this documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)). 

Nevertheless, we detail here several methods a trust you will choose the one most appropriate to your situations.

**important**: the course materials were developped and tested with **python 3.11.5**. Any anterior version may give errors and warnings aplenty!

## method 1 : new conda environment from `.yml`

Download the file <a href="https://downgit.github.io/#/home?url=https://github.com/sib-swiss/pytorch-practical-training/blob/main/pytorch_course.yml" target="_blank">pytorch_course.yml</a>.


If you are on Windows and/or are allergic to command line, you can use the [anaconda navigator](https://docs.anaconda.com/anaconda/navigator/tutorials/manage-environments/#importing-an-environment) (if you don't know how to start the navigator, [here's how](https://docs.anaconda.com/anaconda/navigator/getting-started/#starting-navigator)).


Otherwise, just open a terminal, navigate to where the file is, and use the following command:
```
conda env create -f pytorch_course.yml
```

Activate the new environment: `conda activate pytorch_course`

Verify that the new environment was installed correctly: `conda env list`

## method 2 : conda and pip commands to install 

These first 2 commands create and activate a new enviroment
```
conda create --name pytorch_course python=3.11.5
conda activate pytorch_course
```

> Important note this configuress pytorch for a cpuonly usage, which should be fine during the course as we use smaller examples. If you want to leverage the power of your GPU, please refer to the [pytorch website](https://pytorch.org/get-started/locally/) (NB: you will also have to install the proper GPU API libraries suich as CUDA or ROCm)

These commands install all necessary modules and their dependencies:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install plotly
pip install pytorch-model-summary
conda install matplotlib
conda install pandas
conda install -c anaconda pytables 
conda install seaborn
pip install psutil
pip install "ray[data,train,tune,serve]"
conda install -c conda-forge shap
```

> conda install command will prompt you for confirmation before installing once they have retrieved the dependencies.

## method 3 : install the following however you want

Python : we recommend at least 3.11.5

 * pytorch 
 * torchvision
 * torchaudio
 * plotly
 * pytorch-model-summary
 * matplotlib
 * pandas
 * pytables 
 * seaborn
 * psutil
 * ray tune
 * shap
