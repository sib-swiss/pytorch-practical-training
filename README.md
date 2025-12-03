
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17799102.svg)](https://doi.org/10.5281/zenodo.17799102)


# Practical dip into deep learning - a PyTorch short crash-course

This repository regroups works on the "Practical dip into deep learning - a PyTorch short crash-course" course of the SIB.

Its goal is to propose a practical introduction to pytorch and deep-learning models by reproducing or getting inspiration from a number published deep-learning models in the field of biology.

This course eschew the theory (which is covered in another SIB course) and focuses instead on the details of the implementation.

## pre-requisites

The course is targeted to life scientists who are already familiar and fluent with the Python programming language and who have a solid knowledge in machine learning.



In order to follow the course you need to have installed [python](https://www.python.org/) and [jupyter notebooks](https://www.jupyter.org/), as well as a number of prerequisite libraries.

See the [intructions on installing prerequisite libraries](installation_instructions.md) of more details.


## course organization 

The course is organized in several, numbered, jupyter notebooks, each corresponding to a model which interleaves code demo, and exercises.

The course does not require any particular expertise with jupyter notebooks to be followed, but if it is the first time you encounter them we recommend this [gentle introduction](https://realpython.com/jupyter-notebook-introduction/).


 * [01 my_first_pytorch_neural_network](01_my_first_pytorch_neural_network.ipynb): a very basic Neural Network on synthetic data. The "Hello World!" of pytorch.
 * [02 protein_subcellular_localisation_classifier](02_protein_subcellular_localisation_classifier.ipynb): classical Deep Learning architecture for classification
 * [03 single_cell_autoencoder](03_single_cell_autoencoder.ipynb): An autoencoder with a custom loss function
 * [04 chest_Xray_CNN](04_chest_Xray_CNN.ipynb): Convolutional Neural Network for image classification
 * [05 chest_Xray_Transfer_Learning](05_chest_Xray_Transfer_Learning.ipynb): Leveraging pre-existing models 
 * [GRU](GRU.ipynb): a small demonstration of Gated Recurrent Units
 * [ladybug_GRU](ladybug_GRU.ipynb): exercise notebook on Gated Recurrent Units

Solutions to each practical can be found in the [`solutions/`](solutions/) folder and should be loadable directly in the jupyter notebook themselves.

Note also the `pytorchtools.py` file which contain some early stopping utilities.


## directory structure


* data : contains the datasets
* images : images generated or used in the notebooks
* drafts : some drafts notebooks, with many failed attempts...
* solutions: exercise solutions

## citation 

If you (re-)use this material, please cite as:

Duchemin, W., Tran, V. D., & Mueller, M. (2025, December 3). Diving into deep learning - theory and applications with PyTorch. Zenodo. [https://doi.org/10.5281/zenodo.17799102](https://doi.org/10.5281/zenodo.17799102)
