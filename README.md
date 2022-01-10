## PyPSA meets Africa - Inferring Energy Infrastructure

This repository provides methods for __manipulations of satellite imagery__, __training and testing methods for object detection__ using detectron2 and __methods for inference on satellite imagery__.

Together, these aim to provide a full and open-source pipeline to infer electric infrasture on a large scale. This repository is a work in progress and only provides central features.

## How it works

 We are building on the dataset of fully annotated grid infrastructure in satellite imagery provided in [this paper](https://arxiv.org/abs/2101.06390) to train our models. However, the quality of scalable imagery can be suboptimal, so the question is how that training performance can be __transferred__ to images with more scalable quality. We are working on solving that problem using approaches such as cycle-GANS and superresolution models.


## How it is structured

As the repository joins a broad range of methods, the tools are clustered into the following directories:

1. src/geo: Methods to create annotated datasets from tif files and the respective dataframes 
2. src/train: Methods to train object detection networks and enhance training performance in various ways
3. src/infer: Tools to check performance or conduct inference on whole .tif files
4. src/utils: Helper-functions used across the other directories

## Installation

Many of the methods require quite intricate environments and some even can only be executed when supported by high-performance GPUs. Therefore, we have stored dependencies for each directory separately, feel free to take a look at the directory you are interested in.

## Contact

We are always happy about feedback and potential collaborators. Feel free to reach out at <lukas.franken@ed.ac.uk>!




