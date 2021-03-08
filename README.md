# Keras-Tensorflow-DenoisingModelsFramework-withTNRD

Ryan Cecil, 2021, Duquesne University.

## Summary
This is a subset of the current code base that I use for 
my research under Dr. Stacey Levine in her deep learning, computer vision, and image processing group at Duquesne University to 
train/test/analyze new models. The current code gives a Keras-Tensorflow implementation of the Trainable 
Nonlinear Reaction Diffusion (TNRD) model of Chen and Pock: https://arxiv.org/pdf/1508.02848.pdf
The implementation can be found in models.py. New models can also be easily defined there.


## Purpose
This work allows me to easily define new denoising models, train/test them, and analyze them. To see how a model is
trained, please see the file example.sh, which gives commands for training a small TNRD model, and applying the different
functions I have to get information about the model. 


Note: If you wish to use this code, in models.py and icg.python.ops.icg_ops.py change all occurrences of 
{/home/cpm/Research/KfTNRD/2021/Denoising_Framework} to the path to the project

## Code Overview

Here is a list of the directories and files plus explanations:


Files:

actions.py - Implements the current actions we can take with a model to train,test, and analyze it.

call_model.py - Implements an easy way to train,test, and analyze new models in sequence via bash files

losses.py - Stores custom loss functions for the models. Only contains one for the TNRD model

models.py - Contains implemented models. All the models we create have to have a similar structure so the models found in this directory inherit
    from the My_Model class found in My_Model.py. See both directories for more details
    
My_Model.py - Contains main class for models. Every model we create inherits from this class

schedulers.py - Contains training schedulers for model training in keras. Currently only has one for TNRD.

TTA.py - Stands for Train, Test, Analyze. Implements a class that accepts a model_class defined in models.py as input
    and allows us to easily train, test, and analyze the model. Although currently, the only analysis functions I have
    in this subset of my code are to plot the parameters of the model, look at the ranges of layer outputs, and look at layer outputs.
    
Directories:

   Analysis - This directory contains the code for my analysis functions used in TTA. Current methods allow us to plot 
   the parameters of the model, look at the ranges of layer outputs, and look at layer outputs.
   
   CondaEnvandICG - Contains compiled code from here: https://github.com/VLOGroup/denoising-variationalnetwork, which
    was the first implementation of TNRD in Tensorflow. They created customa CUDA/C++ code to implement the radial 
    basis function activiations in TNRD. Their code, however, is severly outdated.  In CondaEnvandICG I just store
    their CUDA/C++ code and a version I compiled to use with Tensorflow. If you want to know how to load custom
    compiled bazel code look in icg.python.ops.icg_ops.py. This directory also contains the current Conda environment
    I use for my research.
    
   data - Stores the data I use for the different pipelines. Each data directory must contain a folder of images 
   to train, a folder of images to test, and a folder of images to analyze. Current data is only Train400Im68. These 
   are the standard 400 training images denoising models are usually first trained on, and the then the standard
   set of im68 images these models are usually tested on. Metric for denoising is PSNR (stems from MSQE) and SSIM.
   
   data_generators - This directory contains the different data pipelines. Only pipeline currently implemented is
    for Train400Im68. Every pipeline must have a data generator for the training. I liked the data generator used 
    in https://github.com/cszn/DnCNN so I copied some of their code.
    
   icg: Loads the Tensorflow operator for the Gaussian radial basis activation functions. CUDA/C++ implementation from
    https://github.com/VLOGroup/denoising-variationalnetwork
    
   layers: Contains custom keras layers for models. Currently just has a few for TNRD.
   
   model_results: Intended to store results of model. Results directory for a model can be specified differently in
    models.py
    
   saved_models: Intended to store saved models. Can be specified differently in
    models.py
    
   Testing: This directory contains the code for my testing functions used in TTA. Currently, has code for a custom keras
   callback during training and for applying the model to the test data and saving the results.
   
   Training: This directory contains the code for training the models. Uses model.fit to train the model, saves loss values,
   loss plots, validation plots. Training settings such as number of epochs, scheduler, etc. will be dependant upon the 
   model settings either passed through call_model.py or specified manually in models.py.
    
