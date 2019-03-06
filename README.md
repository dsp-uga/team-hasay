# team-hasay

This repo consists of several efforts to tackle Cilia Segmentation. 

Connecting the ciliary motion with clinical phenotypes is an active area of research. This project is about to design an algorithm to segment the cilia. The Cilias are microscopic hairlike structures that protrude from every cell in your body. They beat in regular, rhythmic patterns to perform number of tasks like, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia and their beating patterns, are increasingly being implicated in a wide variety of syndromes that affected multiple organs.

## Getting Started

Follow the below steps for installation and to run the training and testing sets.

## Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- [Keras](https://keras.io/#installation) - Open-source neural network library
- [Tensorflow](https://www.tensorflow.org/) - API used as Backend for Keras
- [OpenCV](https://opencv.org/) - Open-source library aimed for real-time Computer Vision
- [Theano](http://www.deeplearning.net/software/theano/) - API used as Backend for Keras

## Installation

## Anaconda 

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment.

Download and install Anaconda from (https://www.anaconda.com/distribution/#download-section). 

### Running Environment

•	Once Anaconda is installed, open anaconda prompt using windows command Line.\
•	Run ```conda env create -f environment.yml``` will install all packages required for all programs in this repository.

### To start the environment 

•	For PC like systems ```activate P2-theano```\
•	For Unix like systems ```source activate P2-theano```

## Keras 

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. You can install keras using pip on command line ```sudo pip install keras```.

## Tensorflow 

You can install Tensorflow using pip on command line, for CPU ```sudo pip install tensorflow``` and for GPU ```sudo pip install tensorflow-gpu```

## Data 

The data itself are grayscale 8-bit images taken with [DIC optics](https://en.wikipedia.org/wiki/Differential_interference_contrast_microscopy) of cilia biopsies [published in this 2015 study.](http://stm.sciencemag.org/content/7/299/299ra124) For each video, you are provided 100 subsequent frames, which is roughly equal to about 0.5 seconds of real-time video (the framerate of each video is 200 fps). Since the videos are grayscale, if you read a single frame in and notice its data structure contains three color channels, you can safely pick one and drop the other two. Same goes for the masks. Speaking of the masks: each mask is the same spatial dimensions (height, width) as the corresponding video. Each pixel, however, is colored according to what it contains in the video:

•	2 corresponds to cilia (what you want to predict!)\
•	1 corresponds to a cell\
•	0 corresponds to background (neither a cell nor cilia)

The actual input to our models are located in the ./data/frames_one_std subdirectory. The images in that directory are the result of
calculating the variance of each pixel for every video, and removing pixels intensities whose variance fell below a threshold. 
We assumed the pixel variances adhered to a normal(Gaussian) distribution, and because of such we set our threshold value to the mean 
of the variances plus one standard deviation. The purpose was so that only the top 32% of pixel varainces, hopefully majority cilia,
would remain. We also tested thresholding with the mean plus two and three standard deviations, but recognized too much information
was lost to be effective.

## Scripts
Inside the scripts directory, three python files exist to illustrate the steps taken to pre-process the raw video frames.

- Untar.py simply extracted the contents of each video tar file.
- Variance.py  computed the variance for each pixel, based on the pixel intensity fluctations between frames.
- Movement_Frames.py applied thresholding to the first frame of each video based on the variances created using Variances.py.

## Model Usage
In order to recreate our results, inside the src directory run the command "python Models.py". By default the U-Net model will run.

## Results 

| Method |     Configuration    |   IOU    |     Personnel    |
|--------|----------------------|----------|------------------|
|  FCN   | epochs:200, batch:32 |   30.7   | [Marcus Hill](https://github.com/Tallcus)    |
|  U-Net | epochs:200, batch: 8 |   31.9   | [Dhaval Bhanderi](https://github.com/dvlbhanderi)|
|  MRF   |        -             | Not completed | [Dhaval Bhanderi](https://github.com/dvlbhanderi)

### References:

- [FCN Paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- [U-Net Paper](https://arxiv.org/pdf/1505.04597.pdf)
- https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html

This paper was a major help in learning how to implement complex neural network sturctures in Keras using,
the functional API. The model architecture and the y_train mask segementation code are the main areas that
our code will resemble that found on this webpage. Changes that we made to edit this code is  
changing the default dimensions of the model input image, and subsequently change the dimensions of a later
convolutional layer to accomodate this change. Also, our architecture trains from scratch, rather than use
the VCG pre-trained weights to aid the learning process, like their modeld did.

- OpenCV Fourier Transform: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
- https://medium.com/coinmonks/learn-how-to-train-u-net-on-your-dataset-8e3f89fbd623
- https://github.com/AliMorty/Markov-Random-Field-Project
