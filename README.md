# team-hasay

This repo consists of several efforts to tackle Cilia Segmentation. 

Connecting the ciliary motion with clinical phenotypes is an active area of research. This project is about to design an algorithm to segment the cilia. The Cilias are microscopic hairlike structures that protrude from every cell in your body. They beat in regular, rhythmic patterns to perform number of tasks like, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia and their beating patterns, are increasingly being implicated in a wide variety of syndromes that affected multiple organs.

## Getting Started

Follow the below steps for installation and to run the training and testing sets.

### Prerequisites

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [Anaconda](https://www.anaconda.com/) - Python Environment virtualization.
- [Keras](https://keras.io/#installation) - Open-source neural network library
- [Tensorflow](https://www.tensorflow.org/) - API used as Backend for Keras
- [OpenCV](https://opencv.org/) - Open-source library aimed for real-time Computer Vision
- [Theano](http://www.deeplearning.net/software/theano/) - API used as Backend for Keras

### Instalation




References:

	FCN Paper

	U-Net Paper

	https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
		This paper was a major help in learning how to implement complex neural network sturctures in Keras using,
		the functional API. The model architecture and the y_train mask segementation code are the main areas that
		our code will resemble that found on this webpage. Changes that we made to edit this code is  
		changing the default dimensions of the model input image, and subsequently change the dimensions of a later
		convolutional layer to accomodate this change. Also, our architecture trains from scratch, rather than use
		the VCG pre-trained weights to aid the learning process, like their modeld did.

	OpenCV Fourier Transform: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

