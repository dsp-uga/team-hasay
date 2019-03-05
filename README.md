# team-hasay

This repo consists of several efforts to tackle Cilia Segmentation. 

Connecting the ciliary motion with clinical phenotypes is an active area of research. This project is about to design an algorithm to segment the cilia. The Cilias are microscopic hairlike structures that protrude from every cell in your body. They beat in regular, rhythmic patterns to perform number of tasks like, from moving nutrients in to moving irritants out to amplifying cell-cell signaling pathways to generating calcium fluid flow in early cell differentiation. Cilia and their beating patterns, are increasingly being implicated in a wide variety of syndromes that affected multiple organs.

##Prerequisites

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

