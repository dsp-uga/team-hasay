# team-hasay

References:
	FCN Paper

	U-Net Paper

	https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
		This paper was a major help in learning how to implement complex neural network sturctures in Keras using,
		the functional API. The model architecture and the y_train mask segementation code are the main areas that
		our code will resemble the code found on this webpage. Changes that we made to edit, this code is by 
		changing the default dimensions of the model input image, and subsequently change the dimensions of a later
		convolutional layer to accomodate this change. Also, our architecture trains from scratch, rather than use
		the VCG pre-trained weights to aid the learning process, like their modeld did.

	OpenCV Fourier Transform: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
