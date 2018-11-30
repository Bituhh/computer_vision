Card Recognition Assignment by Victor Coelho De Oliveira - M00579250 - Middlesex University
Assessor Dr. Eris Chinellato - Middlesex University

All tools used:
	OpenCV - Computer Vision library
	Numpy - Numerical/Array/Scientifical library
	Scikit Learn - Neural Network library

This project should contain 3 sets of folder:

./data - It contain pre-trained data for the neural network. A coefficient file (./data/coefs_.npy) and a intercept file (./data/intercepts_.npy). Deleting these, the initial training performed in ./script/card_neural_net.py file will be set to random.

./images - It contains the full dataset of images used to trained the neural net. A total of 13,000 samples 4.7 MB of data.

./script - It contains all scripts related to the project, such as, card cropping for dataset building and card recognitions code. Breakdown of the files below:
	The main program is ./script/card_neural_net.py, once it runs a the program will attempt to load pre-trained data unless the ./data folder is empty. The code will automatically try to identify the card, no trigger is required! An option menu is available by pressing 'o' while the image is in focus, it will be prompt at the terminal. Options availabe are 'train' and 'exit'. Caution is required when performing 'train'. Exiting the program through the menu is recommended!!!!
	The dataset was created with ./script/card_crop.py file. Again an option menu is available by pressing 'o'. Available option are 'c' for cropping the image and saving it and 'exit'.
	The ./script/card_utils.py file contains all utilities and tools/functions used throughout the project. Both Card and NeuralNet class can be found here.


Throught the project Github was used 'https://www.github.com/Bituhh/computer_vision.git'.

For any queries related to this project, contact details can be found below:

Email: victor@oliveira.org.uk
Linkedin: https://www.linkedin.com/in/vcoliveira
Website: http://www.oliveira.org.uk - Working Progress

I'm happy to help at anytime.
