## Project Overview

Computer vision is the field of study that focuses on how to train computers to understand and interpret visual information such as images or videos. The resulting models have shown to be really useful for different applications such as self-driving cars, facial recognition, defect detection on production lines, and more. However, the training of these models usually requires massive amounts of data and the performance of the developed algorithms varies between different cases. The projects within the field of computer vision are often challenging as the benchmark for comparison is human vision, which is incredibly successful in performing these types of tasks. 

This project goal is to create a model that receives images as inputs and identifies the presence of a human or a dog and the canine breed they resemble or belong.


## Dataset and Libraries

### Downloading Datasets

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 
4. Make sure you have already installed the necessary Python packages according to the README in the program repository.
5. Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.
	
	```
		jupyter notebook dog_app.ipynb
	```

### Libraries

		Numpy
		cv2                
		matplotlib.pyplot
		Pytorch
		PIL
		os
		glob
		tqdm
		









