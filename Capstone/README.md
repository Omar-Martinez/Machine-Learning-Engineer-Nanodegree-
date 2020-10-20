## Project Overview

Computer vision is the field of study that focuses on how to train computers to understand and interpret visual information such as images or videos. The resulting models have shown to be really useful for different applications such as self-driving cars, facial recognition, defect detection on production lines, and more. However, the training of these models usually requires massive amounts of data and the performance of the developed algorithms varies between different cases. The projects within the field of computer vision are often challenging as the benchmark for comparison is human vision, which is incredibly successful in performing these types of tasks.

The dog breed classification problem focuses on solving two main problems. First, the model is trained to recognize if the input image contains a human or a dog, and second, it identifies the breed that these resemble. It is a supervised multi-class classification problem that predicts among 133 types of dog breeds.


<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/output_dog.png">
  <img width="350" height="520" src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/Output_me.png">
</p>



## Dataset and Packages

### Downloading Datasets

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.

<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/sample_dogs.png">
</p>

3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/sample_humans.png">
</p>

4. Make sure you have already installed the necessary Python packages mentioned at the end of the README.

### Python Packages
- Numpy
- cv2                
- matplotlib.pyplot
- Pytorch
- PIL
- os
- glob
- tqdm

## Metrics

For this classification problem, the cross-entropy loss will be used for training the models and the accuracy is used as the evaluation metric. This metric is useful for this classification task, as we want to know how many predictions are done correctly over the dataset. It evaluates if the models are identifying the presence of humans and dogs and if the latter are correctly classified within the corresponding breeds.

<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/Accuracy.png">
</p>
	
## Methodology
The implementation of the project consists of the following 7 major steps in order to achieve the desired outcome:

1. <b> Import Datasets </b>

2.  <b> Face Detector Model: </b> Used a pre-trained face detector from OpenCV’s implementation of the Haar feature-based cascade classifier to detect human faces in images

3. <b> Dog Detector Model: </b> A pre-trained VGG-16 model on ImageNet was used as a detector that classified whether there is a dog on the input images. The model was imported using Pytorch

4. <b> CNN to classify dog breeds from scratch (Benchmark Model): </b> In order to achieve at least 10% accuracy on the test dataset, AlexNet (A milestone for CNN Image Classification) architecture was selected for the model. During the training of the full Alexnet architecture, it was clear that the model was suffering from low bias and high variance problem. It was probable that the complexity of the network and the limited amount of data were making the model overfit the training set.
<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/Alexnet.png">
</p>

5. <b> CNN to classify Dog Breeds Using Transfer Learning: </b> The ResNet50 model was chosen for this task as it has shown outstanding performance on image classification problems thanks to the shortcut connections that enable a deeper network that performs better on training.

6. <b> Final Model: </b> Integration of Face Detector, Dog Detector, and ResNet50 as a final model
7. <b> Evaluation Final Model </b> 

## Results Summary
### Face Detector
The performance of the model was assessed and it was identified that:
- In the first 100 images in human_files, the model detected that 98% corresponded to a human face
- In the first 100 images in dog_files, the model detected that 17% corresponded to a human face
It would be interesting to research other face detection pre-trained models and compare their performance. It is possible that there is a model that could perform better on the dog dataset.

### Dog Detector
The performance of the Dog Detector is outstanding as the VGG-16 model identified that 0% of the first 100 images on human_files have a dog and that 100% of the first 100 images on dog_files have detected a dog face. It would be advised to check if this performance maintains when evaluating a larger part of the datasets.

### CNN to classify Dog Breeds from Scratch
After 25 epochs of training with mini-batches of size 32 and a learning rate of 0.05, the simplified AlexNet based model achieved a 12% accuracy on the test set (104/836), which surpassed the expectation of 10%. This a good performance taking into account the complexity of the network, the training time, and the amount of available data.

### CNN to classify Dog Breeds Using Transfer Learning
After 25 epochs of training with mini-batches of size 32 and a learning rate of 0.01, the ResNet50 model achieved an 83% accuracy on the test set (696/836 images), which surpassed the expectation of 60%. A great performance.

### Final Model
The model performed better than expected on 6 images (3 humans and 3 dogs) with different attributes. However, it still not perfect as it classified a dog like a human and a pug breed as a bulldog. It is not surprising as the face detectors previously predicted human faces on the dog images and the similarity between pug breeds and bulldogs is high.

<p align="center">
  <img src="https://github.com/Omar-Martinez/Machine-Learning-Engineer-Nanodegree-/blob/master/Capstone/sample_images/Wrong%20outputs.png">
</p>

## Conclusion
The results are outstanding and the usefulness of pre-trained algorithms is clear. However, It would be interesting to perform the following proposals and evaluate if there is an increase in performance:
- Gather more data (images) and perform more augmentation for training
- Unfreeze more parameters of the model used for transfer learning (ResNet50) so it
would perform better on this task
- Train for more epochs
- Used other models for Face Detection
- Use other models for transfer learning such as VGGs, Inception, GoogLeNet, and
more







