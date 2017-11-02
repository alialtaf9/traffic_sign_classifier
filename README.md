#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Can be found in section "Include an exploratory visualization of the dataset" in the notebook. I included graphs to show the distribution of each class across the training, test and validation sets in addition to plotting one example of each class.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

There were 2 major preprocessing steps
1. Normalizing the images using the mean and variance of the training data set
2. Converting images to greyscale

For the additional german traffic sign images there was one prior step of converting images to 32 x 32 pixels.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted was modelled after VGG16 and included the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x24 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x24 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x96 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x192 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x192 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x192	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x192 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x192 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 2x2x192 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 2x2x192	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 2x2x192 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 1x1x192 				|
| Fully connected		| outputs 1x1x512        									|
| Fully connected		| outputs 1x1x512        									|
| Fully connected		| outputs 1x1x43        									|
| Softmax				|         									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Epochs = 50
Batch size = 64
Mu = 0.0
Sigma = 0.05
Learning rate = 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? (did not calculate as what we care about during training is validation accuracy)
* validation set accuracy of 0.962
* test set accuracy of 0.941

If a well known architecture was chosen:
* What architecture was chosen?

VGG16, a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper Very Deep Convolutional Networks for Large-Scale Image Recognition. I had to reduce the depth of the layers to allow the model to train on the g2.2xl machine.

* Why did you believe it would be relevant to the traffic sign application?

Because the model achieves 92.7% top-5 test accuracy in ImageNet  , which is a dataset of over 14 million images belonging to 1000 classes, which suggested it would work well for image recognition constrained to just traffic signs.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The test accuracy of 94.1% suggests the model does a great job of predicting images it wasn't trained on which is a sign of a good classifier. Additionally, it got 6/6 of the new german traffic signs correct.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

See Step 3: Test a Model on New Images in notebook for images.

For each of these images the major difficulty was that these weren't precropped to 32x32. They were all different sizes and had to be resized which would mess up the proportions in the images. This is not something my model had been tested with.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Speed limit 60/hour    			| Speed limit 60/hour 										|
| Right of way ahead					| Right of way ahead											|
| Turn right ahead	      		| Turn right ahead					 				|
| Children crossing			| Children crossing     							|
| No entry			| No entry     							|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This performed better than on the test data but is too small a sample size to draw any major conclusions.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Image 1: Stop sign

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .962         			| Stop sign   									|
| .0192     				| Yield 										|
| .00866					| Priority road										|
| .00656	      			| Keep right				 				|
| .000281				    | Roundabout mandatory     							|

Image 2: Speed limit 60/hour

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (60km/h)   									|
| ~0     				| Speed limit (50km/h) 										|
| ~0					| Speed limit (80km/h)											|
| ~0	      			| Turn right ahead					 				|
| ~0				    | Ahead only      							|

Image 3: Right of way ahead

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Right-of-way at the next intersection   									|
| ~0     				| Road narrows on the right 										|
| ~0					| Road work											|
| ~0	      			| Beware of ice/snow					 				|
| ~0				    | Double curve      							|


Image 4: Turn right ahead

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Turn right ahead   									|
| ~0     				| Ahead only										|
| ~0					| End of all speed and passing limits											|
| ~0	      			| Speed limit (20km/h)					 				|
| ~0				    | Speed limit (30km/h)      							|


Image 4: Turn right ahead

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.955         			| Children crossing   									|
| 0.0446     				| Bicycles crossing 										|
| ~0					| Right-of-way at the next intersection										|
| ~0	      			| Beware of ice/snow					 				|
| ~0				    | Speed limit (60km/h)      							|

Image 6: No entry

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| No entry  									|
| ~0     				| No passing 										|
| ~0					| Speed limit (20km/h)											|
| ~0	      			| Speed limit (30km/h)					 				|
| ~0				    | Speed limit (50km/h)      							|
