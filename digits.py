from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import os
from scipy.io import loadmat

################################################
#Importing Data
################################################

def get_all_images(dictionary_images, data_type):
	total_number_images = 0
	for i in range(0,10):
		total_number_images+= len(dictionary_images[data_type+str(i)])	
	images = np.ones((785, total_number_images))
	classifier = np.zeros((10, total_number_images))
	image_count = 0
	for i in range(0,10):
		for j in range(len(dictionary_images[data_type+str(i)])):
			images[1:785,image_count] = dictionary_images[data_type+str(i)][j]/255.0
			classifier[i,image_count] = 1
			image_count+=1
	return images,classifier

def get_batch_images(dictionary_images, data_type, total_number_images):
	images = np.ones((785, total_number_images))
	classifier = np.zeros((10, total_number_images))
	image_count = 0
	for i in range(0,10):
		for j in range(total_number_images/10):
			images[1:785,image_count] = dictionary_images[data_type+str(i)][j]/255.0
			classifier[i,image_count] = 1
			image_count+=1
	return images,classifier

def get_all_dataset(dictionary_images):
	train_images,train_classifier =  get_all_images(dictionary_images,"train")
	test_images,test_classifier =  get_all_images(dictionary_images,"test")
	return train_images,train_classifier, test_images, test_classifier

def get_batch_dataset(dictionary_images, batch_size):
	train_images,train_classifier =  get_batch_images(dictionary_images,"train",batch_size)
	test_images,test_classifier =  get_batch_images(dictionary_images,"test",batch_size)
	return train_images,train_classifier, test_images, test_classifier

################################################
#Logistic Regression
################################################

def logreg_softmax(weights, images):
	'''
	Function: generate softmax probabilities 
	Input: 
		weights: matrix of weights of size (785 x 10)
		images: matrix of flattened images as column vectors (785 x #images)
	Output:
		probability: probability for classifying as each digit for each image is stored as a column vector (10 x #images)
	'''
	guess = get_classification(weights, images)
	probability = np.exp(guess)
	probability = probability/probability.sum(axis=0)[None,:]
	return probability

def logreg_get_cost_matrix(probability, classifier):
	probability = np.log(probability)
	cost_matrix = (-1)*np.multiply(probability, classifier)
	return cost_matrix

def logreg_get_cost(probability, classifier):
	cost_matrix = logreg_get_cost_matrix(probability, classifier)
	cost = np.sum(cost_matrix)
	return cost

def logreg_get_gradient(probability, images, classifier):
	difference = probability-classifier
	gradient = np.dot(images, difference.T)
	return gradient

def logreg_grad_descent(weights,images,classifier,iterations, get_history, step_size):
	'''
	Function: runs a gradient descent to minimize the cost function by updating the weight matrix 
	Input:
		weights: intial guess of classifier function (matrix or vector)
		images: image data used to calculate output of classifier function (matrix)
		classifier: target outputs also used to calculate output of classifier function (matrix)
		iterations: number of iterations to run for the gradient descent
		get_history: Boolean to determine whether to save history of weights
		step_size: how frequent to save history of weights
	Output: 
		old_weights: the weight matrix after gradient descent
		history: a list of the history of weights
	'''
	alpha =  0.01
	old_weights = weights
	probability = logreg_softmax(weights, images)
	current_cost = logreg_get_cost(probability, classifier)
	i = 0
	history = [old_weights]
	initial_cost = current_cost
	initial_weights = old_weights
	intial_probability = probability
	while iterations>=i:
		if i%step_size == 0 and i !=0: history.append(old_weights)	
		if i%100 == 0 and i !=0: print "iteration:", i, "cost:", current_cost
		gradient = logreg_get_gradient(probability, images, classifier)
		new_weights = old_weights - alpha*gradient 
		new_probability = logreg_softmax(new_weights, images)
		new_cost = logreg_get_cost(new_probability, classifier)
		if initial_cost < new_cost or math.isnan(new_cost):
			current_cost = initial_cost
			old_weights = initial_weights
			probability = intial_probability
			alpha = alpha*0.95
			history = [initial_weights]
			i = 0
		else:
			old_weights = new_weights
			current_cost = new_cost
			probability = new_probability 
			i+=1
	if get_history == True:
		return old_weights, history
	else:
		return old_weights

################################################
#Linear Regression
################################################

def get_error(theta, data, desired_result):
	'''
	Function: computes error of classifier function from a given desired result
	Input: 
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	output:
		error: the calculated difference between the classifier function output and the desired result (matrix of vector)
	'''
	h_theta = np.dot(data.T, theta.T) #dot product to calculate output of classifier function
	error = h_theta-desired_result #subtracting desired result from classifier function for error
	return error #return error

def get_grad_vector(theta, data, desired_result):
	'''
	Function: returns a gradient vector or matrix of the direction to minimize cost function
	Input: 
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	Output:
		grad_vector: a gradient vector or matrix of the direction to minimize cost function
	'''
	error = get_error(theta, data, desired_result) #acquire error between classifier function and desired result
	grad_vector = np.dot(data, error) #dot product of image data with error for gradient vector of cost function
	return grad_vector #return gradient vector

def cost_function(theta, data, desired_result):
	'''
	Function: returns the cost function of a given theta, x, and target output y using squared error
	Input:
		theta: classifier function (matrix or vector)
		data: image data used to calculate output of classifier function (matrix)
		desired_result: desired result from classifier function (matrix or vector)
	Output: 
		cost: the average square error of the classifer function output compared to the target output (float)
	'''
	error = get_error(theta, data, desired_result) #aquire error between classifier output and target output
	error_squared = np.square(error) #square errors 
	cost = np.sum(error_squared) #average all errors
	return cost #return cost 

def grad_descent(theta, training_data_x, training_data_y, iterations, print_data):
	'''
	Function: runs a gradient descent to minimize the cost function to a target cost by updating the theta classifier matrix 
	Input:
		theta: intial guess of classifier function (matrix or vector)
		training_data_x: image data used to calculate output of classifier function (matrix)
		training_data_y: target outputs also used to calculate output of classifier function (matrix)
		alpha: used to reduce the step size of each iteration such that the gradient descent does not diverge (float)
		target_cost: the target cost of the gradient descent (float)
		print_data: whether to print some statistics from the gradient descent (final cost, # iterations, alpha used for descent)
	Output: 
		theta_new: the theta classifier matrix which satisfies the target cost for the training set data (matrix)
		theta_history: a list of the thetas at every 10th iteration of the gradient descent
	'''
	alpha = 1
	training_data_y = training_data_y.T
	theta = theta.T
	theta_old = theta #store initially guessed theta as the old theta
	act_cost = cost_function(theta_old, training_data_x, training_data_y) #calculate the cost of the guessed theta
	theta_history = [] #intilize list to contain thetas every 10th iteration
	i = 0 #initialize iteration counter
	while iterations>=i: #continue to gradient descent until the cost is below the target cost is reached
		if i == 0: #check if it is the first time in the loop
			act_cost_zero = act_cost #store cost of guessed theta
			theta_zero = theta_old #store guessed theta
		if i%1 == 0: #check if 10th iteration
			theta_history.append(theta_old) #store theta in history
		grad_vector = get_grad_vector(theta_old, training_data_x, training_data_y) #obtain gradient vector of cost function
		theta_new = theta_old - alpha*grad_vector.T #update theta using gradient descent
		new_act_cost = cost_function(theta_new, training_data_x, training_data_y) #calculate new cost with updated theta
		if act_cost_zero < new_act_cost or math.isnan(new_act_cost): #redefine alpha if newly calculated cost diverges from cost of initially guessed theta
			theta_old = theta_zero #reinstate initially guessed theta
			act_cost = act_cost_zero #reinstate cost of initially guessed theta
			alpha = alpha*0.95 #reduce alpha by a factor of 0.95
			theta_history = [] #clear theta history 
			i == 0 #restart iteration count
		else: #cost did not diverge from cost of previous iteration
			theta_old = theta_new #store updated theta as the old theta
			act_cost = new_act_cost #update cost of updated theta as the cost to be compared against next iteration
			i+=1 #update number of iterations
		if alpha == 0.0: #check if alpha has been reduced so much that it is zero
			print "Error: alpha reduced to zero" #print error message
			return theta_new, theta_history #break loop and exit function
	if print_data: #check if the data is to be displayed
		print "Final Cost:", cost_function(theta_new, training_data_x, training_data_y)
		print "Total Number of Iterations:", i
		print "Actual alpha Used:", alpha
	return theta_new, theta_history #return final theta and history of thetas


################################################
#Finite Differences
################################################

def get_finite_difference(weights, images, classifier,step_size):
	'''
	Function: Returns a gradient vector or matrix of the direction to minimize cost function
	Input: 
		weights: classifier function (matrix)
		images: image data used to calculate output of classifier function (matrix)
		classifier: desired result from classifier function (matrix)
		step_size: step sized used in approximation of gradient (float)
	Output:
		gradient: a gradient vector or matrix of the direction to minimize cost function
	'''
	gradient = np.ones(weights.shape)
	probability = logreg_softmax(weights, images) #calculate softmax
	bwd = logreg_get_cost(probability, classifier) #f(x)
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			weights[i][j]+= step_size 
			probability_stepped = logreg_softmax(weights, images)
			fwd = logreg_get_cost(probability_stepped, classifier) #f(x+h)
			gradient[i][j] = (fwd-bwd)/(step_size) #update slope
			weights[i][j]-= step_size  
	return gradient

def unit_vector(vector):
	''' 
    Function: Returns the unit vector of the vector
    '''
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	'''
	Function: compresses v1 and v2 into vectors and returns the angle between the vectors
	Input: 
		v1: first vector or matrix
		v2: second vector or matrix
	Output: 
		angle between v1 and v2 in radians
	'''
	v1 = v1.reshape(1, np.size(v1)) #reshape into vector
	v1 = v1[0]
	v2 = v2.reshape(1, np.size(v2)) #reshape into vector
	v2 = v2[0]
	v1_u = unit_vector(v1) #calculate unit vector direction
	v2_u = unit_vector(v2) #calculate unit vector direction
	angle = np.arccos(np.clip(np.dot(v1_u.T, v2_u), -1.0, 1.0))
	if angle > math.pi: angle = angle-math.pi
	return angle

################################################
#Testing
################################################

def get_classification(weights, images):
	'''
	Function: generate outputs O_0 to O_9 
	Input: 
		weights: matrix of weights of size (10 x 785)
		images: matrix of flattened images as column vectors (785 x #images)
	Output:
		classification: classification matrix where the identification for each image stored as a column vector (10 x #images)
	'''
	classification = np.dot(weights.T,images)
	return classification

def logreg_test_data(weights, test_images, test_classifier):
	probability = logreg_softmax(weights, test_images)
	correct = 0
	for i in range(probability.shape[1]):
		if probability[:,i].argmax() == test_classifier[:,i].argmax():
			correct +=1
	return float(correct)/float(probability.shape[1])

def linreg_test_data(weights, test_images, test_classifier):
	'''
	Function: determine number of correctly/incorrectly classified 
	Input:
		weights: hypothesis matrix, calculated through the gradient descent function
		test_images: matrix of flattened image vectors
		test_classifier:
	Output:
		correct: number of images correctly classified
		incorrect: number of images incorrectly classified 
	'''
	correct = 0 #initialize counter for correctly classified images
	prediction = get_classification(weights.T, test_images) #compute output vector of hypothesis function
	for i in range(prediction.shape[1]):
		if prediction[:,i].argmax() == test_classifier[:,i].argmax():
			correct +=1	#if incorrect, update incorrect counter
	return float(correct)/float(prediction.shape[1]) #return final counters of correct and incorrect

def display_weights(weights):
	for i in range(0,10):
		print "Displaying", i
		image = weights[1:785,i]*255
		image = image.reshape((28,28))
		imshow(image, cmap=cm.gray)
		show()
	return

################################################
#Parts
################################################
def part1(dictionary_images):
	for i in range(0,10):
		for j in range (0,10):
			image = dictionary_images["train"+str(i)][j]
			image = image.reshape((28,28))
			print "Saving the", j,"th image of", i
			name = str(i)+"_"+str(j)+"th_image.png"
			imsave(name, image, cmap=plt.cm.gray)
	return

def part3b(train_images, train_classifier):
	weights = np.ones((10,785))
	plot_step_size = 100
	iterations = 1000
	logweights, logreg_history =  logreg_grad_descent(weights,train_images,train_classifier,iterations, True, plot_step_size)
	angles = []
	costs = []
	for i in range(len(logreg_history)):
		print i
		probability = logreg_softmax(logreg_history[i], train_images)
		actual_grad = logreg_get_gradient(probability, train_images, train_classifier)
		finite_diff_grad = get_finite_difference(logreg_history[i], train_images, train_classifier,0.001)
		angles.append(angle_between(actual_grad, finite_diff_grad))
		costs.append(logreg_get_cost(probability, train_classifier))
	plt.suptitle('Difference between Finite Diff Gradient and Actual Gradient vs Location on Cost Function')
	plt.xlabel('Cost')
	plt.xscale('log')
	plt.ylabel('Vector Angle between Flattened Matricies (Radians)')
	plt.plot(costs, angles, 'ro', label="Step Size of 0.001")
	plt.legend(loc='best')
	plt.show()
	return 	

def part4(train_images, train_classifier, test_images, test_classifier):
	weights = np.ones((785,10))
	plot_step_size = 10
	iterations = 1000
	logweights, logreg_history =  logreg_grad_descent(weights,train_images,train_classifier,iterations, True, plot_step_size)
	display_weights(logreg_history[2])
	iterations = arange(iterations/plot_step_size)*plot_step_size
	logtest = []
	logtrain = []
	for i in range(len(iterations)):
		logtrain.append(logreg_test_data(logreg_history[i], train_images, train_classifier))
		logtest.append(logreg_test_data(logreg_history[i], test_images, test_classifier))
	plt.suptitle('Performance of Logistic Regression')
	plt.xlabel('Iterations')
	plt.ylabel('Percent Correctly Classified')
	plt.plot(iterations, logtest, label="Testing Performance")
	plt.plot(iterations, logtrain, label="Training Performance")
	plt.legend(loc='best')
	plt.show()
	return 

def part5(train_images, train_classifier, test_images, test_classifier):
	weights = np.random.rand(785,10)
	train_images_noise =  train_images.copy()
	train_images_noise[1:785,:] += (np.random.rand(train_images.shape[0]-1,train_images.shape[1])-0.5)*2
	train_images_noise[1:785,:] = abs(train_images_noise[1:785,:])
	plot_step_size = 100
	iterations = 10000
	linweights, linreg_history =  grad_descent(weights, train_images, train_classifier, iterations, True)
	logweights, logreg_history =  logreg_grad_descent(weights,train_images,train_classifier,iterations, True, plot_step_size)
	linweights_noise, linreg_history_noise =  grad_descent(weights, train_images_noise, train_classifier, iterations, True)
	logweights_noise, logreg_history_noise =  logreg_grad_descent(weights,train_images_noise,train_classifier,iterations, True, plot_step_size)
	iterations = arange(iterations/plot_step_size)*plot_step_size
	logtest = []
	lintest = []
	logtest_noise = []
	lintest_noise = []
	for i in range(len(iterations)):
		lintest.append(linreg_test_data(linreg_history[i], test_images, test_classifier))
		logtest.append(logreg_test_data(logreg_history[i], test_images, test_classifier))
		lintest_noise.append(linreg_test_data(linreg_history_noise[i], test_images, test_classifier))
		logtest_noise.append(logreg_test_data(logreg_history_noise[i], test_images, test_classifier))
	plt.suptitle('Performance of Logistic Regression versus Linear Regression with Noise')
	plt.xlabel('Iterations')
	plt.ylabel('Percent Correctly Classified')
	plt.plot(iterations, lintest, 'y', label="Linear Performance with No Noise")
	plt.plot(iterations, logtest, 'b',label="Logistic Performance with No Noise")
	plt.plot(iterations, lintest_noise, 'g', label="Linear Performance with Noise")
	plt.plot(iterations, logtest_noise, '-',label="Logistic Performance with Noise")
	plt.legend(loc='best')
	plt.show()
	print "Displaying Image With No Noise"
	image1 = train_images[1:785,1000]*255
	image1 = image1.reshape((28,28))
	imshow(image1, cmap=cm.gray)
	show()
	print "Displaying Image With Noise Image"
	image2 = train_images_noise[1:785,1000]*255
	image2 = image2.reshape((28,28))
	imshow(image2, cmap=cm.gray)
	show()
	return 


M = loadmat("mnist_all.mat") #dictionary
#uncomment this line for batch data set
# train_images,train_classifier, test_images, test_classifier =get_all_dataset(M) 

#uncomment this line for mini batch data set
train_images,train_classifier, test_images, test_classifier =get_batch_dataset(M,5000)

#uncomment to run each part
# part1(M)
# part3b(train_images, train_classifier)
# part4(train_images, train_classifier, test_images, test_classifier)
# part5(train_images, train_classifier, test_images, test_classifier)





