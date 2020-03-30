#!/usr/bin/env python3

####----------------------------------####
####        Load libraries            ####    
##########################################
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 

####-----------------------------------------####
#### Building cost and update theta functions####
#################################################
def cost_function(theta, X, Y):
	hypothesis = np.zeros(X.shape[0])
	for idx in range(num_train):
		hypothesis[idx] = 1.0/(1.0 + np.exp(-np.dot(X[idx], theta)))
	objective = np.sum( (- Y[i]*np.log2(hypothesis[i]) \
						 - (1.0 - Y[i])*np.log2(1.0 - hypothesis[i]) \
						   for i in range(X.shape[0]) ) )
	objective = objective/X.shape[0]
	return objective

def update_theta(theta, X, Y):
	hypothesis = np.zeros(X.shape[0])
	for idx in range(X.shape[0]):
		hypothesis[idx] = 1.0/(1.0 + np.exp(-np.dot(X[idx], theta)))

	theta_update=np.zeros(X.shape[1])
	for J_X_DO in range(X.shape[1]):
		theta_update[J_X_DO] = np.sum( ( -X[i, J_X_DO]*(Y[i] - hypothesis[i]) 
									      for i in range(X.shape[0]) ) )
	theta_update=theta_update/X.shape[0]
	return theta_update

#####-------------------------------------####
#####         Main programe               ####
##############################################

np.random.seed(0)
# Data Creation using make_blobs()
X_origin, Y = make_blobs(n_samples = 100, centers = 2, n_features = 2)
num_train = X_origin.shape[0]
X = np.c_[np.ones((num_train,1)),X_origin]


# Steepest descent 
num_iteration = 10000
learning_rate = 0.1

# random generate 'theta' at iteration 0
theta = np.random.randn(X.shape[1])

# save values for cost and theta in all iterations
cost_value=np.zeros(num_iteration)
theta_save=np.zeros((num_iteration, X.shape[1]))

# run steepest descent over 'num_iterations'
for idx in range(num_iteration):
	theta_save[idx,:]=theta
	cost_value[idx]=cost_function(theta,X,Y)
	if (idx % 200 == 0) or (idx == num_iteration-1):
		print("iter:",idx, "cost:", cost_value[idx], "theta:", theta_save[idx])	
	delta_theta = update_theta(theta, X, Y)
	theta = theta - learning_rate*delta_theta

# plot the iteration versus the cost functions
plt.plot(np.arange(num_iteration), cost_value)
plt.xlabel('Iterations')
plt.ylabel('cost values')
plt.show()

# Plot the classification line
if X_origin.shape[1] == 2:
	# prepare the contour
	xx, yy = np.meshgrid(np.linspace(-2.5, 5, 1000), np.linspace(-2.5, 7.5, 1000))
	X_pred = np.c_[np.ones(len(xx.ravel())), xx.ravel(), yy.ravel()]
	Z_ = 1.0/(1 + np.exp(-np.dot(X_pred,theta_save[-1,:])))
	Z = np.array([1 if i > 0.5 else -1 for i in Z_])
	Z = Z.reshape(xx.shape)
	contours = plt.contour(xx, yy, Z, levels=[0], colors = ['green'] ,linewidths=2, linestyles='dashed' )
    
    # prepare the training points for scatter plot	
	colors = np.where(Y == 0, 'r', 'b')
	plt.scatter(X_origin[:,0], X_origin[:,1], c = colors ,marker='x')
	plt.show()
else:
	print('More than 2 dimensions are not show the graphics')

