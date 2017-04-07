## Author: Francisco Chima Sanchez
## Created: 04-07-2017
## Last modified: 04-07-2017

# This script implements logistic regression with a linear decision boundary on a dataset from Coursera's Introduction to Machine Learning course, in a separate text file.

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import math

# This function calculates the cost function for the dataset provided, separated into features vector X, classification vector y, and a theta vector which defines the model for the decision boundary.

def costFunction(theta, X, y):

    m = len(y) # Number of training examples
    grad = np.zeros(len(theta)) # Will hold the gradient function
    inner = np.zeros(m) # Placeholder to be used in calculating the cost function

    X_theta_product = np.dot(X, theta)
    hypothesis = 1/(1+np.exp(-X_theta_product)) # Sigmoid function, defines the probability density function for the likelihood of classification
    
    # Generates the cost function
    for i in range(m):
        inner[i] = -y[i]*math.log(hypothesis[i]) - \
                   (1-y[i])*math.log(1-hypothesis[i])
    cost = (1 / m) * sum(inner)
    
    # Generates the gradient function
    for j in range(len(theta)):
        inner = np.multiply(np.subtract(hypothesis, y), X[:,j])
        grad[j] = (1 / m) * sum(inner)
    return cost

# Main function that calls costFunction defined above
plt.style.use('ggplot')

# Loads data from auxiliary file
data = np.loadtxt('exData1.txt', delimiter=',')
X_raw = data[:,[0,1]]
y = data[:,2]
m = len(y)

# Creates feature matrix X and initializes the theta vector
X0 = np.ones(m)
X = np.c_[X0.transpose(), X_raw]
init_theta = np.zeros(3)

# Trains logistic regression using the cost function and a preset optimization scheme
myargs = (X, y)
theta = op.fmin(costFunction, x0=init_theta, args=myargs)
print(theta) # Print the resultant theta that defines the model

# Creates vectors for classified data
pos = np.nonzero(y)
neg = np.where(y == 0)[0]

# Creates plots for classified data and decision boundary
fig, ax = plt.subplots()
plot_x = [np.min(X[:,1])-2,  np.max(X[:,1])+2]
plot_y = (-1/theta[2])*(np.multiply(theta[1], plot_x) + theta[0])
ax.plot(X[pos, 1], X[pos, 2], 'o', color = '#ffbb78')
ax.plot(X[neg, 1], X[neg, 2], 'o', color = '#1f77b4')
ax.plot(plot_x, plot_y, 'k--')
plt.show()
