# -*- coding: utf-8 -*-
"""
@author: Omer Faruk Dursun
"""
from math import sqrt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Implementation of KNN Algoritm
class KNNClassifier:
    
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(self,row1, row2):
    	distance = 0.0
    	for i in range(len(row1)-1):
    		distance += (row1[i] - row2[i])**2
    	return sqrt(distance)
     
        
    # Locate the most similar neighbors
    def get_neighbors(self, X_train, y_train, test_row, num_neighbors):
        distances = []
        for i in range(0,len(X_train)):
            dist = self.euclidean_distance(test_row, X_train[i])
            distances.append((y_train[i], dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(num_neighbors):
           	neighbors.append(distances[i][0])
        return neighbors     
    
    
    # Predict the label based on neighbour labels
    def predict_classification(self, X_train, y_train, X_test, num_neighbors = 3):
        predictions = []
        for x in X_test:
            neighbors = self.get_neighbors(X_train, y_train, x, num_neighbors)
            prediction = str(max(set(neighbors), key=neighbors.count))
            predictions.append(prediction)
        return predictions

# Implementation of Naive Bayes Classifier
class NaiveBayesClassifier:
    
    # Function that takes X_train and y_train and calculates mean, variance and prior probabilty
    def train(self, X, y):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        self.unique_classes = np.unique(y)
        number_of_classes = len(self.unique_classes)

        self.mean = np.zeros((number_of_classes, number_of_features), dtype=np.float64)
        self.var = np.zeros((number_of_classes, number_of_features), dtype=np.float64)
        self.prior_probabilities =  np.zeros(number_of_classes, dtype=np.float64)
        
        # calculate mean, var, and prior probabilty for each class
        for i, class_label in enumerate(self.unique_classes):
            X_sub_class = X[y==class_label]
            self.mean[i, :] = X_sub_class.mean(axis=0)
            # Add epsilon value to avoid divide by zero
            self.var[i, :] = X_sub_class.var(axis=0) + 0.001
            # Prior probability is just the frequency of each class in the data set
            self.prior_probabilities[i] = X_sub_class.shape[0] / float(number_of_samples) 
            
            
    # Function that takes X_test and predicts the class label
    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            # calculate posterior probability for each class
            # take logs to prevent underflow
            # because we take logs, we sum the probabilities instead of multiplying
            for idx, _ in enumerate(self.unique_classes):
                prior = np.log(self.prior_probabilities[idx])
                posterior = np.sum(np.log(self.pdf(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)
            y_pred.append(self.unique_classes[np.argmax(posteriors)])
        return np.array(y_pred)

    # Probabilty Density Function of Normal (Gaussian) Distribution
    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        upper_part = np.exp(- (x-mean)**2 / (2 * var))
        lower_part = np.sqrt(2 * np.pi * var)
        return upper_part / lower_part

class SklearnDecisionTree:
    
    def decision_tree(self, X_train, y_train, X_test):
        clf = DecisionTreeClassifier()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)
        return y_pred
    
class SklearnRandomForest:
    
    def random_forest(self, X_train, y_train, X_test):
        clf = RandomForestClassifier(max_depth=10, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred
    