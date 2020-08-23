#!/usr/bin/env python3

import math
import numpy as np
from time import time
from collections import Counter

class Data:
    def __init__(self):
        self.features = []	# list of lists (size: number_of_examples x number_of_features)
        self.labels = []	# list of strings (lenght: number_of_examples)

def read_data(path):
    # TODO: function that will read the input file and store it in the data structure
    # use the Data class defined above to store the information
    data = Data()
    f = open(path, "r") # open function opens text file
    lines = f.readlines() # reads all lines in file and stores as list of strings
    f.close() # close file
    for line in lines:
        data.features.append(line.split(",", 4)[:4]) # Gets the numerical values / features
        labelLine = str(line.split(",",4)[4]) # Get the label from same line
        data.labels.append(labelLine.strip()) # Remove trailing newline and add label to list
    return data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        u*v
    """
    # TODO: implement the kernel function
    # Example: Given [x1 x2 x3] and [y1 y2 y3], the dot product will be x1y1 + x2y2 + x3y3. Lists must be same length

    dotProductValue = 0 # This will hold the dot product result
    if(len(u) != len(v)): # If both lists are not of same length, can not compute dot product. Output error message
        print("Error: Lengths of List don't match")
        return -1
    else:
        for i in range(len(u)): # For the length of any list. In this case, u
            dotProductValue += u[i]*v[i] # Take the sum of the products of u and v element wise

    return dotProductValue

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        # TODO: implement the kernel function
        # Get dot product and add by 1, then square by argument.
        return (dot_kf(u,v) + 1) ** d
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return
    return kf

class Perceptron:
    def __init__(self, kf, lr):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.MissedPoints = []
        self.MissedLabels = []
        self.kf = kf
        self.lr = lr

    def train(self, data):
        # TODO: Main function - train the perceptron with data
        

    def update(self, point, label):
        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        # TODO
        return is_mistake

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        return

    def test(self, data):
        predictions = []
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        return predictions


# Feel free to add any helper functions as needed.
if __name__ == '__main__':
    
