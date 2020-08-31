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
            dotProductValue += float(u[i])*float(v[i]) # Take the sum of the products of u and v element wise
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
        subtractedList = [] # List will hold values once subtraction of two lists take place
        subtractedMagnitude = 0 # Represents the magnitude of the subtracted list
        # Obtains the subtracted list
        for i in range(len(u)):
            value = float(u[i]) - float(v[i])
            subtractedList.append(value)
        # Obtains the magnitude
        for i in range(len(subtractedList)):
            subtractedMagnitude += subtractedList[i] ** 2
        return math.exp((-1*subtractedMagnitude)/(2*(s**2)))
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
        myLabels = [] # This will hold the data labels in numerical values
        # For loop converts labels to numerical values
        for i in range(len(data.labels)):
            if(data.labels[i] == "Iris-setosa"):
                myLabels.append(1)
            else:
                myLabels.append(-1)
        #  While loops runs until converged = True
        converged = False
        while(converged == False):
            converged = True # If converged remains true, loop will exit
            for i in range(len(data.features)):
                trainPrediction = self.calculatePrediction(data.features[i]) # Calculate the prediction
                if(self.update(trainPrediction, myLabels[i]) == True): # If update is necessary, we need to add missedLabl and missedPoints to respective list. converged is set to false
                    self.MissedLabels.append(myLabels[i])
                    self.MissedPoints.append(data.features[i])
                    converged = False
        return self.MissedLabels, self.MissedPoints #

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
        if(point * label <= 0):
            is_mistake = True
        else:
            is_mistake = False
        return is_mistake

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        summation = 0
        # Use formula from Lecture Slide 9
        for i in range(len(self.MissedLabels)):
            summation += self.MissedLabels[i] * self.kf(point,self.MissedPoints[i])
        return np.sign(summation) # Return -1 or 1

    def test(self, data):
        predictions = []
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        predictions = [] # Create an empty list.
        for i in range(len(data.features)): # Iterate through data and obtain prediction. Store in list and return once process is finished
            predictions.append(self.predict(data.features[i]))
        return predictions

    # Function is used to calculate the summation portion
    def calculatePrediction(self,point):
        if not self.MissedPoints: #Checks if it is an empty list
            trainPrediction = 0
        else:
            trainPrediction = 0
            # If not empty, calculate summation
            for i in range(len(self.MissedLabels)):
                trainPrediction += self.MissedLabels[i] * self.kf(point,self.MissedPoints[i])
        # Use Signum function to convert to -1, 0 , or 1
        trainPrediction = np.sign(trainPrediction)
        return trainPrediction


# Feel free to add any helper functions as needed.
if __name__ == '__main__':
# Gets the accuracy percentage
    def accuracyTest(testingLabels, predictionValues):
        correct = 0 # Holds total correct
        for i in range(len(testingLabels)): # Iterate through labels and predictionValueList
            if(testingLabels[i] == predictionValues[i]):
                correct += 1
        return str((correct/len(testingLabels)) * 100) + "%"

# Obtain training data.
trainData = Data()
trainData = read_data("trainData.txt")
# Convert data to homogenous data
for i in trainData.features:
    i.append(1)
# Send data to train
myPerceptron = Perceptron(poly_kernel(2), 0.2) # change kf to apply different kernel
myPerceptron.train(trainData)
# Retrieve Test Data
testData = Data()
testData = read_data("testData.txt")
# Convert data to homogenous data
for i in testData.features:
    i.append(1)
# Obtain Prediction
predictions = myPerceptron.test(testData)
# Convert test labels to numerical value
myTestLabels = []
for i in testData.labels: 
    if(i == "Iris-setosa" ):
        myTestLabels.append(1)
    else:
        myTestLabels.append(-1)
# Print accuracy
print(accuracyTest(myTestLabels, predictions))


