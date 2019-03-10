import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(x):
    return x * (x>0)

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def computeLayer(X, W, b):
    return np.matmul(X, W) + b

def CE(target, prediction):
    N = target.shape[0]
    log_score = np.log(prediction)
    return (-1/N) * np.sum(np.multiply(target, log_score))

def gradCE(target, prediction):
    N = target.shape[1]
    return ((-1/N)*(np.sum(np.division(target, prediction), axis=0))).T

def grad_descent(W_h, W_o, b_h, b_o, trainingData, trainingLabels, epoch, alpha, gamma):
    trainingLabels = convertOneHot(trainingLabels)
    W_old = W
    b_old = b
    loss = []
    while(epoch > 0):
        # Forward Pass
        x = computeLayer(trainingData, W_h, b_h)
        x = relu(S)
        x = computeLayer(S, W_o, b_o)
        x = softmax(x)
        print(x[0])
        loss = CE
        




        gradW, gradB = gradMSE(W_old, b_old, trainingData, trainingLabels, reg)
        
        # Save weights and biases
        weights = np.concatenate((weights, W_old[np.newaxis,:,:]), axis=0)
        biases = np.concatenate((biases, b_old[np.newaxis,:,:]), axis=0)

        W_new = W_old - alpha * gradW
        b_new = b_old - alpha * gradB
        # Check for convergence
        if(np.sqrt(np.linalg.norm(W_new - W_old)**2 + np.linalg.norm(b_new - b_old)**2) < EPS):
            return W_new, b_new, weights, biases

        epoch = epoch + 1
        W_old = W_new
        b_old = b_new

    return W_new, b_new, np.array(weights), np.array(biases)

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    print(trainData[0,0,])
    #print("trainData: {}, trainTarget: {}\nvalidData: {}, validTarget: {}\ntestData: {}, testTarget: {}".format(
    #    trainData.shape, trainTarget.shape, validData.shape, validTarget.shape, testData.shape, testTarget.shape
    #))
    trainData = trainData.reshape(-1, 784)
    validData = validData.reshape(-1, 784)
    testData = testData.reshape(-1, 784)

    epochs = 200
    hidden_size = 1000
    W_h = np.random.normal(0, 2/(784+hidden_size), (trainData.shape[1], hidden_size))
    W_o = np.random.normal(0, 2/(hidden_size+10), (hidden_size, 10))
    print(W_h.shape)
    print(W_o.shape) 
