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
    sums = e_x.sum(axis=1)
    sums = np.tile(sums, (10,1)).T
    return np.divide(e_x, sums)

def computeLayer(X, W, b):
    b = (b.T).repeat(X.shape[0], axis=0)
    return np.matmul(X, W) + b

def CE(target, prediction):
    N = target.shape[0]
    log_score = np.log(prediction)
    return (-1/N) * np.sum(np.multiply(target, log_score))

def gradCE(target, prediction):
    N = target.shape[1]
    return ((-1/N)*(np.sum(np.division(target, prediction), axis=0))).T

def gradrelu(x):
    y = x
    y[y<=0] = 0
    y[y>0] = 1
    return y

def grad_descent(W_h, W_o, b_h, b_o, trainingData, trainingLabels, epoch, alpha, gamma):
    losses = []
    N = trainingLabels.shape[0]
    v1 = np.ones(W_o.shape) * (1e-5)
    v2 = np.ones(W_h.shape) * (1e-5)
    v3 = np.ones(b_o.shape) * (1e-5)
    v4 = np.ones(b_h.shape) * (1e-5)
    i = 0
    while(i < epoch):
        # Forward Pass
        s1 = computeLayer(trainingData, W_h, b_h)
        x1 = relu(s1)
        s2 = computeLayer(x1, W_o, b_o)
        x2 = softmax(s2)
        
        # Loss
        loss = CE(trainingLabels, x2)
        losses.append(loss)

        # Backprop
        dL_dWo = (1/N) * np.matmul(x1.T, (x2-trainingLabels))
        dL_dbo = (1/N) * np.sum(x2-trainingLabels, axis=0)
        q = np.multiply(gradrelu(s1), np.matmul(x2-trainingLabels, W_o.T))
        dL_dWh = (1/N) * np.matmul(trainingData.T, q)
        dL_dbh = (1/N) * np.sum(np.multiply(gradrelu(s1), np.matmul(x2-trainingLabels, W_o.T)), axis=0)

        # Update
        v1 = gamma * v1 + alpha * dL_dWo
        W_o = W_o - v1
        v2 = gamma * v2 + alpha * dL_dWh
        W_h = W_h - v2
        v3 = gamma * v3 + alpha * dL_dbo
        b_o = b_o - v3
        v4 = gamma * v4 + alpha * dL_dbh
        b_h = b_h - v4
        acc = 0
        #print(x2.shape)
        for j in range(x2.shape[0]):
            ind = np.argmax(x2[j])
            if(np.argmax(trainingLabels[j]) == ind):
                acc = acc + 1
        acc = acc/x2.shape[0]
        print("Epoch: {} | Loss: {} | Acc : {}".format(i, loss, acc))
        i = i+1
    return W_h, W_o, b_h, b_o, losses

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape(-1, 784)
    validData = validData.reshape(-1, 784)
    testData = testData.reshape(-1, 784)

    epochs = 200
    hidden_size = 1000
    gamma = 0.9
    alpha = 0.001

    W_h = np.random.normal(0, 2/(784+hidden_size), (trainData.shape[1], hidden_size))
    W_o = np.random.normal(0, 2/(hidden_size+10), (hidden_size, 10))

    b_h = np.random.normal(0, 2/(hidden_size), (hidden_size, 1))
    b_o = np.random.normal(0, 2/(10), (10, 1))
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    grad_descent(W_h, W_o, b_h, b_o, trainData, newtrain, epochs, alpha, gamma)
