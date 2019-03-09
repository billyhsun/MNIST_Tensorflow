import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Hyperparameters
epochs = 50
batch_size = 32
learn_rate = 0.0001
regularization = 0.1
num_classes = 10

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def model_forward(x, weights, biases):

    # first conv2d layer
    x = tf.nn.conv2d(x, filter=weights['conv2d_filter1'], strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases['bias1'])

    # first relu layer
    x = tf.nn.relu(x)

    # batch normalization layer
    mean, variance = tf.nn.moments(x, axes=[0])
    x = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-8)

    # 2x2 max pooling layer
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # shape of maxpool = (batch_size, 14, 14, 32)
    # Flatten Layer
    x = tf.reshape(x, [-1, weights['fc1_weight'].get_shape().as_list()[0]])

    # fully connected layer 1 (784 output units)
    x = tf.add(tf.matmul(x, weights['fc1_weight']), biases['fc1_bias'])

    # dropout layer
    x = tf.layers.dropout(x, rate=0.9)

    # second RELU layer
    x = tf.nn.relu(x)

    # fully connected layer 2 (10 output units)
    x = tf.add(tf.matmul(x, weights['out_weight']), biases['out_bias'])

    # softmax the output
    res = tf.nn.softmax(x)

    return res


if __name__ == '__main__':

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    # reshape to 4D tensors for input into CNN
    trainData = np.expand_dims(trainData, 3)
    validData = np.expand_dims(validData, 3)
    testData = np.expand_dims(testData, 3)

    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, num_classes])

    weights = {
        'conv2d_filter1': tf.get_variable('W1', shape=(3, 3, 1, 32),
                                          initializer=tf.contrib.layers.xavier_initializer()),
        'fc1_weight': tf.get_variable('W2', shape=(32 * 14 * 14, 784),
                                      initializer=tf.contrib.layers.xavier_initializer()),
        'out_weight': tf.get_variable('W6', shape=(784, num_classes),
                                      initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'bias1': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'fc1_bias': tf.get_variable('B2', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
        'out_bias': tf.get_variable('B3', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer())
    }

    # convert to one-hot
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    pred = model_forward(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)) + \
           tf.multiply(
               tf.reduce_sum(tf.square(weights['conv2d_filter1'])) + tf.reduce_sum(tf.square(weights['fc1_weight']))
               + tf.reduce_sum(tf.square(weights['out_weight'])), regularization / 2)

    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate accuracy across all the given images and average them out.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_batches = int(trainData.shape[0] / batch_size)
        for epoch in range(epochs):

            shuffle(trainData, newtrain)

            for i in range(num_batches):
                train_Batch = trainData[i * batch_size: min((i+1)*batch_size, len(trainData))]
                train_Target_Batch = newtrain[i * batch_size: min((i+1)*batch_size, len(trainData))]

                sess.run(optimizer, feed_dict={x: train_Batch, y: train_Target_Batch })

            # at each epoch calculate the loss and accuracy
                train_loss, train_acc = sess.run([cost, accuracy], feed_dict={x: train_Batch,
                                                                  y: train_Target_Batch})

                #valid_loss, valid_acc = sess.run([cost, accuracy], feed_dict={x: validData, y: newvalid})
                #test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: testData, y: newtest})


            print("Epoch: {}, | Training loss: {:.5f} "
                      "Training Accuracy: {:.5f}  "
                      .format(epoch + 1, train_loss, train_acc))
