#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(trainFeatures, trainLabels), (testFeatures, testLabels) = fashion_mnist.load_data()
accuracies = []

for i in range(1, 50, 2):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    placeholderTrainFeatures = tf.compat.v1.placeholder(trainFeatures.dtype, 
                                                        (None, 28, 28))
    placeholderTrainLabels = tf.compat.v1.placeholder(trainLabels.dtype, 
                                                      (None,))
    placeholderTest = tf.compat.v1.placeholder(testFeatures.dtype, 
                                               (28, 28))

    x = tf.cast(placeholderTrainFeatures, 'float32')
    y = tf.cast(placeholderTest, 'float32')
    substracted = tf.subtract(x, y)
    distance = tf.sqrt(tf.reduce_sum(tf.square(substracted), axis=(1, 2)))#2,3 for multidim

    _, indices = tf.nn.top_k(tf.negative(distance), k=i, sorted=False)
    top_k_labels = tf.gather(placeholderTrainLabels, indices)
    labels, _, counts = tf.unique_with_counts(top_k_labels)
    prediction = tf.gather(labels, tf.argmax(counts))

    accuracy = 0.
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for testFeature, testLabel in zip(testFeatures, testLabels):
            predicted = sess.run(prediction, 
                            feed_dict={placeholderTrainFeatures: trainFeatures[:],
                                       placeholderTrainLabels : trainLabels[:],
                                       placeholderTest: testFeature})

            if predicted == testLabel:
                accuracy += 1./ len(testFeatures)
        accuracies.append((i, accuracy))
        print("Done!")
        print("Accuracy:", accuracy)

