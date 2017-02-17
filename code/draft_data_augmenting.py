#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:27:09 2017

@author: EmileMathieu
"""

import tensorflow as tf

# Randomly crop a [height, width] section of the image.
distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

# Randomly flip the image horizontally.
distorted_image = tf.image.random_flip_left_right(distorted_image)

# Because these operations are not commutative, consider randomizing
# the order their operation.
distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

# Subtract off the mean and divide by the variance of the pixels.
float_image = tf.image.per_image_standardization(distorted_image)

