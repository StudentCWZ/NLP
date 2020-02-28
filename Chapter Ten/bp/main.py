#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-28 12:05:39
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-28 12:05:54
'''


import bp
import mnist_loader

net = bp.Network([784, 100, 10])


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
