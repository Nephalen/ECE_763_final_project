import numpy as np
import torch

def categorical_accuracy(predict, target):
    return np.mean(predict == target)