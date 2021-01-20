import numpy as np

class Loss:
    def __init__(self):
        pass
    def compute_loss(self, y_true, y_pred):
        pass
    def compute_gradient(self, y_true, y_pred):
        pass
    
def softmax(x):
    exp_y = np.exp(x - np.max(x)) # numerial stability
    return exp_y / np.sum(exp_y, axis=-1, keepdims=True)
    
class SoftmaxCrossEntropy(Loss):
    def compute_loss(self, y_true, y_pred):
        softmax_y = softmax(y_pred)
        return np.sum(-y_true * np.log(softmax_y+1e-15), axis=-1)
    
    def compute_gradient(self, y_true, y_pred):
        softmax_y = softmax(y_pred)
        return softmax_y-y_true
    
class MeanSquaredError(Loss):
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2, axis=-1)
    
    def compute_gradient(self, y_true, y_pred):
        return np.mean(-2*(y_true - y_pred), axis=-1, keepdims=True)
