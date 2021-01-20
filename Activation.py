import numpy as np

class Activation:
    def __init__(self):
        pass
    def apply(self, x):
        pass
    def apply_gradient(self, x):
        pass
    
class Tanh(Activation):
    def apply(self, x):
        return np.tanh(x)
    
    def apply_gradient(self, x): 
        return 1-np.tanh(x)**2
