import numpy as np

class Layer:
    def __init__(self):
        pass
    
    def get_num_params(self):
        raise NotImplementedError
        
    def get_params(self):
        raise NotImplementedError
    
    def set_params(self, params):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, delta_out, learning_rate):
        raise NotImplementedError
    
    def save_params(self, file):
        params = self.get_params()
        for param in params:
            np.save(file, param)
            
    def load_params(self, file):
        params = []
        for i in range(self.get_num_params()):
            params.append(np.load(file))
        self.set_params(params)

class DenseLayer(Layer):
    def __init__(self, num_inputs, num_outputs):
        self.weights = np.random.randn(num_inputs, num_outputs) * 0.001#np.sqrt(1.0/num_inputs)
        self.bias = np.zeros((1, num_outputs))
        
    def get_num_params(self):
        return 2

    def get_params(self):
        return [np.copy(self.weights), np.copy(self.bias)]
    
    def set_params(self, params):
        [W, b] = params
        self.weights = np.copy(W)
        self.bias = np.copy(b)
        
    # x has shape (d0, d1, ..., num_inputs)
    def forward(self, x):
        self.input = x
        product = np.tensordot(x, self.weights, axes=1)
        return product + self.bias

    # delta_out has shape (d0, d1, ..., num_outputs)
    def backward(self, delta_out, learning_rate):
        axes_size = len(self.input.shape)-1
        dW = np.tensordot(self.input, delta_out, axes=(range(axes_size), range(axes_size)))
        db = np.sum(delta_out, axis=tuple(range(axes_size))).reshape(self.bias.shape)
        delta_in = np.tensordot(delta_out, self.weights.T, axes=1)
        # Updates parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        return delta_in, [dW, db]
