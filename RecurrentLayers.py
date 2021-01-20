from CoreLayers import Layer
from Activation import Tanh
import numpy as np

class SimpleRNNLayer(Layer):
    def __init__(self, num_units, input_size):
        # input_size is keras like: (timesteps, num_features)
        self.timesteps, self.num_features = input_size[0], input_size[1]
        self.num_units = num_units
        self.state_activation = Tanh()
        # Parameters
        self.W_input = np.random.randn(self.num_features, self.num_units) \
                       * np.sqrt(1.0/self.num_features)
        self.W_state = np.random.randn(self.num_units, self.num_units) \
                       * np.sqrt(1.0/self.num_units)
        self.bias_state = np.zeros((1, self.num_units))
        
    def get_num_params(self):
        return 3
        
    def get_params(self):
        return [np.copy(self.W_input), np.copy(self.W_state), \
                np.copy(self.bias_state)]
    
    def set_params(self, params):
        [W_i, W_s, b_s] = params
        self.W_input = np.copy(W_i)
        self.W_state = np.copy(W_s)
        self.bias_state = np.copy(b_s)
    
    # Unfolds the recurrent layer and returns computed states.
    # input shape = (batch_size, timesteps, input_size)
    def forward(self, x, initial_states = None, training=False):
        self.batch_size = x.shape[0]
        self.inputs = x
        self.initial_states = initial_states
        # Initialize internal state
        self.states = np.zeros((self.batch_size, self.timesteps, self.num_units), \
                               dtype=np.float32)
        if self.initial_states is None:
            self.initial_states = np.zeros((self.batch_size, self.num_units), \
                                           dtype=np.float32)
        for t in range(self.timesteps): 
            if t == 0:
                prev_state = self.initial_states
            else:
                prev_state = self.states[:,t-1,:]
            self.states[:,t,:] = self.state_activation.apply( \
                                 np.dot(prev_state, self.W_state) \
                                 + np.dot(x[:,t,:], self.W_input) \
                                 + self.bias_state)
        return self.states

    # delta_out has shape (batch_size, timesteps, output_size)
    def backward(self, delta_out = None, learning_rate = 0.001, ds_step = None): 
        if delta_out is None:
            delta_out = np.zeros((self.batch_size, self.timesteps, self.num_features))
        if ds_step is None:
            ds_step = np.zeros((self.batch_size, self.num_units))
        dW_input = np.zeros_like(self.W_input)
        dW_state = np.zeros_like(self.W_state)
        db_state = np.zeros_like(self.bias_state)
        for t in reversed(range(self.timesteps)):
            ds = delta_out[:,t,:] + ds_step
            ds_rec = (1 - self.states[:,t,:] * self.states[:,t,:]) * ds
            # Calculate accumulative parameter gradients
            dW_input += np.dot(self.inputs[:,t,:].T, ds_rec)
            if t == 0:
                state = self.initial_states
            else:
                state = self.states[:,t-1,:]
            dW_state += np.dot(state.T, ds_rec)
            db_state += np.sum(ds_rec, axis=0)
            ds_step = np.dot(ds_rec, self.W_state.T)
            
        # Update parameters
        self.W_input -= learning_rate * dW_input
        self.W_state -= learning_rate * dW_state
        self.bias_state -= learning_rate * db_state
        return ds_step, [dW_input, dW_state, db_state]
    