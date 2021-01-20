import numpy as np
from Network import Network
from CoreLayers import DenseLayer
from RecurrentLayers import SimpleRNNLayer
from Loss import SoftmaxCrossEntropy

def network_params_to_vector(network_params):
    param_vector = []
    param_shapes_per_layer = []
    param_sizes_per_layer = []
    for layer_params in network_params:
        layer_param_vector, layer_param_shapes, layer_param_total_size = \
            layer_params_to_vector(layer_params)
        param_vector.append(layer_param_vector)
        param_shapes_per_layer.append(layer_param_shapes)
        param_sizes_per_layer.append(layer_param_total_size)
    param_vector = np.concatenate(param_vector)
    return param_vector, param_shapes_per_layer, param_sizes_per_layer
        
def vector_to_network_params(param_vector, param_shapes_per_layer, param_sizes_per_layer):
    network_params = []
    offset = 0
    for i in range(len(param_shapes_per_layer)):
        layer_param_vector = param_vector[offset:offset+param_sizes_per_layer[i]]
        layer_params = vector_to_layer_params(layer_param_vector, param_shapes_per_layer[i])
        network_params += [layer_params]
        offset += param_sizes_per_layer[i]
    return network_params

def layer_params_to_vector(params):
    param_vector = []
    param_shapes = []
    total_size = 0
    for param in params:
        param_vector.append(param.reshape(-1))
        param_shapes.append(param.shape)
        total_size += param.shape[0] * param.shape[1]
    param_vector = np.concatenate(param_vector)
    return param_vector, param_shapes, total_size

def vector_to_layer_params(param_vector, param_shapes):
    params = []
    offset = 0
    for param_shape in param_shapes:
        param_length = param_shape[0] * param_shape[1]
        param = param_vector[offset:offset+param_length].reshape(param_shape)
        params += [param]
        offset += param_length
    return params

def calculate_numerical_grad(x, y, network, loss, param_plus_eps, \
                             param_minus_eps, param_shapes_per_layer, \
                             param_sizes_per_layer):
    # Calculate output at param + epsilon
    params = vector_to_network_params(param_plus_eps, param_shapes_per_layer, param_sizes_per_layer)
    network.set_params(params)
    y_pos = network.predict(x, y)
    loss_pos = loss(y, y_pos)
    # Calculate output at param - epsilon   
    params = vector_to_network_params(param_minus_eps, param_shapes_per_layer, param_sizes_per_layer)   
    network.set_params(params)
    y_neg = network.predict(x, y)
    loss_neg = loss(y, y_neg)
    
    numerical_grad = (loss_pos-loss_neg)/(2.0*epsilon)
    return numerical_grad
    
# Gradient checker for a network.
def gradient_check(x, y, network, loss, epsilon):
    # Save initial parameters of the network before backward update
    p = network.get_params()
    param_vector, param_shapes_per_layer, param_sizes_per_layer = \
                    network_params_to_vector(p)
    # Get delta of parameters calculated by the layer in one step
    y_pred = network.predict(x, y)
    delta_parameters = network.back_propagation(y, y_pred, learning_rate = 0.1) # learning_rate irrelevant
    # Delta of the parameters are returned in reverse order of the layers
    delta_param_vector, _, _ = network_params_to_vector(list(reversed(delta_parameters)))
    numerical_grad = np.zeros_like(param_vector)
    # Iterate through each component of all parameters
    for i in range(len(param_vector)):
        eps = np.zeros_like(param_vector)
        eps[i] = epsilon
        
        param_plus_eps = param_vector + eps
        param_minus_eps = param_vector - eps
        numerical_grad[i] = calculate_numerical_grad(x, y, network, loss, \
                                                     param_plus_eps, \
                                                     param_minus_eps, \
                                                     param_shapes_per_layer, \
                                                     param_sizes_per_layer)
        
        # Single component difference check
        diff = np.abs(numerical_grad[i] - delta_param_vector[i])
        if diff > epsilon:
            print("Difference at component %d is %f" % (i, diff))
    # General difference check
    diff = np.linalg.norm(numerical_grad - delta_param_vector) / \
            (np.linalg.norm(numerical_grad) + np.linalg.norm(delta_param_vector))
    if diff > epsilon:
        print("General difference is", diff)
                
#%%        
""" 
Gradient check for a network with one dense layer, using softmax cross entropy loss.
"""

# One hot encoding for data of size (batch_size, 1).
def one_hot(y, num_outputs):
    encoded = np.zeros((y.shape[0], num_outputs))
    for i in range(y.shape[0]):
        encoded[i][y[i]] = 1
    return encoded

# Generates a random pair of data and target.
# data: sampled from standard normal distribution, size (1, num_inputs).
# target: size (1, num_outputs) in one_hot encoding.
def data_gen(num_inputs, num_outputs):
    x = np.random.randn(1, num_inputs)
    y = np.random.randint(0, num_outputs, size=(1,1))
    y = one_hot(y, num_outputs)
    return x, y

#  --------- HYPER PARAMETERS --------------
iterations = 10
num_inputs = 100
num_outputs = 100
epsilon= 1e-4

model = Network()
model.add(DenseLayer(num_inputs, num_outputs))
model.build(SoftmaxCrossEntropy())

for i in range(iterations):
    print("Iteration", i+1)
    x_test, y_test = data_gen(num_inputs, num_outputs)
    gradient_check(x_test, y_test, model, SoftmaxCrossEntropy().compute_loss, epsilon)
    
#%%

""" 
Gradient check for a network with one simple RNN layer and one dense layer,
using softmax cross entropy loss.
"""

# Loss function for sequencing data.
def loss(y_true, y_pred):
    return np.sum(SoftmaxCrossEntropy().compute_loss(y_true, y_pred), axis=-1)
    
# One hot encoding for sequencing data of size (batch_size, timesteps, 1).
def one_hot(y, batch_size, timesteps, num_outputs):
    encoded = np.zeros((batch_size, timesteps, num_outputs))
    for i in range(batch_size):
        for j in range(timesteps):
            encoded[i][j][y[i][j]] = 1
    return encoded

# Generates a random pair of sequencing data and target.
# data: sampled from standard normal distribution, size (1, timesteps, num_inputs).
# target: size (1, num_outputs) in one_hot encoding.
def data_gen(timesteps, num_inputs, num_outputs):
    x = np.random.randn(1, timesteps, num_inputs)
    y = np.random.randint(0, num_outputs, size=(1,timesteps,1))
    y = one_hot(y, 1, timesteps, num_outputs)
    return x, y

#  --------- HYPER PARAMETERS --------------
iterations = 10
num_units = 10
timesteps = 5
num_inputs = 50
num_outputs = 50
epsilon= 1e-3

from Network import Network
model = Network()
model.add(SimpleRNNLayer(num_units, (timesteps, num_inputs)))
model.add(DenseLayer(num_units, num_outputs))
model.build(SoftmaxCrossEntropy())

for i in range(iterations):
    print("Iteration", i+1)
    x_test, y_test = data_gen(timesteps, num_inputs, num_outputs)
    gradient_check(x_test, y_test, model, loss, epsilon)