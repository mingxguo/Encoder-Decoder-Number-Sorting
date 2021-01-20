import numpy as np

def default_acc(y_true, y_pred):
    return np.mean(y_true == y_pred)

class Network:
    def __init__(self):
        self.layers = []
    
    # Adds a layer to the network.
    def add(self, layer):
        self.layers.append(layer)
        
    # Equips the network with a loss object and an accuracy function.
    def build(self, loss, accuracy = default_acc):
        self.loss = loss
        self.accuracy = accuracy
        
    # Returns network parameters as a list.
    def get_params(self):
        params = []
        for layer in self.layers:
            param = layer.get_params()
            params.append(param)
        return params

    # Sets network parameters to the given ones.
    def set_params(self, params):
        for i in range(len(self.layers)):
            layer_params = params[i]
            self.layers[i].set_params(layer_params)     
    
    # Saves network parameters to file with file_name.
    def save_model(self, file_name):
        file = open(file_name, 'wb')
        for layer in self.layers:
            layer.save_params(file)
        file.close()
        
    # Loads network parameters from file with file_name.
    def load_model(self, file_name):
        file = open(file_name, 'rb')
        for layer in self.layers:
            layer.load_params(file)
        file.close()
        
    # Predicts the output given the input x.
    def predict(self, x, y):
        y_pred = x
        for layer in self.layers:
            y_pred = layer.forward(y_pred)
        return y_pred
      
    # Backpropagates loss delta through the network and updates its parameters.
    def back_propagation(self, y_true, y_pred, learning_rate):
        delta_params = []
        delta = self.loss.compute_gradient(y_true, y_pred)
        for layer in reversed(self.layers):
            delta, delta_param = layer.backward(delta, learning_rate)
            delta_params += [delta_param]
        return delta_params

    # Trains the network on one batch of data.
    def train_on_batch(self, x_batch, y_batch, learning_rate):
        y_pred = self.predict(x_batch, y_batch)
        self.back_propagation(y_batch, y_pred, learning_rate)
        loss = np.mean(self.loss.compute_loss(y_batch, y_pred))
        acc = self.accuracy(y_batch, y_pred)
        return loss, acc
    
    # Trains the network with batches of the data for a given number of epochs.
    def train(self, x_train, y_train, epochs, batch_size, learning_rate):
        num_batches = x_train.shape[0] // batch_size
        for i in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for j in range(num_batches):
                x_batch = x_train[j*batch_size: (j+1)*batch_size]
                y_batch = y_train[j*batch_size: (j+1)*batch_size]
                
                loss, acc = self.train_on_batch(x_batch, y_batch, learning_rate)
                epoch_loss += loss
                epoch_acc += acc
            # Print epoch summary
            print("Epoch %d: loss %f, acc %f" % (i, epoch_loss/num_batches, \
                                                 epoch_acc/num_batches))

    # Prints the loss and accuracy of the test and returns network's prediction.
    def test(self, x_test, y_test):
        y_pred = self.predict(x_test, y_test)
        loss = np.mean(self.loss.compute_loss(y_test, y_pred))
        acc = self.accuracy(y_test, y_pred)
        print("Test loss %f, acc %f" % (loss, acc))
        return y_pred
