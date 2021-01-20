import numpy as np
from Network import Network

# Simple encoder-decoder network containing 3 layers: one recurrent layer
# for the encoder, one recurrent layer for the decoder and one dense layer
# for the output.
class EncoderDecoderNetwork(Network):
    def predict(self, x, y):
        # Builds input for the decoder.
        decoder_input = np.zeros_like(y)
        # Dummy token to indicate the start of the sequence.
        seq_start = -np.ones_like(y[:,0,:])
        decoder_input[:,0,:] = seq_start
        for t in range(y.shape[1] - 1):
            decoder_input[:,t+1,:] = y[:,t,:]
            
        internal_state = self.layers[0].forward(x)[:,-1,:]
        # Pass encoder's internal state at last timestep to the decoder.
        dense_input = self.layers[1].forward(decoder_input, internal_state)
        y_pred = self.layers[2].forward(dense_input)
        return y_pred
    
    def back_propagation(self, y_true, y_pred, learning_rate):
        delta = self.loss.compute_gradient(y_true, y_pred)
        delta, delta_param_dense = self.layers[2].backward(delta, learning_rate)
        ds_step, delta_param_decoder = self.layers[1].backward(delta, learning_rate)
        _, delta_param_encoder = self.layers[0].backward(delta, learning_rate, ds_step)
        return [delta_param_dense, delta_param_decoder, delta_param_encoder]
