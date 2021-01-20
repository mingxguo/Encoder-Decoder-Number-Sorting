import numpy as np
from EncoderDecoderNetwork import EncoderDecoderNetwork
from RecurrentLayers import SimpleRNNLayer
from CoreLayers import DenseLayer
from Loss import *

"""
Training encoder-decoder network for the number sorting problem.

Approach:
Given an array 'x' of numbers to be sorted. To ensure that the output are
at least numbers of the original array 'x', we let the network predict the indices
that would sort 'x' (argsort) and post process these indices to get a sorted array.
Since indices are integers ranging from 0 to the array length, we can use one hot
encoding for the labels 'y' and softmax cross entropy as loss function.

    Example of one pair of data:
    array x = [3, 9, 2, 7, 0]
    label y = [4, 2, 0, 3, 1]
    
In this script we use encoder-decoder recurrent neural network to sort arrays
of fixed length. The representation of input arrays is not restricted. They can
contain integers or real numbers of various range. 

Below are two different models to sort arrays of length 10. The first model get
input arrays ranging from 0 to 50, encoded with one-hot. The second model get
input arrays sampled from standard normal distribution.
"""

#%%

def one_hot_encode(y, batch_size, timesteps, num_outputs):
    encoded = np.zeros((batch_size, timesteps, num_outputs))
    for i in range(batch_size):
        for j in range(timesteps):
            encoded[i][j][y[i][j]] = 1
    return encoded

def one_hot_decode(y, batch_size, timesteps):
    decoded = np.zeros((batch_size, timesteps, 1), dtype=np.int32)
    for i in range(batch_size):
        for j in range(timesteps):
            digit = np.argmax(y[i][j])
            decoded[i][j] = digit
    return decoded

def batch_gen(batch_size, seq_len, max_no):
    x = np.random.randint(max_no, size=(batch_size, seq_len, 1))
    y = np.argsort(x, axis=1)
    x = one_hot_encode(x,batch_size, seq_len, max_no)
    y = one_hot_encode(y,batch_size, seq_len, seq_len)
    return x, y

def batch_gen2(batch_size, seq_len):
    x = np.random.randn(batch_size, seq_len, 1)
    y = np.argsort(x, axis=1)
    y = one_hot_encode(y, batch_size, seq_len, seq_len)
    return x, y

# Returns array x sorted according to index.
def sort(x, index, batch_size, timesteps):
  index = one_hot_decode(index, batch_size, timesteps).reshape(timesteps,)
  y = np.zeros_like(x)
  for i in range(x.shape[0]):
      y[i] = x[index[i]]
  return y

def accuracy(y_true, y_pred):
    batch_size, timesteps = y_true.shape[0], y_true.shape[1]
    return np.mean(one_hot_decode(y_true, batch_size, timesteps) == \
                   one_hot_decode(y_pred, batch_size, timesteps))

#%%
""" Model for integer inputs """
# Hyperparameters
epochs = 500000
learning_rate = 0.001
batch_size = 64
timesteps = 10
max_no = 50
num_units = 100

# Build network architecture
nn = EncoderDecoderNetwork()
nn.add(SimpleRNNLayer(num_units, (timesteps, max_no))) # encoder RNN
nn.add(SimpleRNNLayer(num_units, (timesteps, timesteps))) # decoder RNN
nn.add(DenseLayer(num_units, timesteps))
nn.build(SoftmaxCrossEntropy(), accuracy)

i = 0
while i <= epochs:
    x_train, y_train = batch_gen(batch_size, timesteps, max_no)
    loss, acc = nn.train_on_batch(x_train, y_train, learning_rate)
    if i % 250 == 0:
        print("Iteration", i, "loss", loss, "acc", acc)
    if i % 2000 == 0 and i != 0:
        #decay learning rate
        if learning_rate > 1e-6:
            learning_rate = learning_rate / 2
        x_test, y_test = batch_gen(1, timesteps, max_no)
        y_pred = nn.test(x_test, y_test)
        x_test = one_hot_decode(x_test, 1, timesteps).reshape(timesteps,)
        print("x", x_test)
        print("Real sort", sort(x_test, y_test, 1, timesteps))
        print("Predicted sort", sort(x_test, y_pred, 1, timesteps))
    i += 1
nn.save_model("encoder_decoder_int.txt")

# Last iterations
# --------------------------------------------------------
#Iteration 498250 loss 0.19627530660069842 acc 0.91875
#Iteration 498500 loss 0.15698993464164293 acc 0.9421875
#Iteration 498750 loss 0.16722319123648735 acc 0.9359375
#Iteration 499000 loss 0.16767385189762296 acc 0.9328125
#Iteration 499250 loss 0.1784334328965363 acc 0.9265625
#Iteration 499500 loss 0.15010864181201 acc 0.940625
#Iteration 499750 loss 0.18828350514105704 acc 0.915625
#Iteration 500000 loss 0.156750232090343 acc 0.94375
#test: loss 0.071459, acc 1.000000
#x [ 0 47 30 30  3 23 42 31 11 43]
#Real sort [ 0  3 11 23 30 30 31 42 43 47]
#Predicted sort [ 0  3 11 23 30 30 31 42 43 47]

#%%

# Load saved model and make a prediction.
# ---------------------------------------
trained_model = EncoderDecoderNetwork()
trained_model.add(SimpleRNNLayer(num_units, (timesteps, max_no))) # encoder RNN
trained_model.add(SimpleRNNLayer(num_units, (timesteps, timesteps))) # decoder RNN
trained_model.add(DenseLayer(num_units, timesteps))
trained_model.build(SoftmaxCrossEntropy(), accuracy)
trained_model.load_model("encoder_decoder_int.txt")

x_test, y_test = batch_gen(1, timesteps, max_no)
y_pred = trained_model.test(x_test, y_test)

x_test = one_hot_decode(x_test, 1, timesteps).reshape(timesteps,)
print("x", x_test)
print("Real sort", sort(x_test, y_test, 1, timesteps))
print("Predicted sort", sort(x_test, y_pred, 1, timesteps))

#%%
""" Model for standard uniform distributed inputs """
# Hyperparameters
epochs = 500000
learning_rate = 0.001
batch_size = 64
timesteps = 10
num_units = 100

# Build network architecture
nn = EncoderDecoderNetwork()
nn.add(SimpleRNNLayer(num_units, (timesteps, 1))) # encoder RNN
nn.add(SimpleRNNLayer(num_units, (timesteps, timesteps))) # decoder RNN
nn.add(DenseLayer(num_units, timesteps))
nn.build(SoftmaxCrossEntropy(), accuracy)

i = 0
while i <= epochs:
    x_train, y_train = batch_gen2(batch_size, timesteps)
    loss, acc = nn.train_on_batch(x_train, y_train, learning_rate)
    if i % 250 == 0:
        print("Iteration", i, "loss", loss, "acc", acc)
    if i % 2000 == 0 and i != 0:
        #decay learning rate
        if learning_rate > 1e-6:
            learning_rate = learning_rate / 2
        x_test, y_test = batch_gen2(1, timesteps)
        y_pred = nn.test(x_test, y_test)
        x_test = x_test.reshape(timesteps,)
        print("x", x_test)
        print("Real sort", sort(x_test, y_test, 1, timesteps))
        print("Predicted sort", sort(x_test, y_pred, 1, timesteps))
    i += 1

# Last iterations
# --------------------------------------------------------
#iter 98250 loss 0.29339287592202773 acc 0.878125
#iter 98500 loss 0.33747890368054134 acc 0.8609375
#iter 98750 loss 0.3364023210457451 acc 0.8421875
#iter 99000 loss 0.31988857699685114 acc 0.8734375
#iter 99250 loss 0.327678642077999 acc 0.8703125
#iter 99500 loss 0.31541331560039815 acc 0.8734375
#iter 99750 loss 0.3239279964625597 acc 0.859375
#test: loss 0.635576, acc 0.800000
#x [ 0.34435131 -0.4681474   0.2305473  -0.53832086 -0.39418798 -0.27966765
# -0.5174107  -0.339543    0.29630575  2.07622973]
#Real sort [-0.53832086 -0.5174107  -0.4681474  -0.39418798 -0.339543   -0.27966765
#  0.2305473   0.29630575  0.34435131  2.07622973]
#Predicted sort [-0.4681474  -0.4681474  -0.4681474  -0.39418798 -0.339543   -0.27966765
#  0.2305473   0.29630575  0.34435131  2.07622973]