# Encoder-Decoder-Number-Sorting
Implementation of an encoder-decoder architecture for number sorting using only numpy.

## Problem approach
Given an array 'x' of numbers to be sorted. To ensure that the output are
at least numbers of the original array 'x', we could let the network predict the indices
that would sort 'x' ([argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)) 
and post process these indices to get a sorted array.
Since indices are integers ranging from 0 to the array length, we can use one hot
encoding for the labels 'y' and softmax cross entropy as loss function.

Example of one pair of data:

    x = [3, 9, 2, 7, 0]
    y = [4, 2, 0, 3, 1]
    
## Trained models
    
In the script `NumberSortingTest.py` I use a simple encoder-decoder recurrent neural network 
to sort arrays of fixed length. The representation of input arrays is not restricted. 
They can contain integers or real numbers of various range. 

I trained two different models to sort arrays of length 10. The first model get
input arrays ranging from 0 to 50, encoded with one-hot. The second model get
input arrays sampled from standard normal distribution. 


### Model performance
After training the first model for 500000 batches, loss stays between 0.15-0.20 and accuracy > 0.90.
The last lines of output are:

```
Iteration 498250 loss 0.19627530660069842 acc 0.91875
Iteration 498500 loss 0.15698993464164293 acc 0.9421875
Iteration 498750 loss 0.16722319123648735 acc 0.9359375
Iteration 499000 loss 0.16767385189762296 acc 0.9328125
Iteration 499250 loss 0.1784334328965363 acc 0.9265625
Iteration 499500 loss 0.15010864181201 acc 0.940625
Iteration 499750 loss 0.18828350514105704 acc 0.915625
Iteration 500000 loss 0.156750232090343 acc 0.94375

test: loss 0.071459, acc 1.000000
x [ 0 47 30 30  3 23 42 31 11 43]
Real sort [ 0  3 11 23 30 30 31 42 43 47]
Predicted sort [ 0  3 11 23 30 30 31 42 43 47]
```

The second model has not finished training yet and its current performance is:

```
iter 98250 loss 0.29339287592202773 acc 0.878125
iter 98500 loss 0.33747890368054134 acc 0.8609375
iter 98750 loss 0.3364023210457451 acc 0.8421875
iter 99000 loss 0.31988857699685114 acc 0.8734375
iter 99250 loss 0.327678642077999 acc 0.8703125
iter 99500 loss 0.31541331560039815 acc 0.8734375
iter 99750 loss 0.3239279964625597 acc 0.859375

test: loss 0.635576, acc 0.800000
x [ 0.34435131 -0.4681474   0.2305473  -0.53832086 -0.39418798 -0.27966765
 -0.5174107  -0.339543    0.29630575  2.07622973]
Real sort [-0.53832086 -0.5174107  -0.4681474  -0.39418798 -0.339543   -0.27966765
  0.2305473   0.29630575  0.34435131  2.07622973]
Predicted sort [-0.4681474  -0.4681474  -0.4681474  -0.39418798 -0.339543   -0.27966765
  0.2305473   0.29630575  0.34435131  2.07622973]
```

### Model loading

The txt file `encoder_decoder_int.txt` contains the first model and can be loaded as follows:

```
model = EncoderDecoderNetwork()
model.add(SimpleRNNLayer(num_units, (timesteps, max_no))) # encoder RNN
model.add(SimpleRNNLayer(num_units, (timesteps, timesteps))) # decoder RNN
model.add(DenseLayer(num_units, timesteps))
model.build(SoftmaxCrossEntropy(), accuracy)
model.load_model("encoder_decoder_int.txt")
```
