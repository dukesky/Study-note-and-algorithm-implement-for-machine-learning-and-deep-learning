# Numpy implement of CNN model

In this post, I try to implement a  simple CNN model by numpy, the model has train and predict function, can forward and backward propagation, and I use [MNIST](#http://yann.lecun.com/exdb/mnist/) data set to test model training performance.
## Model Overview
To understanding what is CNN model, I have this [Deep Learning study note](https://github.com/dukesky/Study-note-and-algorithm-implement-for-machine-learning-and-deep-learning/tree/master/DeepLearning ), include my understanding of:

[Forward and Backward Propogation](https://github.com/dukesky/Study-note-and-algorithm-implement-for-machine-learning-and-deep-learning/tree/master/DeepLearning#simple-dnn) 

[CNN structure](https://github.com/dukesky/Study-note-and-algorithm-implement-for-machine-learning-and-deep-learning/tree/master/DeepLearning#cnn)

[Optimizers](https://github.com/dukesky/Study-note-and-algorithm-implement-for-machine-learning-and-deep-learning/tree/master/DeepLearning#different-kinds-of-optimizers)

This implement of CNN model is a basic CNN model with one convolution layer, one maxpooling layer, one flattern layer and one denselayer, and activation function is Relu for hidden layers and softmax for output layer.

The training parameters are as following:
```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 26, 26, 10)        100       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 10)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 1690)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                16910     
=================================================================
Total params: 17,010
Trainable params: 17,010
Non-trainable params: 0
```
## Parameters Initiate
Parameters are seperated in three parts:
- Parameter in input layer
    - Input image size(num_channel, image_x_size, image_y_size)
- Parameters in Concolution layer
    - Kernel(filter) (num_kernel, Kernel_size_x, kernel_size_y)
    - Kernel bias (num_Kernel,1)
    - Stride
- Parameters in Dense layer
    - output size
    - W (num_flatten_size, num_output)
    - b (num_output,1)

```py
class cnn():

    ## input size is a tuple=(n_color, input_x, input_y)
    def __init__(self,input_size,kernel_size,n_filters,output_size=1,stride=1):
        ## parameters in input layer
        self.input_size = input_size
        self.n_c = input_size[0]
        self.img_x = input_size[1]
        self.img_y = input_size[2]
        
        ## parameters in cnn layer
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.filter = np.random.uniform(size=(self.n_filters,self.n_c,self.kernel_size,self.kernel_size))
        self.bias = np.random.uniform(size=(self.n_filters))
        self.output_size = output_size
        self.stride = stride

        
        ## parameters in dense layer
        out_dim_x = int((input_size[1] - kernel_size+2)/stride) + 1
        out_dim_y = int((input_size[2] - kernel_size+2)/stride) + 1
        n_flatten_size = n_filters*input_size[0]*(int(out_dim_x/2) +int(out_dim_x%2))*(int(out_dim_y/2) +int(out_dim_y%2))
        self.W = np.random.uniform(-1,1,size = (n_flatten_size,output_size))
        self.b = np.random.uniform(size = (output_size,1))
```
And define basic functions **Relu**, **Softmax**

```py
def softmax(raw_preds): 
    preds = raw_preds-np.max(raw_preds)
    out = np.exp(preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.
    
def Relu(x):
    return np.maximum(0,x)
```

## Concolution and Flatten function



## Forward Propagation

```py
    def forward_propagation(self,x):
        cnn_out = Relu(self.convolution(x,self.filter,stride=self.stride))
        max_pool_out = self.maxpool(cnn_out)
        flatten_out = self.flatten(max_pool_out)
        output = self.W.T.dot(flatten_out)+self.b
    
        return softmax(output)
```

## Backward Propagation

## Gradient Algorithm

## Epoch and Parameter update


full code written by Jupyter notebook is [here](https://github.com/dukesky/Study-note-and-algorithm-implement-for-machine-learning-and-deep-learning/tree/master/DeepLearning/CNN)