# **Deep Learning**
[Introduction](#Introduction)

[Gradient Descent](#gradient-descent)

[when to use deep learning](#when-to-use-deep-learning)

[MMSE](#mmse)

[Decision Theory](#decision-theory)

[MAP](#7928-1567186463618)

[ML](#1993-1567189779980)

[Entropy](#5410-1567189884927)

[Regulatization](#1575-1567189974024)

[Train/Test Split](#7521-1567190547514)

[Simple DNN](#9123-1567190535988)

[Forward Propagation](#1453-1567149114098)

[Back Propagation](#2985-1567149114416)

[idea](#3649-1566969003923)

[summary](#2172-1567208589376)

[regularizer](#2993-1567208764349)

[Multi-Layer Perceptron](#4570-1567209192600)

[A Big Picture](#8540-1566968770087)

[Activation Function](#5765-1567210487712)

[Cost Function](#6063-1567211450851)

[Optimizer](#1010-1567211895158)

[Regularization](#3949-1567225394632)

[Dropout](#4623-1567225772323)

[Normalization](#8112-1567228892584)

[Dimensionality Reduction](#8876-1567228892762)

[PCA](#1098-1567228892885)

[LDA](#3712-1567276007695)

[Hyper parameter](#6887-1567361193398)

[Universal Approximation Theorem](#7660-1567276219530)

[CNN](#7974-1567363066859)

[Introduction](#5962-1567402688177)

[Kernel(Filter)](#8472-1567453876673)

[Conv layer](#3748-1567276235052)

[Pooling layer](#3079-1567465440089)

[De-Conv(Transpose convolution) layer](#3228-1567453882535)

[Transposed 2D convolution with no padding, stride of 2 and kernel of 3](#3986-1567634359536)

[Group Convolution](#3361-1567647496745)

[Seperable Convolutions](#8820-1567647808750)

[Full Connect layer](#8992-1567647050727)

[# of Parameters computation](#4080-1567544290778)

[Visualization](#1512-1567626353562)

[Different Nets](#5650-1567544298601)

[LeNet(1998)](#6542-1567630247173)

[AlexNet(2012)](#4872-1567544325127)

[VGG](#3016-1567630339075)

[Resnet](#6320-1567544338806)

[GoogleNet (Inception)](#2268-1567626564512)

[MobileNet](#6124-1567630748011)

[Comparation](#8150-1567710768197)

[CNN in NLP](#4072-1567631300168)

[Coding](#6832-1567712454135)

[Keras](#4059-1567559736866)

[RNN](#4888-1566955799496)

[Introduction](#5735-1591211628380)

[Vanish Gradient Problem](#1888-1591211806350)

[LSTM](#3853-1591229892046)

[GRU](#3289-1591237107911)

[Back-Propagation through Time (BPTT)](#1891-1591226440731)

[Word Embedding](#2071-1591211806725)

[BERT](#8553-1567149183370)

[TensorFlow and Keras](#9326-1566968927669)

[sequential model](#5571-1566968973805)

[Pytorch](#4828-1566968942504)



## **Introduction**

### **Gradient Descent**

Steepest Gradient Descent

![Steepest Gradient Descent](../pic/微信截图_20190829094647.png)

Stochastic Gradient Descent

![](../pic/微信截图_20190829094840.png)

single point stochastic gradient descent

![](../pic/微信截图_20190830000924.png)

### **when to use deep learning**

Don't

![](../pic/微信截图_20190829100044.png)

Do

![](../pic/微信截图_20190829100059.png)

### **MMSE**

![](../pic/微信截图_20190829102637.png)

![](../pic/微信截图_20190829102852.png)

### **Decision Theory**

![](../pic/微信截图_20190830001140.png)

![](RackMultipart20200609-4-1vghqmz_html_b3f4bac3c68b4815.png)

### **MAP**

![](RackMultipart20200609-4-1vghqmz_html_18fe83c9f4be1a5e.png)

### **ML**

![](RackMultipart20200609-4-1vghqmz_html_29665d19846bf31c.png)

![](RackMultipart20200609-4-1vghqmz_html_21b80969e3531f6.png)

### **Entropy**

![](RackMultipart20200609-4-1vghqmz_html_8da56602c46b5245.png)

### **cross entropy**

![](RackMultipart20200609-4-1vghqmz_html_bf5cb1284660f2.png)

### **binary cross entropy**

![](RackMultipart20200609-4-1vghqmz_html_77e47b852ac63f18.png)

### **Regulatization**

enforce penalty on weights to bias toward a prior distribution. --\&gt; to reduce over-fitting (by smaller weight)

![](RackMultipart20200609-4-1vghqmz_html_db726c0bea78679c.png)

![](RackMultipart20200609-4-1vghqmz_html_aa37b368896c0ba3.png)

### **Train/Test Split**

Training Set -- for define trainable parameters

Validation Set -- for define hyper-parameters

Test Set -- for verify model performance

Mini- Batch -- for do one SGD update per mini-batch

epoch -- one training run through all data set

iteration -- number of mini-batches per epoch

## **Simple DNN**

1. Input an example from a dataset.
2. The network will take that example and apply some complex computations to it using randomly initialised variables (called weights and biases).
3. A predicted result will be produced.
4. Comparing that result to the expected value will give us an error.
5. Propagating the error back through the same path will adjust the variables.
6. Steps 1–5 are repeated until we are confident to say that our variables are well-defined.
7. A predication is made by applying these variables to a new unseen input.

![](RackMultipart20200609-4-1vghqmz_html_593d7265c38aaa0a.png)

### **Forward Propagation**

### **Back Propagation**

### **idea**

![](RackMultipart20200609-4-1vghqmz_html_aab59b6a6559d55a.png)

To be specified, one step of forward neural net can be extracted as follow:

![](RackMultipart20200609-4-1vghqmz_html_78be517db7bedf90.png)

![](RackMultipart20200609-4-1vghqmz_html_7473f83e3c859d70.png)

![](RackMultipart20200609-4-1vghqmz_html_31b2abbc3309ab7a.png)

![](RackMultipart20200609-4-1vghqmz_html_3d160f94249cacc2.png)

![](RackMultipart20200609-4-1vghqmz_html_6d471208f6d3a84e.png)

![](RackMultipart20200609-4-1vghqmz_html_60124f597cfba9b9.png)

work for matrix (vector case)

![](RackMultipart20200609-4-1vghqmz_html_ba7eb470ecdf0e55.png)

![](RackMultipart20200609-4-1vghqmz_html_aa41cbdcadded46c.png)

![](RackMultipart20200609-4-1vghqmz_html_af39175dcc414a86.png)

### **summary**

![](RackMultipart20200609-4-1vghqmz_html_6c880c0f2355fd49.png)

![](RackMultipart20200609-4-1vghqmz_html_4ad67bab355f62fe.png)

### **using batch**

so it&#39;s like compute several data points with same **w** and **δ** and use average as the update delta

![](RackMultipart20200609-4-1vghqmz_html_dd452a41409226cd.png)

### **regularizer**

![](RackMultipart20200609-4-1vghqmz_html_aa36bef44448933e.png)

## **Multi-Layer Perceptron**

### **A Big Picture**

![](RackMultipart20200609-4-1vghqmz_html_83036332887d36d4.png)

### **Activation Function**

### **middle layers Activation Function**

sigmoid &amp; tanh

![](RackMultipart20200609-4-1vghqmz_html_a1bc06a8a3100d73.png)

problem: vanish

![](RackMultipart20200609-4-1vghqmz_html_8d38e86f14f8b11f.png)

more important: **ReLU family -- will not vanish**

![](RackMultipart20200609-4-1vghqmz_html_239029d623799778.png)

![](RackMultipart20200609-4-1vghqmz_html_8dfcf8dbc61619a4.png)

![](RackMultipart20200609-4-1vghqmz_html_c8532893f22ac0d0.png)

### **Output Activation Function**

![](RackMultipart20200609-4-1vghqmz_html_c6026c15e8df0c1d.png)

### **Cost Function**

Cross-entropy Cost

One Hot label vs. Softmax

Quadratic Cost (MSE)

![](RackMultipart20200609-4-1vghqmz_html_8b9921d4ba05dfe4.png)

![](RackMultipart20200609-4-1vghqmz_html_3329add99a652ba8.png)

### **Optimizer**

**Hessian**

![](RackMultipart20200609-4-1vghqmz_html_2539257a35f93b01.png)

**Condition**

![](RackMultipart20200609-4-1vghqmz_html_3dad8c63de013b76.png)

will condition lead to fast gradient

**Momentum**

![](RackMultipart20200609-4-1vghqmz_html_28c66c29261f3813.png)

α (Momentum factor) normally i **ncrease** with the step goes

η (Learning Rate) **decrease** with step goes

![](RackMultipart20200609-4-1vghqmz_html_9d4401aae09a3f9c.png)

Nesterov Momentum

![](RackMultipart20200609-4-1vghqmz_html_437504c3b211ba37.png)

### **Differenct Kinds of Optimizers**

![](RackMultipart20200609-4-1vghqmz_html_650754c9af76e2fe.png)

![](RackMultipart20200609-4-1vghqmz_html_6ad5424d2129da4b.gif)

![](RackMultipart20200609-4-1vghqmz_html_45113db764c52998.gif)

**RMSprop**

![](RackMultipart20200609-4-1vghqmz_html_732d86baca3637d.png)

**Adam (more complex compared to RMSprop)**

![](RackMultipart20200609-4-1vghqmz_html_598c8bb68c3b4cee.png)

**SGD**

(may include momentum and learning rate)

more basic, need more experience

**AdaMax**

![](RackMultipart20200609-4-1vghqmz_html_7de0c682f7e012eb.png)

![](RackMultipart20200609-4-1vghqmz_html_f5271d421e21642d.png)

**Adadelta (fastest)**

![](RackMultipart20200609-4-1vghqmz_html_2774de887353b029.png)

![](RackMultipart20200609-4-1vghqmz_html_9a700563d20a42cf.png)

![](RackMultipart20200609-4-1vghqmz_html_c50119a3954193ec.png)

![](RackMultipart20200609-4-1vghqmz_html_124503f3601a19a5.png)

### **which optimizer is best?**

**RMSprop** is an extension of Adagrad that deals with its radically diminishing learning rates. It is **identical to Adadelta** , except that Adadelta uses the RMS of parameter updates in the numinator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, **RMSprop, Adadelta** , and **Adam** are very similar algorithms that do well in similar circumstances.

**Regularization**

bias- variance trade-off (in EE660) ( **for a given MSE** )

### **Dropout**

![](RackMultipart20200609-4-1vghqmz_html_4ad76f17fae7c9ae.png)

Esemble method turn to drop out

![](RackMultipart20200609-4-1vghqmz_html_890e09602d320852.png)

![](RackMultipart20200609-4-1vghqmz_html_affe632fdb720ba5.png)

![](RackMultipart20200609-4-1vghqmz_html_61dce90ac6bc48be.png)

### **Normalization**

![](RackMultipart20200609-4-1vghqmz_html_507bb6ce2744fcc9.png)

### **Dimensionality Reduction**

SVD: singular value decomposition

**PCA**

![](RackMultipart20200609-4-1vghqmz_html_da7c6b4c2d4d5542.png)

**LDA**

need labels of all data, change data based on it&#39;s label

![](RackMultipart20200609-4-1vghqmz_html_b0e618b80cb6491a.png)

![](RackMultipart20200609-4-1vghqmz_html_e099cf39bb8f671b.png)

### **Hyper parameter**

things Neural Network don&#39;t learning by themself

![](RackMultipart20200609-4-1vghqmz_html_8cc67dc43261959b.png)

![](RackMultipart20200609-4-1vghqmz_html_a69602705eeb169c.png)

### **use cross validation to set hyper-parameters**

**Hyper-parameter seatch**

![](RackMultipart20200609-4-1vghqmz_html_93f1eab87cb55241.png)

**Learning Rate Schedules**

![](RackMultipart20200609-4-1vghqmz_html_c7fed6feedeecb9e.png)

![](RackMultipart20200609-4-1vghqmz_html_5f7d00cc82230477.png)

**Universal Approximation Theorem**

basic idea: Neural Network can simulate all function

[http://neuralnetworksanddeeplearning.com/chap4.html](http://neuralnetworksanddeeplearning.com/chap4.html)

![](RackMultipart20200609-4-1vghqmz_html_143ebaad46fc9f8d.png)

## **CNN**

![](RackMultipart20200609-4-1vghqmz_html_feb29546419238d.png)

### **Introduction**

good introduction: http://cs231n.github.io/convolutional-networks/#pool

In summary:

1. A ConvNet architecture is in the simplest case a list of Layers that transform the image volume into an output volume (e.g. holding the class scores)
2. There are a few distinct types of Layers (e.g. CONV/FC/RELU/POOL are by far the most popular)
3. Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function
4. Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don&#39;t)
5. Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn&#39;t)

### **Kernel(Filter)**

Slide

Padding

**size:**

small (3\*3) deep

large (5\*5) fast reduction,shallow, more parameters

![](RackMultipart20200609-4-1vghqmz_html_24f59f1d4eaef514.png)

**Conv layer**

image: 3-D in volume

**basic:** each neural is only connect to some of the neural in next layer

**Basic Function of Conv-layer:**

also like Σ(wx)+b (if **b** exist?)

![](RackMultipart20200609-4-1vghqmz_html_648ad421f704ce85.png)

use dot production to compute kernel\* cut part

Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network.

### **Hyper-Parameters**

**depth**

number of filters we want to use, they will look into the same region

**stride**

commonly--\&gt; 1

**zero-padding**

we could:

1. control the output volume size

2. remain the edge information

**For three channel**

three filters to combine a kernel, three output add up together, and add bias

![](RackMultipart20200609-4-1vghqmz_html_e217497c65e18125.gif)

![](RackMultipart20200609-4-1vghqmz_html_a508f5e52ee8a979.gif)

![](RackMultipart20200609-4-1vghqmz_html_b2178ef8e6216e8d.gif)

**dilation**

it&#39;s possible to have filters that have spaces between each cell, called dilation.

in one dimension a filter w of size 3 would compute over input x the following: w[0]\*x[0] + w[1]\*x[1] + w[2]\*x[2]. This is dilation of 0. For dilation 1 the filter would instead compute w[0]\*x[0] + w[1]\*x[2] + w[2]\*x[4]

so the computed block is not continous

### **compute output size(O)**

by input size( **W** ), filter size( **F** ) and stride( **S** ) and amount of zero padding( **P** )

O=(W−F+2P)/S+1.

input W\*W output O\*O

### **Parameter Sharing(weight sharing)**

one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a  **depth slice**

we are going to constrain the neurons in each depth slice to use the same weights and bias

for example: for a 11\*11 kernel in RGB with 96 filters

there are 96\*11\*11\*3+96 = 34944 parameters in totally

 all 55\*55 neurons in each depth slice will now be using the same parameters.

questions? each filter has three channel or just one?

![](RackMultipart20200609-4-1vghqmz_html_567940aba3dba61b.png)

### **a new method: seperable convolution**

**Xception**

convert 3\*3 kernel to 3\*1 and 1\*3 (used in Enmedded system)

**Pooling layer**

  Its function is to progressively reduce the spatial size of the representation to **reduce the amount of parameters and computation in the network** , and hence to also control overfitting.

Max pooling | average pooling | L2-norm pooling

**Extents--\&gt;** # of points pooling together

**Stride --\&gt;** # of step to move in next pooling

![](RackMultipart20200609-4-1vghqmz_html_a999a2849fffa4f8.png)

**De-Conv(Transpose convolution) layer**

convolution: a pixels block into a point --\&gt; down sampling

de-convolution: a point to a block --\&gt; up sampling, from a low resolution to a higher one

use one point in the previous layer to generate 3\*3 blocks with the help of filter

For an example in the image below, we apply transposed convolution with a 3 x 3 kernel over a 2 x 2 input padded with a 2 x 2 border of zeros using unit strides. The up-sampled output is with size 4 x 4.

![](RackMultipart20200609-4-1vghqmz_html_c8702ee2ff8858a7.gif)

![](RackMultipart20200609-4-1vghqmz_html_f6de5a5883f45b69.png)

![](RackMultipart20200609-4-1vghqmz_html_1726de1daff24533.png)

map e.g. 4-dimensional space to 25-dimensional space

![](RackMultipart20200609-4-1vghqmz_html_64359bb6036e5d8a.gif)

Transposed 2D convolution with no padding, stride of 2 and kernel of 3

**Group Convolution**

In each filter group, the depth of each filter is only half of the that in the nominal 2D convolutions. They are of depth Din / 2. Each filter group contains Dout /2 filters. The first filter group (red) convolves with the first half of the input layer ([:, :, 0:Din/2]), while the second filter group (blue) convolves with the second half of the input layer ([:, :, Din/2:Din]). As a result, each filter group creates Dout/2 channels. Overall, two groups create 2 x Dout/2 = Dout channels. We then stack these channels in the output layer with Dout channels.

**Advantage:**

efficient training.

model is more efficient

Grouped convolution may provide a better model than a nominal 2D convolution.

![](RackMultipart20200609-4-1vghqmz_html_ae1a55d5cbd012.png)

**this is a bit similar like seperable convolutions**

If the number of filter groups is the same as the input layer channel, each filter is of depth Din / Din = 1. This is the same filter depth as in depthwise convolution.

**Seperable Convolutions**

Efficiency!

in **MobileNet:** [https://arxiv.org/pdf/1704.04861.pdf](https://arxiv.org/pdf/1704.04861.pdf)

change the 3\*3 kernel to 3\*1 and 1\*3 two kernel This would require 6 instead of 9 parameters while doing the same operations.

![](RackMultipart20200609-4-1vghqmz_html_a76793b400f386ec.png)

Although spatially separable convolutions save cost, it is rarely used in deep learning. One of the main reason is that not all kernels can be divided into two, smaller kernels. If we replace all traditional convolutions by the spatially separable convolution, we limit ourselves for searching all possible kernels during training. The training results may be sub-optimal.

**Depthwise Separable Convolutions**

much commonly used in deep learning: **depthwise convolution** and **1\*1 convolution**

First, we apply **depthwise convolution** to the input layer. Instead of using a single filter of size 3 x 3 x 3 in 2D convolution, we used **3 kernels** , **separately.** Each filter has size **3 x 3 x 1**. Each kernel convolves with 1 channel of the input layer (1 channel only, not all channels!). We then stack these maps together to create a 5 x 5 x 3 image. After this, we have the output with size 5 x 5 x 3.

As the second step of depthwise separable convolution, to extend the depth, we apply the 1x1 convolution with kernel size 1x1x3. Convolving the 5 x 5 x 3 input image with each 1 x 1 x 3 kernel provides a map of size 5 x 5 x 1.

Thus, after applying 128 1x1 convolutions, we can have a layer with size 5 x 5 x 128.

![](RackMultipart20200609-4-1vghqmz_html_ae2d5c768811c729.png)

![](RackMultipart20200609-4-1vghqmz_html_4b89cc5a2755a4df.png)

**drwaback:**

reduces the number of parameters in the convolution. As such, for a small model, the model capacity may be decreased significantly if the 2D convolutions are replaced by depthwise separable convolutions.

**Full Connect layer**

**# of Parameters computation**

**Visualization**

 layers that are deeper in the network visualize more training data specific features, while the earlier layers tend to visualize general patterns like edges, texture, background

**Visualize different layer**

**Visualize filter**

**Heatmap**

**Different Nets**

**LeNet(1998)**

classifies digits in 32x32 pixel greyscale inputimages

![](RackMultipart20200609-4-1vghqmz_html_97045388cc290811.png)

**AlexNet(2012)**

  deeper, with more filters per layer, and with stacked convolutional layers. It consisted 11x11, 5x5,3x3, convolutions, max pooling, dropout, data augmentation, ReLU activations, SGD with momentum.

train on two GPUs

![](RackMultipart20200609-4-1vghqmz_html_5b8964470d08b957.png)

**VGG**

[https://arxiv.org/pdf/1409.1556.pdf](https://arxiv.org/pdf/1409.1556.pdf)

Baseline feature extractor

16 (weighted) layers CNN 138M parameters

![](RackMultipart20200609-4-1vghqmz_html_a126afd34ff33e86.png)

**Resnet**

gated recurrent units

take a standard feed-forward ConvNet and add **skip connections** that bypass (or shortcut) a few convolution layers at a time. Each bypass gives rise to a residual block in which the convolution layers predict a residual that is added to the block&#39;s input tensor.

aim to avoid vanish gradient

![](RackMultipart20200609-4-1vghqmz_html_3e7de549ae5e4c13.png)

**GoogleNet (Inception)**

[https://arxiv.org/pdf/1512.00567.pdf](https://arxiv.org/pdf/1512.00567.pdf)

[https://arxiv.org/pdf/1409.4842.pdf](https://arxiv.org/pdf/1409.4842.pdf)

no Pooling layer

**Introduction**

The network used a CNN inspired by LeNet but implemented a novel element which is dubbed an inception module. It used batch normalization, image distortions and RMSprop. This module is based on several very small convolutions in order to drastically reduce the number of parameters. Their architecture consisted of a 22 layer deep CNN but reduced the number of parameters from 60 million (AlexNet) to 4 million.

**Speciality**

1. use several small filter to stand large filter --\&gt; n\*1 + 1\*n to replace n\*n

2. for a single input layer, applied many different filters(some are pooling, some are 1\*1 with 1\*n with n\*1) and concatedate result (add together)

---\&gt; **to avoid representational bottlenecks and avoid stop locally**

![](RackMultipart20200609-4-1vghqmz_html_5ed94d0df129e768.png)

![](RackMultipart20200609-4-1vghqmz_html_ebb904f0bfe347c1.png)

**MobileNet**

[https://arxiv.org/pdf/1704.04861.pdf](https://arxiv.org/pdf/1704.04861.pdf)

Deepwise Seperable Convolution + Pointwise Convolution (1\*1)

![](RackMultipart20200609-4-1vghqmz_html_3dd88c0ba55bdaaa.png)

![](RackMultipart20200609-4-1vghqmz_html_1d21cff1e02fdb64.png)

![](RackMultipart20200609-4-1vghqmz_html_2033bf0a7a90fd37.png)

MobileNet的网络结构如表1所示。首先是一个3x3的标准卷积，然后后面就是堆积depthwise separable convolution，并且可以看到其中的部分depthwise convolution会通过strides=2进行down sampling。然后采用average pooling将feature变成1x1，根据预测类别大小加上全连接层，最后是一个softmax层。如果单独计算depthwise

convolution和pointwise convolution，整个网络有28层（这里Avg Pool和Softmax不计算在内）。我们还可以分析整个网络的参数和计算量分布，如表2所示。可以看到整个计算量基本集中在1x1卷积上，如果你熟悉卷积底层实现的话，你应该知道卷积一般通过一种im2col方式实现，其需要内存重组，但是当卷积核为1x1时，其实就不需要这种操作了，底层可以有更快的实现。对于参数也主要集中在1x1卷积，除此之外还有就是全连接层占了一部分参数。

![](RackMultipart20200609-4-1vghqmz_html_2033bf0a7a90fd37.png)

![](RackMultipart20200609-4-1vghqmz_html_eb2256fb47a49c09.png)

**Comparation**

![](RackMultipart20200609-4-1vghqmz_html_73a6a5077dffbe16.png)

**CNN in NLP**

![](RackMultipart20200609-4-1vghqmz_html_649ca37756cc28eb.png)

**Coding**

**Keras**

a basic model

from keras.models import Sequential from keras.layers import Dense, Conv2D, Flatten#create model model = Sequential()#add model layers model.add(Conv2D(64, kernel\_size=3, activation=&#39;relu&#39;, input\_shape=(28,28,1))) model.add(Conv2D(32, kernel\_size=3, activation=&#39;relu&#39;)) model.add(Flatten()) model.add(Dense(10, activation=&#39;softmax&#39;)) # Compiling the model takes three parameters: optimizer, loss and metrics. model.compile(optimizer=&#39;adam&#39;, loss=&#39;categorical\_crossentropy&#39;, metrics=[&#39;accuracy&#39;]) #train the model model.fit(X\_train, y\_train, validation\_data=(X\_test, y\_test), epochs=3) #predict first 4 images in the test set model.predict(X\_test[:4]) model = Sequential() model.add(Conv2D(32, (3, 3), padding=&#39;same&#39;, input\_shape=x\_train.shape[1:])) model.add(Activation(&#39;relu&#39;)) model.add(Conv2D(32, (3, 3))) model.add(Activation(&#39;relu&#39;)) model.add(MaxPooling2D(pool\_size=(2, 2))) model.add(Dropout(0.25)) model.add(Conv2D(64, (3, 3), padding=&#39;same&#39;)) model.add(Activation(&#39;relu&#39;)) model.add(Conv2D(64, (3, 3))) model.add(Activation(&#39;relu&#39;)) model.add(MaxPooling2D(pool\_size=(2, 2))) model.add(Dropout(0.25)) model.add(Flatten()) model.add(Dense(512)) model.add(Activation(&#39;relu&#39;)) model.add(Dropout(0.5)) model.add(Dense(num\_classes)) model.add(Activation(&#39;softmax&#39;))

 &#39; **Flatten&#39; layer.** Flatten serves as a connection between the convolution and dense layers.

# **RNN**

**Introduction**

_x\_1, x\_2, x\_3, …, x\_t_ represent the input words from the text, _y\_1, y\_2, y\_3, …, y\_t_ represent the predicted next words and _h\_0, h\_1, h\_2, h\_3, …, h\_t_ hold the information for the previous input words.

![](RackMultipart20200609-4-1vghqmz_html_b1c5a722421e871a.png)

![](RackMultipart20200609-4-1vghqmz_html_de0f04d469445c85.png)

![](RackMultipart20200609-4-1vghqmz_html_4701d6f28649be72.png)

basic structure

![](RackMultipart20200609-4-1vghqmz_html_7685f3043467f378.png)

**STATE**

state machine: network is stated:

![](RackMultipart20200609-4-1vghqmz_html_41e6214aadc53add.png)

(Vanilla RNN)

Keras code

keras.layers.SimpleRNN(units, activation=&#39;tanh&#39;, use\_bias=True, kernel\_initializer=&#39;glorot\_uniform&#39;, recurrent\_initializer=&#39;orthogonal&#39;, bias\_initializer=&#39;zeros&#39;, kernel\_regularizer=None, recurrent\_regularizer=None, bias\_regularizer=None, activity\_regularizer=None, kernel\_constraint=None, recurrent\_constraint=None, bias\_constraint=None, dropout=0.0, recurrent\_dropout=0.0, return\_sequences=False, return\_state=False, go\_backwards=False, stateful=False, unroll=False)

**Diagram of Neural Net**

![](RackMultipart20200609-4-1vghqmz_html_ee1faaf3b424bcf8.png)

**Number of nodes and parameters in each layer (blue block on upper diagram)**

![](RackMultipart20200609-4-1vghqmz_html_d483efe064cf699a.png)

Node on layer: n

Node on previous layer: m

input: m

output: n

**parameters** : **(m+n)\*n** (weight) **+n** (bias)

When training, a training window length T is selected =\&gt;

a sequence of input of length T

a sequence of label of length T

[http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

**Vanish Gradient Problem**

![](RackMultipart20200609-4-1vghqmz_html_79c2415e24329448.png)

**GATE**

To solve this problem we need to add GATE which is attenuating and/or filtering in the state update equation (my understanding: amplify the influence of previous state)

![](RackMultipart20200609-4-1vghqmz_html_a889ea58591140fb.png)

All gate are trainable parameters and are learned using a single layer feedforward network (my understanding: GATE make simple m\*n parameters network more complex, though if we block the process, it is still a m input and n output problem, the inner parametes(weight) are no longer n(m+n+1), it because more complex, but still based on **V(n\*n), W(m\*n), b (n)**

![](RackMultipart20200609-4-1vghqmz_html_c48cb2de28927154.gif)

tanh make the output in between -1 -\&gt; 1

**LSTM**

![](RackMultipart20200609-4-1vghqmz_html_6fdeda312ac89911.png)

![](RackMultipart20200609-4-1vghqmz_html_78205c093de637c0.png)

![](RackMultipart20200609-4-1vghqmz_html_e67933a5de123164.png)

These operations are used to allow the LSTM to keep or forget information.

**Core Concept**

The core concept of LSTM&#39;s are the cell state, and it&#39;s various gates. The cell state act as a transport highway that transfers relative information all the way down the sequence chain. You can think of it as the &quot;memory&quot; of the network. The cell state, in theory, can carry relevant information throughout the processing of the sequence. So even information from the earlier time steps can make it&#39;s way to later time steps, reducing the effects of short-term memory. As the cell state goes on its journey, information get&#39;s added or removed to the cell state via gates. The gates are different neural networks that decide which information is allowed on the cell state. The gates can learn what information is relevant to keep or forget during training.

there are two state value(cell and hiden in each state)

**Forget Gate**

This gate decides what information should be thrown away or kept. Information from the previous hidden state and information from the current input is passed through the sigmoid function. Values come out between 0 and 1. The closer to 0 means to forget, and the closer to 1 means to keep.

![](RackMultipart20200609-4-1vghqmz_html_8688019d24f82e96.gif)

**Input Gate**

First, we pass the previous hidden state and current input into a sigmoid function. That decides which values will be updated by transforming the values to be between 0 and 1. 0 means not important, and 1 means important. You also pass the hidden state and current input into the tanh function to squish values between -1 and 1 to help regulate the network. Then you multiply the tanh output with the sigmoid output. The sigmoid output will decide which information is important to keep from the tanh output.

![](RackMultipart20200609-4-1vghqmz_html_b8df976ec465ee61.gif)

**Cell Gate**

First, the cell state gets pointwise multiplied by the forget vector. This has a possibility of dropping values in the cell state if it gets multiplied by values near 0. Then we take the output from the input gate and do a pointwise addition which updates the cell state to new values that the neural network finds relevant. That gives us our new cell state.

![](RackMultipart20200609-4-1vghqmz_html_729638f3288789f5.gif)

**Output Gate**

The output gate decides what the next hidden state should be. Remember that the hidden state contains information on previous inputs. The hidden state is also used for predictions. First, we pass the previous hidden state and the current input into a sigmoid function. Then we pass the newly modified cell state to the tanh function. We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. The output is the hidden state. The new cell state and the new hidden is then carried over to the next time step.

![](RackMultipart20200609-4-1vghqmz_html_806fd902524628.gif)

**OTHER VERSION(Keras)**

![](RackMultipart20200609-4-1vghqmz_html_30c2ecd3137c3bf9.png)

![](RackMultipart20200609-4-1vghqmz_html_7b72b43248ff4d5f.png)

**My understanding:** In LSTM, each gate has a group of parameters, so four gates, four group of patameters, each group will generate an output, and they combine together to generate the final output and hiden state so the Keras version will use more parameters (4\*n\*(m+n+1))

**other version of intorduction ()**

![](RackMultipart20200609-4-1vghqmz_html_5e6749d9445f804.png)

![](RackMultipart20200609-4-1vghqmz_html_b39d32350a5d48c8.png)

![](RackMultipart20200609-4-1vghqmz_html_df4c70e64d3b9000.png)

![](RackMultipart20200609-4-1vghqmz_html_ae09f6d76df0bcb3.png)

![](RackMultipart20200609-4-1vghqmz_html_ab4f6352738c94a4.png)

LSTM code:

usful source: [Illustrated Guide to LSTM&#39;s and GRU&#39;s: A step by step explanation]

https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21

[https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359#:~:text=This%20allows%20information%20from%20previous,with%20in%20the%20LSTM%20cell.&amp;text=These%20gates%20determine%20which%20information,of%20%5B0%2C1%5D.](https://medium.com/@kangeugine/long-short-term-memory-lstm-concept-cb3283934359#:~:text=This%20allows%20information%20from%20previous,with%20in%20the%20LSTM%20cell.&amp;text=These%20gates%20determine%20which%20information,of%20%5B0%2C1%5D.)

**GRU**

![](RackMultipart20200609-4-1vghqmz_html_6ea62f4da7bff9c9.png)

![](RackMultipart20200609-4-1vghqmz_html_bb73d8d504e5ad26.png)

**Update gate**

(samilar to foget+input gate)

![](RackMultipart20200609-4-1vghqmz_html_fb8cbddd00aedb40.png)

**Reset Gate**

(samiar to )

![](RackMultipart20200609-4-1vghqmz_html_fdf90057517a3505.png)

![](RackMultipart20200609-4-1vghqmz_html_2625320ab9b5f972.png)

**Back-Propagation through Time (BPTT)**

**Word Embedding**

**BERT**

BERT (Bidirectional Encoder Representations from Transformers), released in late 2018, is a method to **pretrain language representations** that was used to create models that NLP practicioners can then download and use for free. You can either use these models to extract high quality language features from your text data, or you can fine-tune these models on a specific task (classification, entity recognition, question answering, etc.) with your own data to produce state of the art predictions.

GRT-3

# **TensorFlow and Keras**

**sequential model**

# **Pytorch**