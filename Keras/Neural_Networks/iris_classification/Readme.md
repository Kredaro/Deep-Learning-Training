

 

Back in 2015. Google released TensorFlow, the library that will change the field of Neural Networks and eventually make it mainstream. Not only that TensorFlow became popular for developing Neural Networks, it also enabled higher-level APIs to run on top of it. One of those APIs is Keras. Keras is&nbsp;written in Python and it is not supporting only&nbsp;[TensorFlow](https://github.com/tensorflow/tensorflow). It is capable of running on top of&nbsp;[CNTK](https://github.com/Microsoft/cntk)&nbsp;and&nbsp;[Theano](https://github.com/Theano/Theano).  There are many benefits of using Keras, and one of the main ones is certainly&nbsp;user-friendliness. API is easily understandable and pretty straight-forward. Another benefit is modularity. A Neural Network (model) can be observed either as a sequence or a graph of standalone, loosely coupled and fully-configurable modules. Finally, Keras is easily extendable.

![](https://i1.wp.com/i.imgur.com/oLT0DTk.png?w=720&ssl=1)

## Installation and Setup

As mentioned before, Keras is running on top of TensorFlow. So, in order for this library to work, you first need to [**install TensorFlow**](http://rubikscode.net/2018/02/05/introduction-to-tensorflow-with-python-example/). Another thing I need to mention is that for the purposes of this article, I am using Windows 10 and&nbsp;[**Python 3.6**](https://www.python.org/). Also, I am using Spyder IDE for the development so examples in this article may variate for other operating&nbsp;systems and platforms. Since Keras is a Python library installation of it is pretty standard. You can use “native pip” and install it using this command:

```
pip install keras
```

Or if you are using Anaconda you can install Keras by issuing the command:

```
conda install -c anaconda keras
```

Alternatively, the installation process can be done by using Github source. Firstly, you would have to clone the code from the repository:

```
git clone https://github.com/keras-team/keras.git
```

After that, you need to position the terminal in that folder and run the install command:

```
python setup.py install
```

## Sequential Model and Keras Layers

One of the major points for using Keras is that it is one user-friendly API. It has two types of models:

- &nbsp;Sequential model
- Model class used with functional API

Sequential model is probably the most used feature of Keras. Essentially it represents the array of Keras Layers. It is convenient&nbsp;for the fast building of different types of Neural Networks, just by adding layers to it. There are many types of Keras Layers, too. The most basic one and the one we are going to use in this article is called&nbsp;_Dense.&nbsp;_It has many options for setting the inputs, activation functions and so on. Apart from&nbsp;_Dense,&nbsp;_Keras API provides different types of layers for Convolutional Neural Networks, Recurrent&nbsp;Neural Networks, etc. This is out of the scope of this post, but we will cover it in fruther posts. So, let’s see how one can build a Neural Network using _Sequential_ and _Dense.&nbsp;_

 

In this sample, we first imported the&nbsp;_Sequential&nbsp;_and&nbsp;_Dense&nbsp;_from Keras. Than we instantiated&nbsp;one object of the&nbsp;_Sequential_ class. After that, we added one layer to the Neural Network using function _add&nbsp;_and&nbsp;_Dense_ class. The first parameter in the _Dense_ constructor is used to define a number of neurons in that layer.&nbsp;What is specific about this layer is that we used&nbsp;_input\_dim_ parameter. By doing so, we added additional input layer to our network with the number of neurons defined in&nbsp;_input\_dim_ parameter. Basically, by this one call, we added two layers. First one is input layer with two neurons, and the second one is the hidden layer with three neurons.

Another important parameter, as you may notice, is&nbsp;_activation_ parameter. Using this parameter we define [**activation function**](http://rubikscode.net/2017/11/20/common-neural-network-activation-functions/) for all neurons in a specific layer. Here we used _‘relu’&nbsp;_value, which indicates that neurons in this layer will use [**Rectifier activation function**](http://rubikscode.net/2017/11/20/common-neural-network-activation-functions/). Finally, we call&nbsp;_add_ method of the _Sequential_ object once again and add another layer. Because we are not using&nbsp;_input\_dim_ parameter one layer will be added, and since it is the last layer we are adding to our Neural Network it will also be the output layer of the network.

## Iris Data Set Classification Problem

We will use Iris Data Set Classification Problem for this demonstration. Iris Data Set is famous dataset in the world of pattern recognition and it is considered to be “Hello World” example for&nbsp;machine learning classification problems.&nbsp;It was first introduced by&nbsp;[Ronald Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher),&nbsp;British statistician and botanist, back in 1936. In his paper&nbsp;_T__he use of multiple measurements in taxonomic problems,&nbsp;_he used data collected for three different classes of Iris plant: _Iris setosa_,&nbsp;_Iris virginica,&nbsp;_and&nbsp;_Iris versicolor_.

![](https://i0.wp.com/i.imgur.com/PaAaKqr.png?w=720&ssl=1)

This dataset contains 50 instances for each class. What is interesting about it is that first class is linearly separable from the other two, but the latter two are not linearly separable from each other.&nbsp;Each instance has five attributes:

- Sepal length in cm
- Sepal width in cm
- Petal length in cm
- Petal width in cm
- Class (_Iris setosa_,&nbsp;_Iris virginica,&nbsp;__Iris versicolor_)

In next chapter we will build Neural Network using Keras, that will be able to predict the class of the Iris flower based on the provided attributes.

## Code

Keras programs have similar to the workflow of TensorFlow programs. We are going to follow this procedure:

- Import the dataset
- Prepare data for processing
- Create the model
- Training
- Evaluate accuracy of the model
- Predict results using the model

Training and evaluating processes are crucial for any Artificial Neural Network. These processes are usually done using two datasets, one for training and other for testing the accuracy of the trained network. In the real world, we will often get just one dataset and then we will split them into two separate datasets. For the training set, we usually use 80% of the data and another 20% we use to evaluate our model. This time this is already done for us. You can download training set and test set with code that&nbsp;accompanies this article from&nbsp;**[here](https://github.com/NMZivkovic/SimpleNeuralNetworkKeras)**.

However before we go any further, we need to import some libraries. Here is the list of the libraries that we need to import.

 

As you can see we are importing Keras dependencies, _NumPy&nbsp;_and P_andas.&nbsp;NumPy_&nbsp;is the fundamental package for scientific computing and _Pandas&nbsp;_provides easy to use data structures and data analysis tools.

After we imported libraries, we can proceed with importing the data and preparing it for the processing. We are going to use&nbsp;_Pandas_ for importing data:

 

Firstly, we used&nbsp;_read\_csv&nbsp;_function to import the dataset into local variables, and then we separated inputs&nbsp;_(train\_x, test\_x)_&nbsp;and expected outputs&nbsp;_(train\_y, test\_y)&nbsp;_creating four separate matrixes. Here is how they look like:

![](https://i2.wp.com/i.imgur.com/HpD13HC.png?w=720&ssl=1)

However, our data is not prepared for processing yet. If we take a look at our expected output values, we can notice that we have three values: 0, 1 and 2. Value 0 is used to represent Iris&nbsp;setosa, value 1 to represent Iris&nbsp;versicolor and value 2 to represent&nbsp;virginica. The good news about these values is that we didn’t get string values in the dataset. If you end up in that situation, you would need to use some kind of encoder so you can format data to something similar as we have in our current dataset. For this purpose, one can use **_[LabelEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)&nbsp;_**of sklearn library. Bad news about these values in the dataset is that they are not applicable to&nbsp;_Sequential_ model. What we want to do is&nbsp;reshape the expected output from a vector that contains values for each class value to a matrix with a boolean for each class value.&nbsp;This is called&nbsp;**[one-hot encoding](https://en.wikipedia.org/wiki/One-hot)**. In order to achieve this, we will use&nbsp;_np\_utils_ from the Keras library:

 

If you still have doubt what one-hot encoding is doing, observe image below. There are displayed&nbsp;_train\_y_ variable and&nbsp;_encoding\_train\_y_ variable. Notice that first value in&nbsp;_train\_y_ is 2 and see the corresponding value for that row in&nbsp;_encoding\_train\_y._

![](https://i1.wp.com/i.imgur.com/RfnDxA2.png?w=720&ssl=1)

Once we imported and prepared the data we can create our model. We already know we need to do this by using&nbsp;_Sequence&nbsp;_and&nbsp;_Dense_ class. So, let’s do it:

 

This time we are creating:

- one input layer with four nodes, because we are having four attributes in our input values
- two hidden layers with ten neurons each
- one output layer with three neurons, because we are having three output classes

In hidden layers, neurons use [**Rectifier activation function**](http://rubikscode.net/2017/11/20/common-neural-network-activation-functions/), while in output layer neurons use Softmax activation function (ensuring that output values are in the range of 0 and 1). After that, we compile our model, where we define our [**cost function**](http://rubikscode.net/2018/01/15/how-artificial-neural-networks-learn/)&nbsp;and optimizer. In this instance, we will use&nbsp;Adam [**gradient descent optimization algorithm**](http://rubikscode.net/2018/01/15/how-artificial-neural-networks-learn/) with a logarithmic cost function (called _categorical\_crossentropy&nbsp;_in Keras).

Finally, we can train our network:

And evaluate it:


If we run this code, we will get these results:

![](https://i2.wp.com/i.imgur.com/6V9MsMp.png?w=720&ssl=1)

Accuracy – 0.93. That is pretty good. After this, we can call our classifier using single data and get predictions for it.

## Conclusion

Keras is one awesome API which makes building Artificial Neural Networks easier. It is quite easy getting used to it.&nbsp; In this article, we just scratched the surface of this API and in next posts, we will explore how we can implement different types of Neural Networks using this API.


