# Activation functions

Here are some of the questions we need to answer,&nbsp;

- What activation functions to use?
- Why to use activation functions

Without the activation function the output would be linear combination of the weights of inputs, and the continuous output value wouldn’t work of the linear network would in any case help if its a classification problem.

 ![](https://cdn-images-1.medium.com/max/1200/1*-X8MGFQIWi_mtadHGUjIXw.png)

 ![](https://cdn-images-1.medium.com/max/1200/1*vKevkGT_IVJTDT7OJ8X1IA.png)
*Sigmoid activation being used on z to arrive at the final&nbsp;output&nbsp;*

 ![](https://cdn-images-1.medium.com/max/2000/1*NsGPP0vTeF6CheU3vzeslw.png)

 ![](https://cdn-images-1.medium.com/max/2000/1*6A3A_rt4YmumHusvTvVTxw.png)

* * *

#### The Tanh hyperbolic activation function

The tanh activation is known to work better than the sigmoid function since the value of tanh ranges from -1 to 1, this helps the mean of activation to be centered around 0 which inturn helps the training.

But sigmoid is a preferred choice for the output layer since we want the output to be a value between 0 and 1.

 ![](https://cdn-images-1.medium.com/max/2000/1*ZA9OZIT6o5LTtD7HP18xvA.png)

 ![](https://cdn-images-1.medium.com/max/2000/1*sCEV7ZRZL4ekVyqCs4KyKA.png)
*Tanh activation function&nbsp;*

> Notice that derivatives of tanh and sigmoid are close to for large and small values. This will slow down the gradient descent. The RELU activation function overcomes this disadvantage. The following are the equations and graph for&nbsp;RELU.

 ![](https://cdn-images-1.medium.com/max/1600/1*TOlGSonMzi8gqFsKlySc6Q.png)

 ![](https://cdn-images-1.medium.com/max/2000/1*-5-9q7LOvVykztaPe6vBiA.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*xfuB1IPXnwLVECRfm8zsEg.png)

* * *

### Leaky Relu

The gradient of RELU is 0 for negative Z values, this affects the training at times, Leaky RELU returns a small constant value for negative values of Z, thus it eliminates the drawback on RELU.

 ![](https://cdn-images-1.medium.com/max/2000/1*AH9cvzZPtJmC8P0sn-6CKA.png)

 ![](https://cdn-images-1.medium.com/max/2000/1*e4Ikoi2UorMZfcICoApaeg.png)

* * *

### **Summary** &nbsp;

 ![](https://cdn-images-1.medium.com/max/1200/1*i6jXH9kIZ_BYgdAU57bvIA.png)

- Never use sigmoid apart from the final layer.
- Tanh is a better alternative to sigmoid.&nbsp;
- But RELU usually performs better than tanh.
- For z values less than 0 the RELU outputs 0, this at times slows down the training, the Leaky RELU helps solve this problem.

A good exercise would be to try them all on a problem and then evaluate them on a validation set.&nbsp;
* * *

## Activation function for Regression problem
Only the output layer can be set to linear activation function ( no activation) in case of Regression problem.

* * *

#### Resources&nbsp;

- [Great resource for learning about activation functions](https://medium.com/r/?url=https%3A%2F%2Fisaacchanghau.github.io%2Fpost%2Factivation_functions%2F)
- [One more Resource](https://medium.com/r/?url=https%3A%2F%2Ftowardsdatascience.com%2Factivation-functions-neural-networks-1cbd9f8d91d6)
