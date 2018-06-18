## Building Deep Neural Networks 

![](https://cdn-images-1.medium.com/max/2000/1*D-Z9jOQVWCRiWOyGcpV7bw.png)

For a given problem its hard to tell how deep the network should be, its good to start simple by using Logistic Regression and then try a 1 layer neural network and pump up the complexity till the accuracy on validation set is appreciable.&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*VltKTdZ-MzobAbZ53qVaHw.png)

Here is the general formula for forward propagation in Neural Networks.

 ![](https://cdn-images-1.medium.com/max/1600/1*jYF70Ka7IOJA1mDHMHXaAA.png)

Let’s consider a 4 layer Neural network and write the forward propagation equation for each layer.

 ![](https://cdn-images-1.medium.com/max/1600/1*o0T9gBaRb8hOeDhLSe6gsA.png)

* * *

#### **Getting the right dimensions&nbsp;**

One of the challenges while implementing a Neural Network is to get the dimensions of various parameters right.&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*2w311pN0UnxaB137fAhX-g.png)

The size of vectors for m training examples at once&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*vFQgN0mT8Yg24SYzY_3k6w.png)
*The matrix sizes for vectorized implementation of Gradient&nbsp;descent&nbsp;*

* * *

#### One iteration of Gradient&nbsp;Descent&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*a5CsjU3r_JVDrTZqDV71Uw.png)
