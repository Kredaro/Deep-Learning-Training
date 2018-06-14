# Logistic Regression

What we need a algorithm which will help us classify the the input X into one of the 2 classes. Rather we need a probabilistic output about what are the chances that this is equal to one.

Given parameters w and b we cannot use Wt \* X + b&nbsp;, because this is linear and the output might be negative or a large number. We use the sigmoid function to change the inputs.

## Sigmoid Activation function:&nbsp;

The sigmoid function outputs a close to 0 for large negative values of z, and a value close to 1 for large positive values of z, but it’s always bound between 0 and 1.&nbsp;

The objective is to learn the parameters W and b so that the Y^ becomes a good estimate of Y.&nbsp;


## Logistic Regression cost&nbsp;function

To train W and b you need a cost function, in any training of parameters in supervised learning you need a cost function.

The idea is to learn the parameters W and b so that the prediction Y^ atleast on the m training examples is close to the ground truth value.

Once we learn the parameters we can use the following equation to make the prediction for any training example&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*ML2iNBabY6hsksomRFcOnA.png)

The loss or the error function tells us how good the learning algorithm is.

## Why doesn’t the mean square error doesn't work&nbsp;?

Let’s see why this works,&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*87SGdLlQEVkEq_gH9KI6Dw.png)

We have find the values W and b such that the loss is minimized.&nbsp;

Gradient descent takes advantage of the fact that loss function curve w.r.t the parameters W and b looks like this, we need to find the values W and b which corresponds to the minimum of the cost function J.

Here is what we’ll do to find the optimal values for w and b,&nbsp;

Initialize **W** and **b** for some initial values, and then the Gradient decent algorithm modifies the values of parameters such that the loss function starts moving in the direction of the steepest curve and eventually you’ll arrive at the parameters&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*laN3aseisIU3T9QTIlob4Q.gif)

**The Gradient descent update equation**

 ![](https://cdn-images-1.medium.com/max/1600/1*9h-O9EcIFznTebLdGk_mrw.png)

The definition of derivative is the slope of the function at the given point.If the derivative is positive then we subtract the value **_alpha \* dw_** from w and a step to the left will be taken which brings us one step closer to the minimum.&nbsp;

> The derivative suggests the direction of stepping down the Loss Vs Parameters curve&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*r9zyZAg4r4Biu8ZHC2yAag.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*0wd5zlfmEEdosgw-cw8ySw.png)

**Gradient descent on m training dataset**

 ![](https://cdn-images-1.medium.com/max/1600/1*qmpMZVNp8lCeeghEvuVGpQ.png)

* * *

**Vectorization**

Its the process removing explicit for loops. Its important that the code runs fast when you train on big dataset.&nbsp;

Dot product vectorized vs non-vectorized&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*XQijzyzkVCEtHfUKsIE5sg.png)
*Dot product vectorized vs non-vectorized*

Note: **Why is vectorized version faster?**

Numpy takes much better advantage of CPU instructions like Simd for running the code in a parallel manner.

 ![](https://cdn-images-1.medium.com/max/1600/1*ulFnFCXyPACAEmV_YWWEkg.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*CSH9zcr4wrA14eu2fg3u0A.png)
*vectorization for exponential operaation&nbsp;*

> Everytime you implement a explicit for loop check if there is a built in numpy function which implements the&nbsp;same

 ![](https://cdn-images-1.medium.com/max/1600/1*ehtMGTGkWKEu7v1cccFK7A.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*2in60DubcoYR_j10iDqhNg.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*zelqf-eT6o8G-FP4VrtDoQ.png)

 ![](https://cdn-images-1.medium.com/max/800/1*IXATGxjdcLPqfjCIln4JGA.png)
*VECTORIZED ANDNON VECTORIZED GRADIENT&nbsp;DESCENT*

But you need a for-loop for number of iterations for which you want to run Gradient Descent.

#### Broadcasting example&nbsp;

* * *

#### Points to watch out in python&nbsp;numpy

 ![](https://cdn-images-1.medium.com/max/800/1*30DIdzV7tTxdKSb7n2iseA.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*LWXKwJRHrxNlIO175ZD-nA.png)
*[https://colab.research.google.com/drive/1q3O8fAkIKKLHLPcUz0Io-UO-x2J4oNje#scrollTo=Lez1BnxA-Vbj&line=2&uniqifier=1](https://medium.com/r/?url=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1q3O8fAkIKKLHLPcUz0Io-UO-x2J4oNje%23scrollTo%3DLez1BnxA-Vbj%26line%3D2%26uniqifier%3D1)*

> Avoid using rank 1&nbsp;array&nbsp;

- `a = random.randn(5,1)`
- `a = random.randn(1,5)`
- `assert(a.shape == (5,1))`
- `a = a.reshape((5,1))`

* * *
