## Intro to tensorflow, linear regression: Build linear regression model using tensorflow to predict house&nbsp;price

Howdy! Welcome to the 3rd blog of the series for Data science India conference, we began with the following couple of articles on building neural networks from scratch,

- [Intuitive and practical guide for building neural networks from scratch-Part1](https://medium.com/ai-india/intuitive-and-practical-guide-for-building-neural-networks-from-scratch-d60126645d58)
- [Intuitive and practical guide for building neural networks from scratch-Part2](https://medium.com/ai-india/intuitive-guide-to-neural-networks-part2-4312633bc96)

As the preparations for the conference gains momentum let’s move towards other exiting facets of Machine learning/Deep learning.

This blog is about introducing the readers about Tensorflow, which is a open source Machine learning / Deep learning framework by Google. Along with learning its fundamentals, we would also be taking a detailed look at using tensorflow to build machine learning models. In this process we’ll be going through the concept of **_linear regression ML algorithm_** and use tensorflow to build the model and apply it to predict house price. This would be a great start before we begin with using tensorflow for building more complex models.

Here we go! Let’s begin fundamentals of Tensorflow.

* * *

#### Tensorflow fundamentals

First, we’re going to take a look at the **_tensor_** object type. Then we’ll have a graphical understanding of TensorFlow to define computations. Finally, we’ll run the graphs with sessions, showing how to substitute intermediate values.

#### **Tensors**

In TensorFlow, data isn’t stored as integers, floats, or strings. These values are encapsulated in an object called a tensor, a fancy term for multidimensional arrays. If you pass a Python list to TensorFlow, converts it into an tensor of appropriate type.

You’ll hold constant values in `tf.constant`.

*Tensors and constants in tensorflow*

You can perform computations on the constants, but these tensors won’t be evaluated until a session is created and the tensor are run inside the session.

**Tensorflow Session**

Everything so far has just specified the TensorFlow graph. We haven’t yet computed anything. To do this, we need to start a session in which the computations will take place. The following code creates a new session:

`sess = tf.Session()`

TensorFlow’s api is built around the idea of a computational graph, a way of visualizing a mathematical process.

**Running a session and evaluating a tensor**

The code creates a session instance, **_sess_** , using **_tf.Session_**. The **sess.run()** function then evaluates the tensor and returns the results. Once you have a session open, `sess.run(NN)` will evaluate the given expression and return the result of the computation.

After you run the above, you will see the **Hello World** printed out:

```
import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

* * *

#### Computation graph

The computations are structured in the form of a graph in tensorflow, Let’s dive a bit more into computational graph and understand how computation graphs are formed,

Consider a simple computation. Let **_a, b, c_** be the variables used,

```
J = 3 * ( a + bc )
```

There are 3 distinctive computational steps to arrive at the final value J, let’s list them out,

```
u = b * c

v = (a + u)

J = 3 * v
```

These are 3 computational steps which can be represented as a graph,

 ![](https://cdn-images-1.medium.com/max/1600/1*0pNy655CFzFoeYCO_oTMUw.png)
*The graph representing the computation J = 3(a +&nbsp;bc)*

Here are the advantages of organizing the computations as a graph,

 ![](https://cdn-images-1.medium.com/max/1200/1*SmfhKWHXHVEMg8KqNaj-uw.gif)

- **Parallelism.** By using explicit edges to represent dependencies between operations, it is easy for the system to identify operations that can execute in parallel.
- **Distributed execution.** By using explicit edges to represent the values that flow between operations, it is possible for TensorFlow to partition your program across multiple devices (CPUs, GPUs, and TPUs) attached to different machines. TensorFlow inserts the necessary communication and coordination between devices.
- **Compilation.** TensorFlow’s [XLA compiler](https://www.tensorflow.org/performance/xla/index) can use the information in your dataflow graph to generate faster code, for example, by fusing together adjacent operations.

* * *

We used `tf.constant` to represent constant values, but how we need a way to set values dynamically before the tensors are evaluated by `session.run`&nbsp;. This is where `tf.placeholder()` and `feed_dict` comes to the rescue. In this section we’ll understand the mechanisms of feeding data into TensorFlow.

### tf.placeholder()

`tf.placeholder()` returns a tensor that gets its value from data passed to the `tf.session.run()` function, allowing you to set the input right before the session runs.

Let’s revisit our Hello world example, but this time let’s set the value right before the session evaluates the tensor. This helps us in passing different input data,

*Tensorflow session*

Use the `feed_dict` parameter in `tf.session.run()` to set the placeholder tensor. The above example shows the tensor `x` being set to the string `"Hello, world"`. It's also possible to set more than one tensor using `feed_dict` as shown below.

Here is one more example,

*Multiple inputs through feed\_dict*

* * *

There is a long list of operations that can be performed between the tensors, you can view the [comprehensive list here](https://www.tensorflow.org/api_guides/python/math_ops).

But let’s learn how to perform some basic operations,

### Addition

```
x = tf.add(5, 2) # 7
```

You’ll start with the add function. The `tf.add()` function does exactly what you expect it to do. It takes in two numbers, two tensors, or one of each, and returns their sum as a tensor.

### Subtraction and Multiplication

Here’s an example with subtraction and multiplication.

```
x = tf.subtract(10, 4) # 6y = tf.multiply(2, 5) # 10
```

The `x` tensor will evaluate to `6`, because `10 - 4 = 6`. The `y` tensor will evaluate to `10`, because `2 * 5 = 10`. That was easy!

### Converting types

It may be necessary to convert between types to make certain operators work together. For example, if you tried the following, it would fail with an exception:

```
tf.subtract(tf.constant(2.0),tf.constant(1)) # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```

That’s because the constant `1` is an integer but the constant `2.0` is a floating point value and `subtract` expects them to match.

In cases like these, you can either make sure your data is all of the same type, or you can cast a value to another type. In this case, converting the `2.0` to an integer before subtracting, like so, will give the correct result:

```
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
```

Let’s use these functions to run a simple computation,

*Tensorflow math functions*

Move we begin with implementing Machine learning / Deep learning models in Tensorflow let’s understand fundamentals of Linear regression machine learning model.

* * *

**Linear Regression**

 ![](https://cdn-images-1.medium.com/max/1200/1*A16fQML5oFHdC1p-evzGKw.png)

To the left, it’s the plot of the **_size vs the price_** from the boston house pricing dataset **_._** Given this dataset we need to find a relationship between size and the house price so that we could suggest a fair price for a new house to be sold given its size.

Linear Regression is a **Linear Model**. Which means, we will establish a linear relationship between the input variables( **X** ) and single output variable( **Y** ). When the input( **X** ) is a single variable this model is called **Simple Linear Regression** and when there are multiple input variables( **X** ), it is called **Multiple Linear Regression**.

 ![](https://cdn-images-1.medium.com/max/1600/1*ine7Jaq0NXNyw5dxFZxxXQ.png)

This is called Supervised learning, we take the set of **right answers** , find a pattern in it and then use it to make predictions. Since we are predicting the house price, which is a real and continuous valued output, the prediction problem is called as regression.

#### The model or the hypothesis(h)

As we discussed earlier we have an input variable **X** and one output variable **Y**. And we want to build linear relationship between these variables. The input variable is called **Independent Variable** and the output variable is called **Dependent Variable**. Since we are defining a linear relationship it can be defined as follows:

 ![](https://cdn-images-1.medium.com/max/1600/1*s8jNZDyHa9EIW7m-5dp-kg.png)

```
The θ1 is the coefficient and θo is called bias coefficient, which are also called the parameters of the model or hypothesis h(x). This equation is similar to the line equation y = m * x + b with m = θ1(Slope) and b=θo (Intercept), in Simple Linear Regression model we want to draw a line between X and Y which estimates the relationship between X and Y.
```

But how do we find these coefficients? That’s the learning procedure. The technique or algorithm used for learning is called as the learning algorithm. We’ll be using **Gradient Descent Algorithm&nbsp;**.

 ![](https://cdn-images-1.medium.com/max/1600/1*MGNpWCqHS0aq0gjy6EINJw.png)

Given the input X the parameters **_θo, θ1_** control the output **_Y._**

> In any supervised machine learning task we start with the labelled datasets. Hence, input features X and output Y are pretty much known and will not change. Hence, we can consider them to be constants in the equation. The real variables in our equation are `θ0` and `θ1 which influences the prediction.`

But we need a way see how to choose the prediction (Y^) so that its close value actual value Y. **_We need a way to quantify how good the parameters are. That’s where the error/cost functions are used._**

* * *

#### **The cost&nbsp;function**

One common function that is often used is [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), which measure the difference between the actual value from the dataset and the estimated value (the prediction). It looks like this:

 ![](https://cdn-images-1.medium.com/max/1600/1*20m_U-H6EIcxlN2k07Z7oQ.png)

We can [adjust the equation a little](https://datascience.stackexchange.com/questions/10188/why-do-cost-functions-use-the-square-error) to make the calculation little more simple.

 ![](https://cdn-images-1.medium.com/max/1600/1*ACReddqEPXxlQKpQopjoSw.png)

* * *

 ![](https://cdn-images-1.medium.com/max/1200/1*6PxCCeIsTBiUah4VcWAazw.png)

Here is the summary,

- The hypothesis h(x) defines the linear model with parameters θo and θ1.
- The cost function quantifies how the good the parameters are. Poor prediction leads to high value of cost function.
- The goal now is to continuously update the values of **_θo_** and **_θ1_** so that the cost function reduces after every update.

We’ll be using **Gradient Descent algorithm** to optimize and update the **_θo_** and **_θ1_** in such a way that the error/cost function decreases with every update.

* * *

#### Gradient Descent

 ![](https://cdn-images-1.medium.com/max/1200/1*i9MV5pQZrb6WBER7z9Opjw.png)

 ![](https://cdn-images-1.medium.com/max/1200/1*KXhoClXogcckzwxfvIFReQ.png)
*Gradient Decent approach and&nbsp;working*

 ![](https://cdn-images-1.medium.com/max/1600/1*HYHj6X8fs73zVd1bRQvolA.png)
*Plot of Cost function J w.r.t the parameters `θ0` and&nbsp;`θ1`*

So you can see that we need to move towards the bottom by changing the values of parameters where the value of `J` is the lowest. We’ll be using gradient descent to achieve it. As per the algorithm, we need to repeat the below procedure till convergence,

 ![](https://cdn-images-1.medium.com/max/1600/1*_S_pUfPD5yKA7rsIzt9uHw.png)

The derivative of the cost function **_J(θo,θ1)_** w.r.t parameters **_θo_** and **θ1** defines the direction in which the parameters have to changed (increased or decreased) in order to move get to a point where the value of cost function J(θo, θ1) is least.

 ![](https://cdn-images-1.medium.com/max/1600/1*YFfGHuD1Wic22RFeGHWSMA.png)

Don’t worry if the math of gradient descent looks tricky, Tensorflow makes it really easy to apply **gradient descent** on your dataset. Now let’ move onto the final part of the blog to build a linear regression model using tensorflow to predict house price.

* * *

#### Linear Regression for predicting house price using Tensorflow

Firstly let’s load, normalize and plot the dataset, here is how the plot of **size vs price** of house looks like this,

 ![](https://cdn-images-1.medium.com/max/1600/1*XF8jSYxlFb1D9j_ZfDd6_A.png)
*Plot of Size Vs Price of the house after normalization*

Here is the code for reading the dataset, normalizing and plotting it. Here is the link to notebook cell,

Pheww! That was Linear regression from you! Let’s learn how to linear regression model using tensorflow.

* * *

#### **The Linear regression model in Tensorflow**

We know that the Linear regression model is represented by the following equation,

 ![](https://cdn-images-1.medium.com/max/1600/1*s8jNZDyHa9EIW7m-5dp-kg.png)

Firstly, Since we’ll be substituting the values of the input feature **_X_** and the output label **_Y_** with the values from the dataset during session evaluation we represent them with **_tensorflow placeholders (tf.placeholder)_**,

```
X = tf.placeholder("float")
  Y = tf.placeholder("float")
```

But the parameters `θ0` and `θ1` are updated continuously during the optimization process, hence they are represented using **_tensorflow variables (tf.variable)._**

```
# The parameters theta0 and theta1
  theta1 = tf.Variable(np.random.randn(), name="weight")
  theta0 = tf.Variable(np.random.randn(), name="bias")
```

Here is how the linear regression model **_θo + θ1 \* x_** is implemented.

```
# Hypothesis = theta0 + theta1 * X

  x_theta1 = tf.multiply(X, theta1)
  model = tf.add(x_theta1 , theta0)
```

Let’s put all pieces together,

*Linear Regression model using Tensorflow*

* * *

#### **The Cost and the Optimization function**

Now given the model, and the list containing the actual house price we should be able to calculate the cost function, and here’s how to do it,

- `tf.pow(model — Y, 2)` returns the square error, where Y is the actual house price from the data and model gives the prediction value.
- `tf.reduce_sum` is used to calculate the sum of a tensor.
- Hence `tf.reduce_sum(tf.pow(model — Y, 2)) / 2 * size` represents the cost function.

Implementing the gradient descent optimizer for minimizing the cost is merely a task of calling the prebuilt functionality in tensorflow,

```
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate)
 # optimization tensor.
 optimizer = gradient_descent.minimize(cost_function)
```

Again, here’s how it looks when we put all the pieces together,

*cost function and gradient descent optimizer in tensorflow*

* * *

#### Start the&nbsp;training

As we discussed earlier a new session has to be created to evaluate the tensors. In the statement below we create a new session and evaluate the **_gradient descent optimizer,_** we run the optimization for over 100 times.

The cost function can valuated by running its tensor within the tensorflow session,

```
tuning_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
```

#### Making the prediction

Once the training is complete the optimized parameters **_θo_** and **_θ1_** are valuated to make the prediction using the linear regression model,

```
prediction = sess.run(theta1) * X_train + sess.run(theta0)
```

* * *

#### Plotting the&nbsp;results

The straight line represents the best fit produced by the trained parameters, the points in red represent the training dataset and the ones in green represents the test dataset. As you see that the straight line fits really well through both the test and the training dataset.

 ![](https://cdn-images-1.medium.com/max/1600/1*FsNlw2UeMKYvvZWigJSeYg.png)

* * *

That’s for now&nbsp;! [Here is the link to the Google collab notebook of the code](https://drive.google.com/file/d/1XNWM80G-MeXOcy_T1M9PAUq_3dIm9lk_/view?usp=sharing). We went through the journey of learning basics of Tensorflow and Linear regression and building a linear regression model to predict house price, it hope it was exciting.

We’ll be back with more exciting discussions not just on building Deep learning models but also on building robust infrastructure to store, consume and process data at scale. Till then, happy coding&nbsp;!
