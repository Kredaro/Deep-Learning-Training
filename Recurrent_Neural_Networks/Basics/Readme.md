## What is sequence data&nbsp;?

When data has a particular order in which related things follow each other or there is an arrangement of data in an particular order it is called as sequence data.&nbsp;

For example, Audio clip, Text information, DNA information. These information has an order associated with how the data is arranged, A sentence wouldn’t make sense if the order is shuffled randomly.&nbsp;

When we train AI system on sequence the output can vary based on the objective.&nbsp;

#### Representing sequential data

 ![](https://cdn-images-1.medium.com/max/1600/1*J7vswpka3kCxzu3E2kPZvg.png)
*The superscript is used to access a particular word in a&nbsp;sequence&nbsp;*

#### One hot representation

The input words are converted to one hot representation

 ![](https://cdn-images-1.medium.com/max/1600/1*iSePGdCsqsh_SUq7vZimFQ.png)

* * *

Need a special type of neural network to map the sequence data to their corresponding output Y.

Normally we use feedforward networks which propogates through Hidden layers through output layer,&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*0-AMhOyTuycxwEsWCNDw7Q.png)

Issue dealing with sequential data, Sometimes the output depends not just on the current input but also on sequence of previous inputs/outputs.

 ![](https://cdn-images-1.medium.com/max/1600/1*gik6bSSFPNLr0J3VRnzX7w.png)

#### Introducing Recurrrent Neural&nbsp;Networks&nbsp;

 ![](https://cdn-images-1.medium.com/max/2000/1*u7JjmHyGXIbhwJBvoxDexw.png)
*Feedback from previous layers are&nbsp;taken&nbsp;*

 ![](https://cdn-images-1.medium.com/max/1600/1*YLcsegL4gPZzwGTkNcWWsA.png)

Takes feedback from all its previous layers&nbsp;, prediction of the first letter is influenced by its previous layer input “s”, but the prediction of the last letter “p” is influenced all the previous inputs “s”,”t”,”e”,”p”.

> This also has a drawback that the flow is unidirectional, prediction of second character is not influenced by the characters following it, this is essential in case of a natural language processing based problems. Bidirectional Recurrent Neural Network solves this&nbsp;problem.&nbsp;

 ![](https://cdn-images-1.medium.com/max/2000/1*BvNCWw0i5xEiRBbeiJ0sXA.png)

Inputs are fed as one hot encoded values, the output logits are passed through softmax and trained for categorical cross entropy loss.

> In RNN the set of weights which connects input layer and the hidden layer are the same across all time steps or the recurrent layers. But in RNN there are parameters which are connecting the horizontal layers, these too are also shared across all&nbsp;layers.

In the forward propagation representation below the weights connecting the inputs **_x\<t\>_** and their corresponding hidden layers are same for all recurrent layers and its represented by Wax&nbsp;. Similarly the weights connecting the each of the horizontal activation's **_a\<t\>_** are also same across all the horizontal layers.

 ![](https://cdn-images-1.medium.com/max/1600/1*MYvVNoL9OA8MDeJqmL0lPA.png)

In the generic formula to calculate the output **_y^\<t\>_** for layer t and the activation **_a\<t\>_** the weights **_Waa_** and Wax are same for all horizontal layers.

 ![](https://cdn-images-1.medium.com/max/1600/1*PSkdIY_c3xSA8ZRu_zIImA.png)

* * *

#### Types of&nbsp;RNN

 ![](https://cdn-images-1.medium.com/max/1600/1*CGuHQ0_8v4T1f4FCpFM7bw.png)

* * *

#### Disadvantages of&nbsp;RNN’s&nbsp;

- Vanishing / exploding gradient problem&nbsp;
- Failure in capturing Long term/long distance relationships

GRU’s and LSTM’s help solve these above drawbacks of RNN’s&nbsp;

 ![](https://cdn-images-1.medium.com/max/2000/1*peLh8VnN4iz47GjwhzKM1w.png)

* * *

 ![](https://cdn-images-1.medium.com/max/1600/1*Mze328ba1_WJi1755W2UvA.png)

 ![](https://cdn-images-1.medium.com/max/1600/1*Wz4jt8fvirvdRTKLuniL4Q.png)

* * *

#### Deep RNN

Multiple Recurrent Neural Network layers (LSTM, GRU and B-RNN) can be stacked together to form Deep RNN architecture.

 ![](https://cdn-images-1.medium.com/max/1600/1*uT7St2nCvqkLaH6bJcd_-Q.png)
