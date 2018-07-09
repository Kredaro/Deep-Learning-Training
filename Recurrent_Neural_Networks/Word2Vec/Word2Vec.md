## Word2Vec
Semantic relationships between the words has be someway captured so that it could act as a enriched representation 
than what one hot vectors could represent. 

The distance between the vectors in one hot representation are all the same, it doesn't capture the relationship between certain words, 

For example consider the 4 following words, 

```
Dog, puppy, truck, car 
```

Dog and puppy, truck and car are related to each other. What if we could have feature rich representation where the distance between the vectors of related wrods are closer, this rich representation of words could serve as better. This feature representation can be achieved using a 300/600 or higher dimension vector. 

Here is the 2D representation of the 300 dimensional vector representation for the words, you can see in the image that the words which are similar in meaning are close to each other on the plot.

![The word2Vec representation plot](https://github.com/Kredoai/Deep-Learning-Training/blob/master/Recurrent_Neural_Networks/Word2Vec/images/plot-vec.png)


## The Model

The skip-gram neural network model is actually surprisingly simple in its most basic form; I think it’s the all the little tweaks and enhancements that start to clutter the explanation.



Let’s start with a high-level insight about where we’re going. Word2Vec uses a trick you may have seen elsewhere in machine learning. We’re going to train a simple neural network with a single hidden layer to perform a certain task, but then we’re not actually going to use that neural network for the task we trained it on! Instead, the goal is actually just to learn the weights of the hidden layer–we’ll see that these weights are actually the “word vectors” that we’re trying to learn.


Another place you may have seen this trick is in unsupervised feature learning, where you train an auto-encoder to compress an input vector in the hidden layer, and decompress it back to the original in the output layer. After training it, you strip off the output layer (the decompression step) and just use the hidden layer--it's a trick for learning good image features without having labeled training data.

## The Fake Task

So now we need to talk about this “fake” task that we’re going to build the neural network to perform, and then we’ll come back later to how this indirectly gives us those word vectors that we are really after.



We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.


 When I say "nearby", there is actually a "window size" parameter to the algorithm. A typical window size might be 5, meaning 5 words behind and 5 words ahead (10 in total).


The output probabilities are going to relate to how likely it is find each vocabulary word nearby our input word. For example, if you gave the trained network the input word “Soviet”, the output probabilities are going to be much higher for words like “Union” and “Russia” than for unrelated words like “watermelon” and “kangaroo”.



We’ll train the neural network to do this by feeding it word pairs found in our training documents. The below example shows some of the training samples (word pairs) we would take from the sentence “The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. The word highlighted in blue is the input word.



[![Training Data](http://mccormickml.com/assets/word2vec/training_data.png)](http://mccormickml.com/assets/word2vec/training_data.png)



The network is going to learn the statistics from the number of times each pairing shows up. So, for example, the network is probably going to get many more training samples of (“Soviet”, “Union”) than it is of (“Soviet”, “Sasquatch”). When the training is finished, if you give it the word “Soviet” as input, then it will output a much higher probability for “Union” or “Russia” than it will for “Sasquatch”.


## Model Details


So how is this all represented?



First of all, you know you can’t feed a word just as a text string to a neural network, so we need a way to represent the words to the network. To do this, we first build a vocabulary of words from our training documents–let’s say we have a vocabulary of 10,000 unique words.



We’re going to represent an input word like “ants” as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) and we’ll place a “1” in the position corresponding to the word “ants”, and 0s in all of the other positions.



The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, the probability that a randomly selected nearby word is that vocabulary word.



Here’s the architecture of our neural network.



[![Skip-gram Neural Network Architecture](http://mccormickml.com/assets/word2vec/skip_gram_net_arch.png)](http://mccormickml.com/assets/word2vec/skip_gram_net_arch.png)



There is no activation function on the hidden layer neurons, but the output neurons use softmax. We’ll come back to this later.



When _training_ this network on word pairs, the input is a one-hot vector representing the input word and the training output _is also a one-hot vector_ representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating point values, _not_ a one-hot vector).



For our example, we’re going to say that we’re learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).


300 features is what Google used in their published model trained on the Google news dataset (you can download it from [here](https://code.google.com/archive/p/word2vec/)). The number of features is a "hyper parameter" that you would just have to tune to your application (that is, try different values and see what yields the best results).


If you look at the _rows_ of this weight matrix, these are actually what will be our word vectors!



[![Hidden Layer Weight Matrix](http://mccormickml.com/assets/word2vec/word2vec_weight_matrix_lookup_table.png)](http://mccormickml.com/assets/word2vec/word2vec_weight_matrix_lookup_table.png)



So the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!


Let’s get back, though, to working through the definition of this model that we’re going to train.


Now, you might be asking yourself–“That one-hot vector is almost all zeros… what’s the effect of that?” If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, it will effectively just _select_ the matrix row corresponding to the “1”. Here’s a small example to give you a visual.



[![Effect of matrix multiplication with a one-hot vector](http://mccormickml.com/assets/word2vec/matrix_mult_w_one_hot.png)](http://mccormickml.com/assets/word2vec/matrix_mult_w_one_hot.png)



This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the “word vector” for the input word.


## The Output Layer


The `1 x 300` word vector for “ants” then gets fed to the output layer. The output layer is a softmax regression classifier. There’s an in-depth tutorial on Softmax Regression [here](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/), but the gist of it is that each output neuron (one per word in our vocabulary!) will produce an output between 0 and 1, and the sum of all these output values will add up to 1.



Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, then it applies the function `exp(x)` to the result. Finally, in order to get the outputs to sum up to 1, we divide this result by the sum of the results from _all_ 10,000 output nodes.



Here’s an illustration of calculating the output of the output neuron for the word “car”.



[![Behavior of the output neuron](http://mccormickml.com/assets/word2vec/output_weights_function.png)](http://mccormickml.com/assets/word2vec/output_weights_function.png)


Note that neural network does not know anything about the offset of the output word relative to the input word. It _does not_ learn a different set of probabilities for the word before the input versus the word after. To understand the implication, let's say that in our training corpus, _every single occurrence_ of the word 'York' is preceded by the word 'New'. That is, at least according to the training data, there is a 100% probability that 'New' will be in the vicinity of 'York'. However, if we take the 10 words in the vicinity of 'York' and randomly pick one of them, the probability of it being 'New' _is not_ 100%; you may have picked one of the other words in the vicinity.

## Intuition

Ok, are you ready for an exciting bit of insight into this network?



If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. And one way for the network to output similar context predictions for these two words is if _the word vectors are similar_. So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! Ta da!



And what does it mean for two words to have similar contexts? I think you could expect that synonyms like “intelligent” and “smart” would have very similar contexts. Or that words that are related, like “engine” and “transmission”, would probably have similar contexts as well.



This can also handle stemming for you – the network will likely learn similar word vectors for the words “ant” and “ants” because these should have similar contexts.
