![](https://cdn-images-1.medium.com/max/1600/1*DmycpagapYt7lmjtabWMIg.png)

Sequence-to-sequence(Seq-2-Seq) is a _problem setting_, where your input is a sequence and your output is also a sequence. Typical examples of sequence-to-sequence problems are machine translation, question answering, generating natural language description of videos, automatic summarization, etc. LSTM and RNN are _models_, that is, they can be trained using some data and then be used to predict on new data. You can have _sequence-to-sequence models_ that use LSTMs/RNNs as modules inside them, where a “sequence-to-sequence model” is just a model that works for sequence to sequence tasks.

Here is architecture of Seq2seq model,&nbsp;

 ![](https://cdn-images-1.medium.com/max/1600/1*Ir8gyc1-AZh_MzOO7En03A.png)

&nbsp;Seq2Seq model has got 2 pieces, the part in green is the **_encoder_** and the one in blue is the **_decoder_**.

The encoder take the inputs from the source languages and outputs a compact encoded representation of the source sentence, the decoder then takes this encoded representation to output the sentences in the target language.&nbsp;

Each output of the decoder contains softmax probabilities across vocabulary containing all possible words, but picking the words with highest probability at each outputs from the decoder doesn’t always ensure the best translation. The decoder takes in the output of the first timestep as the input for the next, It is quite possible that a less probable output of one layer fed as input to the next can give a highly probable output in a way that their sum probabilities are high. Beam search help us identify what possible combination of words being sampled in each layer could give us the high confidence on the combination of words selected altogether.&nbsp;

* * *

### Beam Search

Let’s discuss this beam search strategy w.r.t to the of Machine Translation/Language-translation use-case.

#### A Perspective…

Machine translation model can be thought of as a **“** [**Conditional Language Model**](https://medium.com/r/?url=https%3A%2F%2Fwww.52coding.com.cn%2Findex.php%3F%2FArticles%2Fsingle%2F62) **”** &nbsp;, for a system that translates French to English, the model can be thought of probability of English sentence conditioned on French sentence.

Due to the probabilistic behavior of our model, there can be more than one translation to input French sentence. **_Should we pick any one at random?_ Absolutely, NO**. We aim to find the most suitable translation _(i.e maximize the join probability distribution over output words given the input sentence.)_

#### Should we act Greedy&nbsp;?

I won’t say YES or NO but, won’t recommend being greedy. Because it may get you sub-optimal translations. Until and unless you are ready to compromise on accuracy, you should not go for this approach.

#### Why not Greedy&nbsp;?

For a given French sentence, consider a English Translation as — 

**Translation 1:** _I am visiting NY this year end._

**Translation 2:** _I am going to be visiting NY this year end._

Considering English language and it’s rules in-general,

> _P(going|i, am) \> P(visiting|i, am)_

If you go with the greedy approach, model is likely to choose Translation 2 the output instead of 1. Because the probability of “ **going** ” after “ **i am** ” should be maximum over the distribution of possible words in the vocabulary compared to “ **visiting** ” after “ **i am** ”.

#### Let’s get to the&nbsp;point…

**Beam Search** is an **approximate search strategy** that tries to solve this in a efficient way. In it’s simplest representation, **B** _(Beam width)_ is the only tunable hyper-parameter for tweaking translation results. **B** in-general decides the number of words to keep in-memory at each step to permute the possibilities.

 ![](https://cdn-images-1.medium.com/max/1600/1*dSjT2an3yo-iiHc0nzWJbA.png)

_2 steps to Beam Search_

We will illustrate this with an example at each step level.

**Step — 1**

As a part of Step-1 to Beam Search the decoder network outputs the [**Softmax probabilities**](https://medium.com/r/?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSoftmax_function) of the top B probabilities and keep them in-memory. Let’s say _am, going, visiting_ are the top 3 probable words.

**Step — 2**

As a part of Step-2 to Beam Search we hardwire the Y1 in the decoder network and try to predict the next probable word if Y1 has already occurred.

Our aim is to maximize the probability of Y1 and Y2 occurring together.

> _P (Y1, Y2 | X) = P (Y1 | X) \* P (Y2 | X, Y1)_

Here X = x1, x2…. xn (all words in the input)

**Step — 3**

Again to work in step-3, we take top B probable (Y1, Y2) pairs from step-2 and hardwire them in the decoder network and try to find conditional probability of Y3. i.e. **P(Y3 | X, Y1, Y2)**

Similar to previous 2 steps we again find top B probable 3 word sequences and so on. We keep this process till **\<EOS\>** is reached.

#### Let’s comment on&nbsp;B

- If we set **B = 1** then the technique is same as **Greedy Search**.
- **Larger the B** &nbsp;, good chances for better output sequences, but would consume more resources and computation power.
- **Smaller the B** &nbsp;, not so good results, but much fast and memory efficient.

* * *
