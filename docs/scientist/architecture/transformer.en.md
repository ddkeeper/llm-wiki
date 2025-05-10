# The Transformer

!!! info "Source"
    This post is adapted from Jay Alammar's [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
    
    Author: [Jay Alammar](https://jalammar.github.io/)

In this post, we will look at **The Transformer** – a model that uses attention to boost the speed with which these models can be trained. The Transformer outperforms the Google Neural Machine Translation model in specific tasks. The biggest benefit, however, comes from how The Transformer lends itself to parallelization.

The Transformer was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Here's the architecture of the model - "A bit of" complex. No  Worry, we will attempt to oversimplify things a bit and introduce the concepts one by one to hopefully make it easier to understand to people without in-depth knowledge of the subject matter.

![transformer_architecture](./images/transformer/transformer_architecture.png)

!!! important "Transformer Implementations"
    - A TensorFlow implementation is available as part of the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) package
    - Harvard's NLP group created a [guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## A High-Level Look

Let's begin by looking at the model as a single black box. In a machine translation application, it would take a sentence in one language, and output its translation in another.

![transformer overview](./images/transformer/the_transformer_3.png)

Popping open that Optimus Prime goodness, we see an encoding component, a decoding component, and connections between them.

![encoder decoder overview](./images/transformer/The_transformer_encoders_decoders.png)

The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there's nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.

![encoder decoder stack](./images/transformer/The_transformer_encoder_decoder_stack.png)

The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:

![transformer encoder](./images/transformer/Transformer_encoder.png)

The encoder's inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We'll look closer at self-attention later in the post.

The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in [seq2seq models](./seq2seq.html)).

![transformer decoder](./images/transformer/Transformer_decoder.png)

## Bringing The Tensors Into The Picture

Now that we've seen the major components of the model, let's look at how the data flows between them. As is the case in NLP applications, we begin by turning each input word into a vector using an embedding algorithm.

![embeddings](./images/transformer/embeddings.png){: style="display: block; margin: 0 auto; max-width: 100%;"}

Each word is embedded into a vector of size 512. We'll represent these vectors with these simple boxes
{: style="text-align: center; font-size: 0.8em;"}

The embedding only happens in the bottom-most encoder. The abstraction that comes out of the top encoder is then processed by all the decoders, with each decoder having access to all the encoder outputs directly through the encoder-decoder attention mechanisms we'll discuss later.

The word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

Let's look at the various vectors/tensors and how they flow between components to better understand the model.

As is the case in NLP applications, we begin by converting each input word to a vector using an embedding algorithm.

<div style="text-align: center; max-width:100%; margin: 0 auto;">
    <img src="./images/transformer/embeddings.png" alt="embeddings" style="width: 100%;">
    <div style="text-align: center; font-style; font-size: 0.8em;">
        Each word is embedded into a vector of size 512. We represent these vectors with these simple boxes
    </div>
</div>

The embedding only happens in the bottom-most encoder. All encoders have the common pattern that they receive a list of vectors each of the size 512 – the size of the embedding vectors:

- In the bottom encoder that's the word embeddings;
- In other encoders, that's the output of the encoder that's directly below.

The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

![encoder with tensors](./images/transformer/encoder_with_tensors.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Here we begin to see one key property of the Transformer: each word in the input sentence flows through its own path through the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer, however, does not have those dependencies, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

Next, we'll switch to looking at the self attention mechanism in detail.

## Self-Attention at a High Level

Don't be fooled by me throwing around the word "self-attention" like it's a concept everyone should be familiar with. I had personally never came across the concept until reading the Attention is All You Need paper. Let's distill how it works.

Say the following sentence is an input sentence we want to translate:

"The animal didn't cross the street because it was too tired"
{: style="color: rgb(179, 18, 18); font-size: 1.2em"} 

What does "it" in this sentence refer to? Is it referring to the street or to the animal? It's a simple question to a human, but not as simple to an algorithm.

When the model is processing the word "it", self-attention allows it to associate "it" with "animal".

As the model processes each word (each position in the input sequence), self-attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

If you're familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it's processing. Self-attention is the method the Transformer uses to bake the "understanding" of other relevant words into the one it's currently processing.

<div style="text-align: center; max-width:70%; margin: 0 auto;">
    <img src="./images/transformer/transformer_self-attention_visualization.png" alt="transformer self-attention visualization" style="width: 100%;">
    <div style="text-align: center; font-style; font-size: 0.8em;">
        When we encode the word "it" in the encoder stack (the top encoder in this case), part of the attention mechanism focuses on "The Animal", and bakes a part of its representation into the encoding of "it"
    </div>
</div>

## Now We’re Encoding!

As we've mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a 'self-attention' layer, then into a feed-forward neural network, then sends out the output upward to the next encoder.

![encoder with tensors 2](./images/transformer/encoder_with_tensors_2.png){: style="display: block; margin: 0 auto; max-width: 100%;"}

The word at each position passes through a self-attention process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately.
{: style="text-align: center; font-size: 0.8em;"}

## Self-Attention in Detail

Let's first look at how to calculate self-attention using vectors, then proceed to look at how it's actually implemented – using matrices.

The **first step** in calculating self-attention is to create three vectors from each of the encoder's input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimension is 64, while the embedding and encoder input/output vectors have dimension of 512. They don't HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

<div style="text-align: center; max-width:80%; margin: 0 auto;">
    <img src="./images/transformer/transformer_self_attention_vectors.png" alt="transformer self attention vectors" style="width: 100%;">
    <div style="text-align: center; font-size: 0.8em;">
    Multiplying \( x_1 \) by the \( W^Q \) matrix produces \( q_1 \), the "query" vector associated with that word. We end up creating a "query" (\( Q \)), a "key" (\( K \)), and a "value" (\( V \)) projection of each word in the input sentence
    </div>
</div>

What are the "query", "key", and "value" vectors? They're abstractions that are useful for calculating and thinking about attention. You'll see how they're used in the next few steps.

The **second step** in calculating self-attention is to calculate a score. Say we're calculating the self-attention for the first word in this example, "Thinking". We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we're scoring. So if we're processing the self-attention for the word at position #1, the first score would be the dot product of \( q_1 \) and \( k_1 \). The second score would be the dot product of \( q_1 \) and \( k_2 \).

![transformer self attention score](./images/transformer/transformer_self_attention_score.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

The **third and fourth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they're all positive and add up to 1.

![self-attention softmax](./images/transformer/self-attention_softmax.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it's useful to attend to another word that is relevant to the current word.
{: style="text-align: center; font-size: 0.8em;"}

The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

![self attention output](./images/transformer/self-attention-output.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. Now let's look at that.

## Matrix Calculation of Self-Attention

**The first step** is to calculate the Query, Key, and Value matrices. We do this by packing our embeddings into a matrix X, and multiplying it by the weight matrices we've trained (WQ, WK, WV).

![self attention matrix calculation](./images/transformer/self-attention-matrix-calculation.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Every row in the X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 in this toy example), and the q/k/v vectors (64, or 3 in this toy example).
{: style="text-align: center; font-size: 0.8em;"}

**Finally**, since we're dealing with matrices, we can condense steps 2 through 6 in one formula to calculate the outputs of the self-attention layer:

![self attention matrix calculation 2](./images/transformer/self-attention-matrix-calculation-2.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

The self-attention calculation in matrix form
{: style="text-align: center; font-size: 0.8em;"}

## The Beast With Many Heads

The paper further refined the self-attention layer by adding a mechanism called "multi-headed" attention. This improves the performance of the attention layer in two ways:

- It expands the model's ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other word, but it could be dominated by the actual word itself. It would be useful if we're translating a sentence like "The animal didn't cross the street because it was too tired", we would want to know which words "it" refers to.

- It gives the attention layer multiple "representation subspaces". With multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

![transformer attention heads qkv](./images/transformer/transformer_attention_heads_qkv.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply \( X \) by the \( W^Q \)/\( W^Q \)/\( W^V \) matrices to produce Q/K/V matrices.
{: style="text-align: center; font-size: 0.8em;"}


If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices. 

![transformer attention heads z](./images/transformer/transformer_attention_heads_z.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it's expecting a single matrix (a vector for each word). So we need a way to condense these eight matrices into a single matrix.

How do we do that? We concat the matrices then multiply them by an additional weights matrix \( W^O \).

![transformer attention heads weight matrix o](./images/transformer/transformer_attention_heads_weight_matrix_o.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place:

![transformer multi-headed self-attention-recap](./images/transformer/transformer_multi-headed_self-attention-recap.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Now that we have touched upon attention heads, let’s revisit our example from before to see where the different attention heads are focusing as we encode the word “it” in our example sentence:

![transformer self-attention visualization 2](./images/transformer/transformer_self-attention_visualization_2.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".
{: style="text-align: center; font-size: 0.8em;"}

If we add all the attention heads to the picture, however, things can be harder to interpret:

![transformer self-attention visualization 3](./images/transformer/transformer_self-attention_visualization_3.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

## Representing The Order of The Sequence Using Positional Encoding

One thing that's missing from the model as we've described it so far is a way to account for the order of the words in the input sequence.

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they're projected into Q/K/V vectors and during dot-product attention.

![transformer positional encoding vectors](./images/transformer/transformer_positional_encoding_vectors.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

To give the model a sense of the order of the words, we add positional encoding vectors -- the values of which follow a specific pattern.
{: style="text-align: center; font-size: 0.8em;"}

If we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

![transformer positional encoding example](./images/transformer/transformer_positional_encoding_example.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

A real example of positional encoding with a toy embedding size of 4
{: style="text-align: center; font-size: 0.8em;"}


## The Residuals

One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step.

![transformer residual layer norm](./images/transformer/transformer_resideual_layer_norm.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

If we're to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

![transformer residual layer norm 2](./images/transformer/transformer_resideual_layer_norm_2.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

![transformer residual layer norm 3](./images/transformer/transformer_resideual_layer_norm_3.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

## The Decoder Side

Now that we've covered most of the concepts on the encoder side, we can look at how the decoder operates. The decoder has all the same pieces, but with a few tweaks here and there.

The encoder starts by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its "encoder-decoder attention" layer which helps the decoder focus on appropriate places in the input sequence:

![transformer decoding 1](./images/transformer/transformer_decoding_1.gif){: style="display: block; margin: 0 auto; max-width: 80%;"}

After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case).
{: style="text-align: center; font-size: 0.8em;"}

The following steps repeat the process until a special symbol is reached indicating the transformer has completed its output. The output of each step is fed into the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

![transformer decoding 2](./images/transformer/transformer_decoding_2.gif){: style="display: block; margin: 0 auto; max-width: 80%;"}

The self attention layers in the decoder operate in a slightly different way than the ones in the encoder:

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to `-inf`) before the softmax step in the self-attention calculation.

The "encoder-decoder attention" layer works just like multiheaded self-attention, except it creates its Queries matrix (More explicitly - \( X_Q \)) from the layer below it, and takes the Keys (\( X_K \)) and Values matrix (\( X_V \)) from the output of the encoder stack.

## The Final Linear and Softmax Layer

The decoder stack outputs a vector of floats. How do we turn that into a word? That's the job of the final linear layer and softmax layer.

The linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much larger vector called a logits vector.

Let's assume our model knows 10,000 unique English words (our model's "output vocabulary") that it's learned from its training dataset. So the logits vector would have 10,000 cells – each cell corresponding to the score of a unique word. That's how we interpret the output of the model followed by the linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

![transformer decoder output softmax](./images/transformer/transformer_decoder_output_softmax.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

This figure starts from the bottom with the vector we receive from the decoder stack. It is then turned into an output word.
{: style="text-align: center; font-size: 0.8em;"}

## Recap Of Training

Now that we've covered the entire forward pass through a trained Transformer, it would be useful to look at how training works.

During training, an untrained model would go through the exact same forward pass. But since we are training it on a labeled training dataset, we can compare its output with the actual correct output.

To visualize this, let's assume our output vocabulary only contains six words ("a", "am", "i", "thanks", "student", and "<eos>" (short for 'end of sentence')).

![vocabulary](./images/transformer/vocabulary.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

The output vocabulary of our model is created in the preprocessing phase before we even begin training.
{: style="text-align: center; font-size: 0.8em;"}

Once we define our output vocabulary, we can use a vector of the same width to represent each word in the vocabulary. This is called a one-hot encoding. For example, we can represent the word "am" using the following vector:

![one-hot vocabulary example](./images/transformer/one-hot-vocabulary-example.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Example: one-hot encoding of our output vocabulary
{: style="text-align: center; font-size: 0.8em;"}

Following this recap, let’s discuss the model’s loss function – the metric we are optimizing during the training phase to lead up to a trained and hopefully amazingly accurate model.

## The Loss Function

Let's say we're training our model. Let's say it's the first step in the training phase, and we're training it on a simple example: translating "merci" into "thanks".

What this means, is that we want the output to be a probability distribution indicating the word “thanks”. But since this model is not yet trained, that’s unlikely to happen just yet.

![transformer logits output and label](./images/transformer/transformer_logits_output_and_label.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Since the model's parameters (weights) are all randomly initialized, the (untrained) model produces a probability distribution with arbitrary values for each cell/word. We can compare it with the actual output, and then use backpropagation to adjust all the model's weights to make the output closer to the desired output.
{: style="text-align: center; font-size: 0.8em;"}

How do we compare two probability distributions? We simply subtract one from the other. For more details, look at [cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) and [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).

But note that this is an oversimplified example. More realistically, we'll use longer sentences. For example – input: "je suis étudiant" and expected output: "i am a student". This actually means we want our model to output probability distributions that look like this:

* Each probability distribution is represented by a vector of vocab_size width (6 in our toy example, but more realistically a number like 30,000 or 50,000)
* The first probability distribution has the highest probability at the cell associated with the word "i"
* The second probability distribution has the highest probability at the cell associated with the word "am"
* And so on, until the fifth output distribution points to `<end of sentence>` token, which also has a cell associated with it in our 10,000-element vocabulary

![output target probability distributions](./images/transformer/output_target_probability_distributions.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

The targeted probability distributions we'll train our model against in the training example for one sample sentence.
{: style="text-align: center; font-size: 0.8em;"}

After training the model for enough time on a large enough dataset, we would hope the produced probability distributions would look like this:

![output trained model probability distributions](./images/transformer/output_trained_model_probability_distributions.png){: style="display: block; margin: 0 auto; max-width: 80%;"}

Hopefully upon training, the model would output the right translation we expect. Of course it's no real indication if this phrase was part of the training dataset (see: [cross validation](https://www.youtube.com/watch?v=TIgfjmp-4BA)). Notice that every position gets a little bit of probability even if it's unlikely to be the output of that time step -- that's a very useful property of softmax which helps the training process.
{: style="text-align: center; font-size: 0.8em;"}

Now, because the model produces the outputs one at a time, we can assume that the model is selecting the word with the highest probability from that probability distribution and throwing away the rest. That’s one way to do it (called greedy decoding). Another way to do it would be to hold on to, say, the top two words (say, ‘I’ and ‘a’ for example), then in the next step, run the model twice: once assuming the first output position was the word ‘I’, and another time assuming the first output position was the word ‘a’, and whichever version produced less error considering both positions #1 and #2 is kept. We repeat this for positions #2 and #3…etc. This method is called “beam search”, where in our example, beam_size was two (meaning that at all times, two partial hypotheses (unfinished translations) are kept in memory), and top_beams is also two (meaning we’ll return two translations). These are both hyperparameters that you can experiment with.

## Go Forth And Transform

I hope you've found this a useful guide to get an intuitive understanding of the Transformer. If you want to go deeper, I'd suggest these next steps:

* Read the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).

* Watch [Łukasz Kaiser's talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details.

* Play with the [Jupyter Notebook provided as part of the Tensor2Tensor library](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb).

* Explore the [Tensor2Tensor library](https://github.com/tensorflow/tensor2tensor).