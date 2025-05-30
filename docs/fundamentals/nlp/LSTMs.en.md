# Understanding LSTM Networks

!!! info "Document Source"
    This document is adapted from Christopher Olah's [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
    
    Original author: [Christopher Olah](https://colah.github.io/)

## Recurrent Neural Networks

Humans don't start their thinking from scratch every second. As you read this article, you understand each word based on your understanding of previous words. You don't throw everything away and start thinking from scratch. Your thoughts have persistence.

Traditional neural networks can't do this, and it seems like a major shortcoming. For example, imagine you want to classify what kind of event is happening at each point in a movie. It's unclear how a traditional neural network could use its reasoning about previous events in the film to inform later ones.

Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.

![RNN-rolled](./images/LSTMs/RNN-rolled.png){: style="width:11.8%; display: block; margin:17px auto 5px"}

**A recurrent neural network and the loop within.**
{: style="text-align:center; margin-bottom:20px"}

In the above diagram, a chunk of neural network, (A), looks at some input (x_t) and outputs a value (h_t). A loop allows information to be passed from one step of the network to the next.

These loops make recurrent neural networks seem kind of mysterious. However, if you think a bit more, it turns out that they aren't all that different from a normal neural network. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Consider what happens when we unroll the loop:

![RNN-unrolled](./images/LSTMs/RNN-unrolled.png){: style="width:70%; display: block; margin:17px auto 5px"}

**An unrolled recurrent neural network.**
{: style="text-align:center; margin-bottom:20px"}

This chain-like nature reveals that recurrent neural networks are intimately related to sequences and lists. They're the natural architecture of neural network to use for such data.

And they certainly are used! In the last few years, there have been incredible success applying RNNs to a variety of problems: speech recognition, language modeling, translation, image captioning… The list goes on. I'll leave discussion of the amazing feats one can achieve with RNNs to Andrej Karpathy's excellent blog post, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). But they really are pretty amazing.

Essential to these successes is the use of "LSTMs," a very special kind of recurrent neural network which works, for many tasks, much better than the standard version. Almost all exciting results based on recurrent neural networks are achieved with them. It's these LSTMs that this essay will explore.

## The Problem of Long-Term Dependencies

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. If RNNs could do this, they'd be extremely useful. But can they? It depends.

Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in "the clouds are in the _sky_," we don't need any further context – it's pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place where it's needed is small, RNNs can learn to use the past information.

![RNN-shorttermdepdencies](./images/LSTMs/RNN-shorttermdepdencies.png){: style="width:50%; display: block; margin:17px auto"}

But there are also cases where we need more context. Consider trying to predict the last word in the text "I grew up in France… I speak fluent ==French=={: style="color: red; font-style: italic;"}." Recent information suggests that the next word is probably the name of a language, but if we want to know which language, we need the context of France, from further back. The gap between the relevant information and the point where it is needed can become very large.

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

![RNN-longtermdependencies](./images/LSTMs/RNN-longtermdependencies.png){: style="width:65.2%; display: block; margin:17px auto"}

In theory, RNNs are absolutely capable of handling such "long-term dependencies." A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don't seem to be able to learn them. The problem was explored in depth by [Hochreiter (1991) [German]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) and [Bengio, et al. (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf), who found some pretty fundamental reasons why it might be difficult.

Fortunately, LSTMs don't have this problem!

## Long Short Term Memory Networks

Long Short Term Memory networks – usually just called "LSTMs" – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf), and were refined and popularized by many people in following work[^1]. They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer.

![LSTM3-SimpleRNN](./images/LSTMs/LSTM3-SimpleRNN.png){: style="width:90%; display: block; margin:15px auto"}

**The repeating module in a standard RNN contains a single layer.**
{: style="text-align:center; margin-bottom:10px"}

LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![LSTM3-chain](./images/LSTMs/LSTM3-chain.png){: style="width:90%; display: block; margin:15px auto"}

**The repeating module in an LSTM contains four interacting layers.**
{: style="text-align:center; margin-bottom:10px"}

Don't worry about the details of what's going on. We'll walk through the LSTM diagram step by step later. For now, let's just try to get comfortable with the notation we'll be using.

![LSTM2-notation](./images/LSTMs/LSTM2-notation.png){: style="width:70%; display: block; margin:8px auto"}

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes represent learned neural network layers. Lines merging denote concatenation, while a line forking denotes its content being copied and the copies going to different locations.

## The Core Idea Behind LSTMs

The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.

The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged.

![LSTM3-C-line](./images/LSTMs/LSTM3-C-line.png){: style="width:90%; display: block; margin:8px auto"}

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

![LSTM3-gate](./images/LSTMs/LSTM3-gate.png){: style="width:12%; display: block; margin:8px auto"}

The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means "let nothing through," while a value of one means "let everything through!"

An LSTM has three of these gates, to protect and control the cell state.

## Step-by-Step LSTM Walk Through

### Step 1: Deciding What to Forget

The first step in our LSTM is to decide what information we're going to throw away from the cell state. This decision is made by a sigmoid layer called the "forget gate layer." It looks at (h_{t-1}) and (x_t), and outputs a number between 0 and 1 for each number in the cell state (C_{t-1}). A 1 represents "keep this" while a 0 represents "throw this away."

Let's go back to our example of a language model trying to predict the next word based on all the previous ones. In such a problem, the cell state might include the gender of the present subject, so that proper pronouns can be used. When we see a new subject, we need to forget the gender of the old subject.

![LSTM3-focus-f](./images/LSTMs/LSTM3-focus-f.png){: style="width:90%; display: block; margin:8px auto"}

### Step 2: Deciding What to Store

The next step is to decide what new information we're going to store in the cell state. This has two parts. First, a sigmoid layer called the "input gate layer" decides which values we'll update. Next, a tanh layer creates a vector of new candidate values, (	ilde{C}_t), that could be added to the state. In the next step, we'll combine these two to create an update to the state.

In the example of our language model, we'd want to add the gender of the new subject to the cell state, to replace the old one we're forgetting.

![LSTM3-focus-i](./images/LSTMs/LSTM3-focus-i.png){: style="width:90%; display: block; margin:8px auto"}

### Step 3: Updating the Cell State

It's now time to update the old cell state, (C_{t-1}), into the new cell state (C_t). The previous steps already decided what to do, we just need to actually do it.

We multiply the old state by (f_t), forgetting things we decided to forget earlier. Then we add (i_t*	ilde{C}_t). This is the new candidate values, scaled by how much we decided to update each state value.

In the case of the language model, this is where we'd actually drop the information about the old subject's gender and add the new information, as we decided in the previous steps.

![LSTM3-focus-C](./images/LSTMs/LSTM3-focus-C.png){: style="width:90%; display: block; margin:8px auto"}

### Step 4: Deciding What to Output

Finally, we need to decide what we're going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we're going to output. Then, we put the cell state through (	anh) (to push the values to be between (-1) and (1)) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.

For the language model example, since it just saw a subject, it might want to output information relevant to a verb, in case that's what is coming next. For example, it might output whether the subject is singular or plural, so that we know what form a verb should be conjugated into if that's what follows.

![LSTM3-focus-o](./images/LSTMs/LSTM3-focus-o.png){: style="width:90%; display: block; margin:8px auto"}

## Variants on Long Short Term Memory

What I've described so far is a pretty normal LSTM. But not all LSTMs are the same as the above. In fact, it seems like almost every paper involving LSTMs uses a slightly different version. The differences are minor, but it's worth mentioning some of them.

One popular LSTM variant, introduced by [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf), is adding "peephole connections." This means that we let the gate layers look at the cell state.

![LSTM3-var-peepholes](./images/LSTMs/LSTM3-var-peepholes.png){: style="width:90%; display: block; margin:8px auto"}

The above diagram adds peepholes to all the gates, but many papers will give some peepholes and not others.

Another variation is to use coupled forget and input gates. Instead of separately deciding what to forget and what we should add new information to, we make those decisions together. We only forget when we're going to input something in its place. We only input new values to the places we're going to forget.

![LSTM3-var-tied](./images/LSTMs/LSTM3-var-tied.png){: style="width:90%; display: block; margin:8px auto"}

A slightly more dramatic variation on the LSTM is the Gated Recurrent Unit, or GRU, introduced by [Cho, et al. (2014)](http://arxiv.org/pdf/1406.1078v3.pdf). It combines the forget and input gates into a single "update gate." It also merges the cell state and hidden state, and makes some other changes. The resulting model is simpler than standard LSTM models, and has been growing increasingly popular.

![LSTM3-var-GRU](./images/LSTMs/LSTM3-var-GRU.png){: style="width:90%; display: block; margin:8px auto"}

These are just a few of the most notable LSTM variants. There are lots of others, like Depth Gated RNNs by [Yao, et al. (2015)](http://arxiv.org/pdf/1508.03790v2.pdf). There are also completely different approaches to tackling long-term dependencies, like Clockwork RNNs by [Koutnik, et al. (2014)](http://arxiv.org/pdf/1402.3511v1.pdf).

Which of these variants is best? Do the differences matter? [Greff, et al. (2015)](http://arxiv.org/pdf/1503.04069.pdf) did a nice comparison of popular variants, finding that they're all about the same. [Jozefowicz, et al. (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) tested more than ten thousand RNN architectures, finding some that worked better than LSTMs on certain tasks.

## Conclusion

Earlier, I mentioned the remarkable results people are achieving with RNNs. Essentially all of these are achieved using LSTMs. They really work a lot better for most tasks!

Written as a set of equations, LSTMs look pretty intimidating. Hopefully, walking through them step by step in this essay has made them a bit more approachable.

LSTMs were a big step in what we can accomplish with RNNs. It's natural to wonder: is there another big step? A common opinion among researchers is: "Yes! There is a next step and it's attention!" The idea is to let every step of an RNN pick information to look at from some larger collection of information. For example, if you are using an RNN to create a description of an image, it might pick a part of the image to look at for every word it outputs. In fact, [Xu, et al. (2015)](http://arxiv.org/pdf/1502.03044v2.pdf) do exactly this – it might be a fun starting point if you want to explore attention! There's been a lot of exciting work with attention, and it seems to be an important idea for the future of deep learning.

Attention isn't the only exciting thread in RNN research. For example, Grid LSTMs by [Kalchbrenner, et al. (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) seem extremely promising. Work using RNNs in generative models – such as [Gregor, et al. (2015)](http://arxiv.org/pdf/1502.04623.pdf), [Chung, et al. (2015)](http://arxiv.org/pdf/1506.02216v3.pdf), or [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf) – also seems very interesting. The last few years have been an exciting time for recurrent neural networks, and the coming ones promise to only be more so!