# 浅尝 LSTM 网络

!!! info "文档来源"
    本文档改编自 Christopher Olah 的 [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)。
    
    原作者：[Christopher Olah](https://colah.github.io/)

## 循环神经网络 (Recurrent Neural Networks, RNN)

人类并不是每一秒都从零开始思考。当你阅读这篇文章时，你理解每个词的含义都是基于对前面词的理解。你不会丢掉所有内容然后重新开始思考。你的思维是持续的。

传统的神经网络无法做到这一点，这似乎是一个主要缺陷。例如，假设你想要对电影中每个时刻发生的事件类型进行分类。传统神经网络很难利用它对电影前面场景的推理来影响后面的判断。

循环神经网络解决了这个问题。它们是包含循环的网络，允许信息持续存在。

![RNN-rolled](./images/LSTMs/RNN-rolled.png){: style="width:11.8%; display: block; margin:17px auto 5px"}

**循环神经网络包含循环。**
{: style="text-align:center; margin-bottom:20px"}

在上图中，神经网络的一个模块 \(A\) 查看某个输入 \(x_t\) 并输出一个值 \(h_t\)。循环允许信息从网络的一个步骤传递到下一个步骤。

这些循环使得循环神经网络看起来有些神秘。然而，如果你仔细想想，就会发现它们其实与普通的神经网络没有太大区别。循环神经网络可以被看作是同一个网络的多个副本，每个副本都将消息传递给后继者。展开这个循环看看会发生什么：

![RNN-unrolled](./images/LSTMs/RNN-unrolled.png){: style="width:70%; display: block; margin:17px auto 5px"}

**展开的循环神经网络。**
{: style="text-align:center; margin-bottom:20px"}

这种链式的特性揭示了循环神经网络与序列和列表有着密切的关系。它们是神经网络处理这类数据的自然架构。

而且它们确实被广泛使用！在过去几年里，循环神经网络在各种问题上都取得了令人难以置信的成功：语音识别、语言建模、翻译、图像描述等等。这个列表还在继续增长。关于使用循环神经网络可以实现的惊人成就，我建议参考 Andrej Karpathy 的优秀博文 [《循环神经网络的不合理效能》](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。它们确实非常神奇。

这些成功的关键在于使用 "LSTM"，这是一种非常特殊的循环神经网络，在许多任务上都比标准版本效果要好得多。几乎所有基于循环神经网络的令人兴奋的成果都是通过它们实现的。这些 LSTM 也将是本文探讨的重点。

## 长期依赖问题

RNN 的一个吸引人的地方在于它们可能能够将先前的信息连接到当前任务，比如使用之前的视频帧可能有助于理解当前帧。如果 RNN 真能做到这一点，它们将会非常有用。但它们真的能做到吗？这要看情况。

有时我们只需要查看最近的信息来执行当前任务。例如，考虑一个语言模型试图根据前面的词来预测下一个词。如果我们试图预测"白云飘在蓝色的_天空_"中的最后一个词，我们不需要更多的上下文 —— 很明显下一个词是天空。在这种情况下，相关信息和需要使用这个信息的地方之间的距离很小，RNN 可以学会使用过去的信息。

![RNN-shorttermdepdencies](./images/LSTMs/RNN-shorttermdepdencies.png){: style="width:50%; display: block; margin:17px auto"}

但有时我们需要更多的上下文。考虑预测 "I grew up in France… I speak fluent ==French=={: style="color: red; font-style: italic;"}" 中的最后一个词。最近的信息表明下一个词可能是某种语言的名称，但如果我们想确定具体是哪种语言，我们需要更早之前提到法国的上下文。这种相关信息和需要使用它的位置之间的距离可能变得非常大。

不幸的是，随着这个距离的增加，RNN 就会变得无法学习连接这些信息。

![RNN-longtermdependencies](./images/LSTMs/RNN-longtermdependencies.png){: style="width:65.2%; display: block; margin:17px auto"}

理论上，RNN 完全有能力处理这种"长期依赖"问题。人类可以仔细选择参数来解决这种形式的玩具问题。但遗憾的是，在实践中，RNN 似乎无法学习这些知识。这个问题被 [Hochreiter (1991) [德语]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) 和 [Bengio 等人 (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf) 深入研究过，他们发现了一些相当根本性的原因来解释为什么这可能很困难。

幸运的是，LSTM 没有这个问题！

## 长短期记忆网络 (Long Short Term Memory networks, LSTM)

长短期记忆网络 —— 通常简称为 "LSTM" —— 是一种特殊的 RNN，能够学习长期依赖关系。它们由 [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf) 提出，并在后续工作中被许多人改进和推广[^1]。它们在各种各样的问题上都表现出色，现在被广泛使用。

LSTM 的设计明确旨在避免长期依赖问题。长时间记住信息实际上是它们的默认行为，而不是它们需要努力学习的东西！

所有循环神经网络都具有神经网络重复模块链的形式。在标准的 RNN 中，这个重复模块具有非常简单的结构，例如一个单独的 tanh 层。

![LSTM3-SimpleRNN](./images/LSTMs/LSTM3-SimpleRNN.png){: style="width:90%; display: block; margin:15px auto"}

**标准 RNN 中的重复模块包含单个层。**
{: style="text-align:center; margin-bottom:10px"}

LSTM 也有这种链式结构，但重复模块有着不同的结构。它不是只有一个神经网络层，而是有四个，并且以非常特殊的方式进行交互。

![LSTM3-chain](./images/LSTMs/LSTM3-chain.png){: style="width:90%; display: block; margin:15px auto"}

**LSTM 中的重复模块包含四个交互的层。**
{: style="text-align:center; margin-bottom:10px"}

先不要担心细节。我们稍后会一步步地解析 LSTM 图。现在，让我们先熟悉一下我们将要使用的符号。

![LSTM2-notation](./images/LSTMs/LSTM2-notation.png){: style="width:70%; display: block; margin:8px auto"}

在上图中，每条线都承载着一个完整的向量，从一个节点的输出到其他节点的输入。粉色圆圈表示逐点运算，如向量加法，而黄色方框则是学习到的神经网络层。线条的合并表示内容连接，而一条线的分叉表示其内容被复制并且副本分别送往不同的位置。

## LSTM 的核心思想

LSTM 的关键是细胞状态，即贯穿图表顶部的水平线。

细胞状态有点像传送带。它沿着整个链条直线运行，只有一些较小的线性交互。信息可以很容易地沿着它保持不变地流动。

![LSTM3-C-line](./images/LSTMs/LSTM3-C-line.png){: style="width:90%; display: block; margin:8px auto"}

LSTM 确实有能力从细胞状态中移除或添加信息，这个过程由称为"门"的结构来精确调节。

门是一种选择性地让信息通过的方式。它们由一个 sigmoid 神经网络层和一个逐点乘法操作组成。

![LSTM3-gate](./images/LSTMs/LSTM3-gate.png){: style="width:12%; display: block; margin:8px auto"}

sigmoid 层输出 0 到 1 之间的数字，描述应该让每个组件通过多少。输出 0 表示"不让任何信息通过"，而输出 1 表示"让所有信息通过"！

LSTM 有三个这样的门，用来保护和控制细胞状态。

## LSTM 的步骤分解

### 第一步：决定要丢弃的信息

我们的 LSTM 的第一步是决定我们要从细胞状态中丢弃什么信息。这个决定由一个称为 "遗忘门层"的 sigmoid 层做出。它查看 \(h_{t-1}\) 和 \(x_t\)，然后为细胞状态 \(C_{t-1}\) 中的每个数字输出一个 0 到 1 之间的数字。1 表示 "完全保留这个"，而 0 表示 "完全舍弃这个"。

让我们回到语言模型的例子，试图根据之前的所有单词预测下一个单词。在这样的问题中，细胞状态可能包括当前主语的性别，这样就可以使用正确的代词。当我们看到新的主语时，我们需要忘记旧主语的性别。

![LSTM3-focus-f](./images/LSTMs/LSTM3-focus-f.png){: style="width:90%; display: block; margin:8px auto"}

### 第二步：决定要存储的新信息

下一步是决定我们要在细胞状态中存储什么新信息。这包含两个部分。首先，一个称为"输入门层"的 sigmoid 层决定我们要更新哪些值。然后，一个 tanh 层创建一个新的候选值向量 \(\tilde{C}_t\)，这些值可能会被添加到状态中。在下一步中，我们将把这两部分结合起来，创建对状态的更新。

在我们的语言模型例子中，我们想要将新主语的性别添加到细胞状态中，以替换我们要忘记的旧主语。

![LSTM3-focus-i](./images/LSTMs/LSTM3-focus-i.png){: style="width:90%; display: block; margin:8px auto"}

### 第三步：更新细胞状态

现在是将旧细胞状态 \(C_{t-1}\) 更新为新的细胞状态 \(C_t\) 的时候了。前面的步骤已经决定了要做什么，我们现在就来执行。

我们将旧状态乘以 \(f_t\)，遗忘我们之前决定遗忘的内容。然后我们加上 \(i_t*\tilde{C}_t\)。这是新的候选值，根据我们决定更新每个状态值的程度进行缩放。

对于语言模型来说，这就是我们实际丢弃旧主语性别信息并添加新信息的地方，正如我们在前面步骤中决定的那样。

![LSTM3-focus-C](./images/LSTMs/LSTM3-focus-C.png){: style="width:90%; display: block; margin:8px auto"}

### 第四步：决定输出内容

最后，我们需要决定要输出什么。这个输出将基于我们的细胞状态，但会是一个过滤后的版本。首先，我们运行一个 sigmoid 层来决定细胞状态的哪些部分我们要输出。然后，我们将细胞状态通过 \(\tanh\)（把值压缩到 \(-1\) 和 \(1\) 之间）并将其与 sigmoid 门的输出相乘，这样我们就只输出我们决定输出的部分。

对于语言模型的例子，由于它刚刚看到了一个主语，它可能需要输出与动词相关的信息，以防这就是接下来要出现的内容。例如，它可能输出主语是单数还是复数，这样我们就知道如果接下来是动词的话应该采用什么形式的变位。

![LSTM3-focus-o](./images/LSTMs/LSTM3-focus-o.png){: style="width:90%; display: block; margin:8px auto"}

## LSTM 的变体

到目前为止我描述的是一个相当标准的 LSTM。但并非所有 LSTM 都与上述完全相同。实际上，似乎每篇涉及 LSTM 的论文都使用着稍微不同的版本。这些差异很小，但值得一提。

一个流行的 LSTM 变体，由 [Gers & Schmidhuber (2000)](ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf) 引入，增加了"窥视孔连接"（peephole connections）。这意味着我们让门层可以查看细胞状态。

![LSTM3-var-peepholes](./images/LSTMs/LSTM3-var-peepholes.png){: style="width:90%; display: block; margin:8px auto"}

上图展示了将窥视孔添加到所有门，但许多论文会选择性地只在某些门添加窥视孔。

另一个变体是使用耦合的遗忘和输入门。我们不是分别决定要遗忘什么和要添加什么新信息，而是一起做这些决定。我们只在要输入新内容的地方进行遗忘。我们只在遗忘旧内容时向状态添加新值。

![LSTM3-var-tied](./images/LSTMs/LSTM3-var-tied.png){: style="width:90%; display: block; margin:8px auto"}

一个稍微更显著的 LSTM 变体是门控循环单元（Gated Recurrent Unit，GRU），由 [Cho 等人 (2014)](http://arxiv.org/pdf/1406.1078v3.pdf) 提出。它将遗忘门和输入门合并成一个单一的"更新门"。它还合并了细胞状态和隐藏状态，并做了一些其他改变。最终的模型比标准的 LSTM 模型更简单，并且越来越受欢迎。

![LSTM3-var-GRU](./images/LSTMs/LSTM3-var-GRU.png){: style="width:90%; display: block; margin:8px auto"}

这些只是一些最著名的 LSTM 变体。还有很多其他变体，比如 [Yao 等人 (2015)](http://arxiv.org/pdf/1508.03790v2.pdf) 的深度门控 RNN。还有一些完全不同的方法来处理长期依赖问题，比如 [Koutnik 等人 (2014)](http://arxiv.org/pdf/1402.3511v1.pdf) 的 Clockwork RNN。

这些变体中哪个最好？这些差异重要吗？[Greff 等人 (2015)](http://arxiv.org/pdf/1503.04069.pdf) 对流行的变体进行了很好的比较，发现它们都差不多。[Jozefowicz 等人 (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 测试了超过一万种 RNN 架构，发现在某些任务上有一些架构比 LSTM 效果更好。

## 结论

前面我提到了人们使用 RNN 取得的显著成果。本质上所有这些成果都是使用 LSTM 实现的。对于大多数任务来说，LSTM 确实效果好得多！

用一组方程式写出来，LSTM 看起来相当令人生畏。希望通过本文一步一步地解析它们，能让它们变得更容易理解。

LSTM 是我们在 RNN 领域能够实现的一个重大进步。很自然地会想：是否还有下一个重大进步？研究人员中的一个普遍观点是："是的！下一步是注意力机制！" 这个想法是让 RNN 的每一步都能从更大的信息集合中选择信息来查看。例如，如果你正在使用 RNN 为图像创建描述，它可能会为它输出的每个词选择图像的一部分来关注。实际上，[Xu 等人 (2015)](http://arxiv.org/pdf/1502.03044v2.pdf) 就是这么做的 - 如果你想探索注意力机制，这可能是一个有趣的起点！已经有许多使用注意力机制的令人兴奋的成果，似乎还会有更多突破...

注意力机制并不是 RNN 研究中唯一令人兴奋的线索。例如，[Kalchbrenner 等人 (2015)](http://arxiv.org/pdf/1507.01526v1.pdf) 的 Grid LSTM 看起来非常有前途。在生成模型中使用 RNN 的工作 - 例如 [Gregor 等人 (2015)](http://arxiv.org/pdf/1502.04623.pdf)、[Chung 等人 (2015)](http://arxiv.org/pdf/1506.02216v3.pdf) 或 [Bayer & Osendorfer (2015)](http://arxiv.org/pdf/1411.7610v3.pdf) 的工作 - 也非常有趣。过去几年对循环神经网络来说是激动人心的时期，而未来只会更加精彩！