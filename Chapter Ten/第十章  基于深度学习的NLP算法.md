# 第十章 基于深度学习的NLP算法
## 深度学习概述
1.前一章我们介绍了NLP算法的基于统计学的机器学习方法体系，这里将继续介绍NLP算法的第二个方法体系：基于人工神经网络(Artificial Neural Network)的深度学习方法。  
2.神经网络是具有自适应的简单单元组成的，广泛的、并行的、互联的网络，它的结构模拟了生物神经网络系统对真实世界所做出的的交互反应。  
3.由于人工神经网络可以对非线性过程进行建模，因此可以解决例如分类、聚类、回归、降维、结构化预测等一系列复杂的问题。  
4.深度学习作为机器学习的一个重要分支，可以自动地学习合适的特征与多层次的表达与输出，在NLP领域，主要是在信息抽取、命名实体识别，机器翻译，市场营销、金融领域的情感分析，问答系统，搜索引擎，推荐系统等方向都有成功的应用。  
5.和传统方式相比，深度学习的重要特性是，用词向量来表示各种级别的元素，传统的算法一般会用统计等方法去标注，而深度学习会直接通过词向量表示，然后通过深度学习网络进行自动学习。
### 神经元模型
1.多层感知机，神经网络中最基本的是神经元模型。在生物神经元中，每个神经元与其他神经元相连，当它处于激活状态时，就会向相连的神经元发送化学信号，从而改变其他神经元的状态，如果某个神经元的电量超过某个阈值，那么将会被激活，再接着发送给其他神经元。
### 激活函数
1.理想的激活函数是跃迁函数，它输入值映射到0或1，显然1对应着神经元激活状态，0则表示神经元处于抑制状态。然而由于跃迁函数不连续且非光滑(无法完美表达大脑神经网络的连续传递过程)，因此实际常用Sigmoid函数作为激活函数。典型的Sigmoid函数把可能的数压缩进(0, 1)输出值之间，因此又名挤压函数(squashing function)。
### 感知机与多层网络
1.两层结构的感知机：输入层接收外界的输入信号然后传递给输出层，输出层为逻辑单元，感知机的输入是几个二进制，输出是一位单独的二进制。  
2.第一层感知机，通过赋予输入的权重，做出三个非常简单的决策。第二层感知机，每一个第二层感知机通过赋予权重给来自第一层感知机的决策结果做出决策。通过这种方式，第二层感知机可以比第一层感知机做出更加复杂以及更高层次抽象的决策。第三层感知机能够做出更加复杂的决策。  
3.目前我们讨论的神经网络，都是前面一层作为后面一层的输入，这种经典的网络被称为前馈神经网络。这就意味着网络中没有回路，信息总是向前传播，从不反向回馈。  
4.整个神经网络的算法框架：  
(1) 训练阶段(training)：是指网络输入样本数据作为初始数据，通过激活函数与网络连接，迭代求得最小化损失。这时网络最终收敛，学习得到权重向量，作为分类器的参数。数学上称这个过程为参数估计的过程。在NLP中如果用于序列标注，则可以称为一个标注器。  
(2) 推导阶段(infer)：拿这个训练好的网络对实际的数据进行分类或回归，称为分类阶段。
## 神经网络模型
1.所谓神经网络就是将很多个单一的神经单元组合到一起，这样，一个神经单元的输出就可以是另一个神经单元的输入。
## 多层输出模型
1.前面一节，我们讨论了一种通用的人工神经网络结构，同时，我们也可以构建另一种结构的神经网络(这里的结构指的是两个神经元的连接方式)，即含有多个隐藏层的神经网络。
## 反向传播算法
1.多层网络的学习能力比单层网络强大得多。想要训练多层网络，前面的简单感知机学习方法显然还不足，需要更加强大的算法。反向传播算法(Back Propagation, BP)其中的经典方法，它是现今最成功的神经网络算法，现实当中使用到神经网络时，大多是使用BP算法训练的。BP算法不仅可以用于多层前馈神经网络，还可以用于其他类型神经网络，例如LSTM。当然，通常所说的BP网络，一般是用BP算法训练的多层前馈网络。
## 最优化算法
1.机器学习完成一个训练任务有三个要素：算法模型、目标函数、优化算法。优化机器学习问题的求解，本质上都是优化问题，最常见的求解方式就是迭代优化。
### 梯度下降
1.优化的目标是损失函数最小化，从优化的角度看，函数的梯度方向代表了函数值增长最快的方向。  
2.梯度下降方法，就是其中最直接的一种方法，直接计算整个训练集所有样本的Loss，每次在全集上求梯度下降方向。  
3.梯度下降，这个方法的好处就是能保证每次都朝着正确的方向前进，最后能收敛于极值点(凸函数收敛于全局极值点，非凸函数可能会收敛于局部极值点)。但它的问题也显而易见，数据集很大的时候(通常都很大)，每次迭代的时间内都很长，而且需要超大的内存空间。
### 随机梯度下降
1.有人在梯度下降方法的基础上做了创新，称为随机梯度下降(Stochastic Gradient Descent)算法，这样做的好处收敛速度更快。但也带来了副作用：由于每次都只是取一个样本，没有全局性，所以不能保证每次更新都是朝着正确的方向前进。
### 批量梯度下降
1.由于上述两种方法的弊端，后来有人在这两个方案中取折中，即批量梯度下降(Mini-Batch Gradient Descent)算法。  
2.跟原始的梯度下降算法比，批量梯度下降算法提高了学习速度，降低了内存开销；跟随机梯度比，它抑制了样本的随机性，降低了收敛的波动性，参数更新更加稳定。所以他是现在接受度最广的方案。
### 丢弃法
1.前面我们介绍了优化方法，也提到了正则法来应对过拟合问题。在深度学习当中，还有个非常常用的方法：丢弃法(Dropout)。  
2.丢弃法比较容易理解，在现代神经网络中，我们所说的丢弃法，通常是对输入层或者隐藏层做如下操作：  
(1) 随机选择部分该层的输出作为丢弃元素  
(2) 把想要丢弃的元素乘以0  
(3) 把非丢弃元素拉伸。  
3.一般情况下，隐含节点Dropout率等于0.5的时候效果最好，原因是0.5的时候Dropout随机生成的网络结构最多。
## 激活函数
### tanh函数
1.相较于Sigmoid函数，它比Sigmoid函数收敛速度更快。而且，其输出以0为中心。但是，它也存在与Sigmoid函数相同的问题-由于饱和性产生的梯度消失。
### ReLU函数
1.ReLU(Rectified Linear Unit, 规划线性单元)是近年来被大量使用的激活函数。  
2.相比起Sigmoid和tanh，ReLU在SGD中能够快速收敛。深度学习中最大的问题是梯度消失问题(Gradient Vanishing Problem)，这在使用Sigmoid、tanh等饱和激活函数情况下特别严重(神经网络在进行方向误差传播时，各个层都要乘以激活函数的一阶倒数，梯度每传递一层都会衰减一层，网络层数较多时，梯度就会不停衰减直到消失)，使得训练网络收敛越来越慢，而ReLU凭借其线性、非饱和的形式，训练速度则是快乐很多。
## 实现BP算法
1.载入数据(10_minst_loader.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
```
2.实现算法(10_bp.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

import numpy as np
import random


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        """return a tuple
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # 存放激活值

        zs = []  # list用来存放z 向量

        # 前向传递
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # 后向传递
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
            return the number of test inputs for which is correct
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self, z):
        """sigmoid函数"""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        """求导"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def feedforward(self, a):
        """
            Return the output of the network if "a " is input
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)

        return a

    def update_mini_batch(self, mini_batch, eta):
        """
            update the networks' weights and biases by applying gradient descent using
            bp to a single mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) *
                        nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) *
                       nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent, the "training_data" is a list of tuples
        (x,y) representing the training inputs and the desired outputs.
        the other non-optional params are self-explanatory
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)  # 50000
        for j in xrange(epochs):  # epochs迭代
            random.shuffle(training_data)  # 打散
            mini_batches = [           # 10个数据一次迭代:mini_batch_size,以 mini_batch_size为步长
                training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:  # 分成很多分mini_batch进行更新
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}:{1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

if __name__ == "__main__":
    nn = Network([3, 4, 1])
    a = [k for k in xrange(0, 500, 50)]
    print(a)

    print([np.zeros(b.shape) for b in nn.biases])
    activation = np.random.randn(3, 1)
    activations = [activation]
    zs = []
    for b, w in zip(nn.biases, nn.weights):
        z = np.dot(w, activation) + b
        print(z)
        zs.append(z)
        activation = nn.sigmoid(z)
        print(activation)
        activations.append(activation)
    print("zs", zs)
    print("activ", activations)
```
3.模型训练与结果输出(10_main.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

import bp
import mnist_loader

net = bp.Network([784, 100, 10])


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```
## 词嵌入算法
1.基于神经网络的表示一般称为词向量，词嵌入(wording embeding)或分布式表示(distributed repressiontation)。神经网络词向量与其他分布式方式类似，都基于分布式表达方式，核心仍然是上下文的表示以及上下文与目标词之间的的关系映射，主要通过神经网络对上下文，以及上下文和目标词汇之间的关系进行建模。
### 词向量
1.词表示方法中的One-hot表达这种方法把每个词顺序编号，每个词就是一个很长的向量，向量的维度等于词表的大小，只有对应位置上的数字为1，其他都为0。  
2.分布式表示基本思想是通过训练将每个词映射成K维实数向量(K一般为模型中的超参数)，通过词之间的距离(比如cosine相似度、欧氏距离等)来判断它们之间的语义相似度。
### word2vec简介
1.word2vec以及其他词向量模型，都基于了同样的假设：衡量词语之间的相似性，在于其相邻词汇是否相识，这是基于语言学的“距离象似性”原理。词汇和它上下文构成了一个象，当从语料库中学习到相似或者相近的象时，他们在语义上总是相似的。  
2.word2vec是一款将词表征为实数值向量的高效工具，采用的模型有CBOW(Continuous Bag-of-Words, 连续的词袋模型)和Skip-Gram两种。  
3.word2vec通过训练，可以把对文本内容的处理简化为K维向量空间的向量运算，而向量空间上的相似度可以用来表示文本语义上的相似度。因此，word2vec输出的词向量可以被用来做很多NLP相关的工作，比如聚类、找同义词、词性分析等。  
4.词向量的评价大致分为两种方式：一种把词向量融入系统当中，提升整个系统的准确度，另外一种是从语言学的角度上对词向量进行分析，例如橘子相似度分析、语义偏移等
### 词向量模型
1.word2vec算法的主要流程：首先对每个词都关联一个特征向量；然后，使用特征向量表示词组序列的概率函数；最后，利用词组数据来学习特征向量和概率函数的参数。第一步比较简单，对每个词，我们随机初始化一个特征向量。第二步，主要是设计神经网络。第三步，通过数据训练神经网络，得到合理的特征向量和神经网络参数。先用前向传播计算输出，然后用BP算法求导计算。
### CBOW和Skip-gram模型
1.2013年开源的工具包word2vec包含了CBOW(Continuous Bag-Of-Words Model)和Skip-gram这两个得到词向量为目标的模型。  
2.CBOW包含了输入层、投影层、以及输出层(没有隐藏层)。其基本运算流程如下：  
(1) 随机生成一个所有单词的词向量矩阵，每一行对应一个单词的向量。  
(2) 对于某个单词(中心词)，从矩阵中提取其周边单词的词向量。  
(3) 求周边单词的词向量的均值向量  
(4) 在该均值向量上使用logistic regression进行训练，softmax作为激活函数。  
(5) 期望回归得到的概率向量可以与真实的概率向量(即中心词one-hot编码向量)相匹配。  
3.CBOW是使用周边的单词去预测该单词，而Skip-Gram正好相反，是输入当前词的词向量，输出周围词的词向量。  
## 训练词向量实践
### 实战用Gensim训练百科语料库
1.数据预处理
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

from gensim.corpora import WikiCorpus

space = " "

with open('wiki-zh-article.txt', 'w', encoding = 'utf8') as f:
    wiki = WikiCorpus('zhwiki-latest-pages-articles.xml.bz2', lemmatize = False, dictionary = {})
    for text in wiki.get_texts():
        f.write(space.join(text) + '\n')

print("Finished Saved")
```
2.繁体字处理  
因为维基语料库包含了繁体字和简体字，为了不影响后续分词，所以统一转化为简体字。这时要用到opencc(https://github.com/BYVoid/OpenCC)，给出如下命令：
```
opencc -i corpus.txt -o wiki-corpus.txt -c t2s.json
```
3.分词
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

import codecs
import jieba

infile = 'wiki-zh-article-zhs.txt'
outfile = 'wiki-zh-words.txt'

descsFile = codecs.open(infile, 'rb', encoding = 'utf-8')
i = 0
with open(outfile, 'w', encoding = 'utf-8') as f:
    for line in descsFile:
        i += 1
        if i % 1000 == 0:
            print(i)
        line = line.strip()
        words = jieba.cut(line)
        for word in words:
            f.write(word + ' ')
        f.write('\n')
```
4.运行word2vec训练
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

import multiprocessing

from gensim.models import Word2vec
from gensim.models import LineSentence

inp = 'wiki-zh-words.txt'
outp1 = 'wiki-zh-model'
outp2 = 'wiki-zh-vector'

model = Word2vec(LineSentence(inp), size = 400, window = 5, min_count = 5, workers = multiprocessing.cpu_count())
model.save(outp1)
model.save_word2vec_format(outp2, binary = False)

model = Word2vec.load('./wiki-zh-model')
# model = Word2vec.load_word2vec_format('./wiki-zh-vector', binary = False) # 如果之前用文本保存的话，用这个方法加载
res = model.most_similar('时间')
print(res)
```
## 朴素Vanilla-RNN
1.前面提到的关于NLP的几个应用，例如分类、聚类，都是为考虑到词的序列信息。而针对序列化学习，循环神经网络(Recurrent Neural Networks, RNN)则能够通过在原有神经网络基础上增加记忆单元，处理任意长度的序列(理论上)，在架构上比一般神经网络更加能够处理序列相关的问题，因此，这也是为了解决这类问题而设计的一种网络结构。  
2.RNN已经被成功应用到各种应用中：语音识别、机器翻译、图像标注等。  
3.RNN取得各项成功的一个关键模型是RNN的变体长短时记忆网络(Long Short Term Memory Networks, LSTM)。
## LSTM网络
### LSTM基本结构
1.长短时记忆网络(Long Short Term Memory Networks, LSTM)是一种特殊的RNN，它能够学习长时间依赖。
### 其他LSTM变种形式
1.之前描述的都是通用的LSTM。还有很多其他形式得LSTM形式，很多LSTM之间有细微差别，其中流行的LSTM变化形式是由Gers和Schmidhuber于2000年提出的，增加了窥视孔连接(peephole connection)。
## Attention
1.Attention机制的基本思想是，打破了传统编码器-解码器结构在编解码时都依赖于内部一个固定长度向量的限制。  
2.Attention机制的实现是通过保留LSTM编码器输入序列的中间输出结果，然后训练一个模型来对这些输入进行选择性学习，并且在模型输出时将输出序列与之进行关联。  
### 文本翻译
1.文本翻译，当给定一个法语句子的输入序列，将它翻译并输出英文句子。注意力机制用于观察输入序列中与输出序列每一个词相对应的具体单词。
### 图说模型
1.基于序列的注意力可以应用在计算机视觉问题上，来帮助找出方法，使输出序列更好地利用卷积神经网络来关注输入的图片。
### 语义蕴涵
1.给定一个前提场景，并且用英文给出关于该场景的假设，输出内容是前提和假设是否矛盾、二者是否相互关联、或者前提是否蕴含假设。
### 语音识别
1.给定一个英文语音片段作为输入，输出一个音素序列。注意力机制被用来关联输出序列中的每一个音素和输入序列中特定的语音帧。
### 文本摘要
1.给定一段文章作为输入序列，输出一段文本来总结输入序列。注意力机制被用来关联摘要文本中的每一个词语与原文本中的对应单词。
### Seq2Seq模型
1.对于一些NLP任务，比如聊天机器人、机器翻译、自动文摘等，传统的方法都是从候选集中选出答案，这对候选集的完善程度要求很高。Encoder-Decoder是近两年来在NLG和NLU方面应用较多的方法。
## 实战Seq2Seq问答机器人
1.数据载入与预处理(10_preprocessing.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

import jieba


class Processor():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']

    def __init__(self):
        self.encoderFile = "./Q.txt"
        self.decoderFile = './A.txt'
        self.stopwordsFile = "./tf_data/stopwords.dat"

    def wordToVocabulary(self, originFile, vocabFile, segementFile):
        # stopwords = [i.strip() for i in open(self.stopwordsFile).readlines()]
        # print(stopwords)
        # exit()
        vocabulary = []
        sege = open(segementFile, "w")
        with open(originFile, 'r') as en:
            for sent in en.readlines():
                # 去标点
                if "enc" in segementFile:
                    sentence = sent.strip()
                    words = jieba.lcut(sentence)
                    print(words)
                else:
                    words = jieba.lcut(sent.strip())
                vocabulary.extend(words)
                for word in words:
                    sege.write(word + " ")
                sege.write("\n")
        sege.close()

        # 去重并存入词典
        vocab_file = open(vocabFile, "w")
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=vocabulary.index)
        _vocabulary = self.vocab + _vocabulary
        for index, word in enumerate(_vocabulary):
            vocab_file.write(word + "\n")
        vocab_file.close()

    def toVec(self, segementFile, vocabFile, doneFile):
        word_dicts = {}
        vec = []
        with open(vocabFile, "r") as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(doneFile, "w")
        with open(segementFile, "r") as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word)) + " ")
                f.write("\n")
        f.close()

    def run(self):
        # 获得字典
        self.wordToVocabulary(
            self.encoderFile, './tf_data/enc.vocab', './tf_data/enc.segement')
        self.wordToVocabulary(
            self.decoderFile, './tf_data/dec.vocab', './tf_data/dec.segement')
        # 转向量
        self.toVec("./tf_data/enc.segement",
                   "./tf_data/enc.vocab",
                   "./tf_data/enc.vec")
        self.toVec("./tf_data/dec.segement",
                   "./tf_data/dec.vocab",
                   "./tf_data/dec.vec")


process = Processor()
process.run()
```
2.模型处理(10_dynamic_seq2seq_model.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple


class dynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.0.0

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
        time_major              控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]


    '''
    PAD = 0
    EOS = 2
    UNK = 3

    def __init__(self, encoder_cell,
                 decoder_cell,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False,
                 time_major=False):

        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size

        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5
        self.time_major = time_major

        # 创建模型
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # 兼容decoder输出数据
        self._init_decoder_train_connectors()

        # embedding层
        self._init_embeddings()

        # 判断是否为双向LSTM并创建encoder
        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        # self.encoder_inputs = tf.Variable(np.ones((10, 50)).astype(np.int32))
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(
                tf.shape(self.decoder_targets))
            # batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat(
                [EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1
            # self.decoder_train_length = self.decoder_targets_length

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])

            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.encoder_embedding_matrix = tf.get_variable(
                name="encoder_embedding_matrix",
                shape=[self.encoder_vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.decoder_embedding_matrix = tf.get_variable(
                name="decoder_embedding_matrix",
                shape=[self.decoder_vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            # encoder的embedd
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.encoder_embedding_matrix, self.encoder_inputs)

            # decoder的embedd
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.decoder_embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        '''
        一般的encdoer
        '''
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=self.time_major,
                                  dtype=tf.float32)
            )

    def _init_bidirectional_encoder(self):
        '''
        双向LSTM encoder
        '''
        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=self.time_major,
                                                dtype=tf.float32)
            )

            self.encoder_outputs = tf.concat(
                (encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(
                    c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat(
                    (encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                self.test_outputs = outputs
                return tf.contrib.layers.linear(outputs, self.decoder_vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(
                    encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(
                        self.encoder_inputs_length) + 100,
                    num_decoder_symbols=self.decoder_vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(
                    self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                 attention_values,
                 attention_score_fn,
                 attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(
                        self.encoder_inputs_length) + 100,
                    num_decoder_symbols=self.decoder_vocab_size,
                )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=self.time_major,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=self.time_major,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(
                self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_MMI(self, logits, targets):
        sum_mmi = 0
        x_value_list = 1

    def _init_optimizer(self):
        # 整理输出并计算loss
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        self.targets = tf.transpose(self.decoder_train_targets, [1, 0])

        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)

        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss)

        # add
        params = tf.trainable_variables()
        self.gradient_norms = []
        self.updates = []

        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         self.max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
```
3.模型输出和结果(10_main.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-25 18:13:58
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-25 18:19:07
'''

import numpy as np
import time
import sys
import os
import re
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from dynamic_seq2seq_model import dynamicSeq2seq
import jieba


class Seq2seq():
    '''
        params:
        encoder_vec_file    encoder向量文件
        decoder_vec_file    decoder向量文件
        encoder_vocabulary  encoder词典
        decoder_vocabulary  decoder词典
        model_path          模型目录
        batch_size          批处理数
        sample_num          总样本数
        max_batches         最大迭代次数
        show_epoch          保存模型步长
    '''

    def __init__(self):
        tf.reset_default_graph()

        self.encoder_vec_file = "./tfdata/enc.vec"
        self.decoder_vec_file = "./tfdata/dec.vec"
        self.encoder_vocabulary = "./tfdata/enc.vocab"
        self.decoder_vocabulary = "./tfdata/dec.vocab"
        self.batch_size = 1
        self.max_batches = 100000
        self.show_epoch = 100
        self.model_path = './model/'

        self.model = dynamicSeq2seq(encoder_cell=LSTMCell(40),
                                    decoder_cell=LSTMCell(40),
                                    encoder_vocab_size=600,
                                    decoder_vocab_size=1600,
                                    embedding_size=20,
                                    attention=False,
                                    bidirectional=False,
                                    debug=False,
                                    time_major=True)
        self.location = ["杭州", "重庆", "上海", "北京"]
        self.dec_vocab = {}
        self.enc_vocab = {}
        self.dec_vecToSeg = {}
        tag_location = ''
        with open(self.encoder_vocabulary, "r") as enc_vocab_file:
            for index, word in enumerate(enc_vocab_file.readlines()):
                self.enc_vocab[word.strip()] = index
        with open(self.decoder_vocabulary, "r") as dec_vocab_file:
            for index, word in enumerate(dec_vocab_file.readlines()):
                self.dec_vecToSeg[index] = word.strip()
                self.dec_vocab[word.strip()] = index

    def data_set(self, file):
        _ids = []
        with open(file, "r") as fw:
            line = fw.readline()
            while line:
                sequence = [int(i) for i in line.split()]
                _ids.append(sequence)
                line = fw.readline()
        return _ids

    def data_iter(self, train_src, train_targets, batches, sample_num):
        ''' 获取batch
            最大长度为每个batch中句子的最大长度
            并将数据作转换:
            [batch_size, time_steps] -> [time_steps, batch_size]

        '''
        batch_inputs = []
        batch_targets = []
        batch_inputs_length = []
        batch_targets_length = []

        # 随机样本
        shuffle = np.random.randint(0, sample_num, batches)
        en_max_seq_length = max([len(train_src[i]) for i in shuffle])
        de_max_seq_length = max([len(train_targets[i]) for i in shuffle])

        for index in shuffle:
            _en = train_src[index]
            inputs_batch_major = np.zeros(
                shape=[en_max_seq_length], dtype=np.int32)  # == PAD
            for seq in range(len(_en)):
                inputs_batch_major[seq] = _en[seq]
            batch_inputs.append(inputs_batch_major)
            batch_inputs_length.append(len(_en))

            _de = train_targets[index]
            inputs_batch_major = np.zeros(
                shape=[de_max_seq_length], dtype=np.int32)  # == PAD
            for seq in range(len(_de)):
                inputs_batch_major[seq] = _de[seq]
            batch_targets.append(inputs_batch_major)
            batch_targets_length.append(len(_de))

        batch_inputs = np.array(batch_inputs).swapaxes(0, 1)
        batch_targets = np.array(batch_targets).swapaxes(0, 1)

        return {self.model.encoder_inputs: batch_inputs,
                self.model.encoder_inputs_length: batch_inputs_length,
                self.model.decoder_targets: batch_targets,
                self.model.decoder_targets_length: batch_targets_length, }

    def train(self):
        # 获取输入输出
        train_src = self.data_set(self.encoder_vec_file)
        train_targets = self.data_set(self.decoder_vec_file)

        f = open(self.encoder_vec_file)
        self.sample_num = len(f.readlines())
        f.close()
        print("样本数量%s" % self.sample_num)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # 初始化变量
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            loss_track = []
            total_time = 0
            for batch in range(self.max_batches + 1):
                # 获取fd [time_steps, batch_size]
                start = time.time()
                fd = self.data_iter(train_src,
                                    train_targets,
                                    self.batch_size,
                                    self.sample_num)
                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)

                stop = time.time()
                total_time += (stop - start)

                loss_track.append(loss)
                if batch == 0 or batch % self.show_epoch == 0:

                    print("-" * 50)
                    print("n_epoch {}".format(sess.run(self.model.global_step)))
                    print('  minibatch loss: {}'.format(
                        sess.run(self.model.loss, fd)))
                    print('  per-time: %s' % (total_time / self.show_epoch))
                    checkpoint_path = self.model_path + "nlp_chat.ckpt"
                    # 保存模型
                    self.model.saver.save(
                        sess, checkpoint_path, global_step=self.model.global_step)

                    # 清理模型
                    self.clearModel()
                    total_time = 0
                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[self.model.decoder_targets].T,
                            sess.run(self.model.decoder_prediction_train, fd).T
                    )):
                        print('  sample {}:'.format(i + 1))
                        print('    dec targets > {}'.format(e_in))
                        print('    dec predict > {}'.format(dt_pred))
                        if i >= 0:
                            break

    def add_to_file(self, strs, file):
        with open(file, "a") as f:
            f.write(strs + "\n")

    def add_voc(self, word, kind):
        if kind == 'enc':
            self.add_to_file(word, self.encoder_vocabulary)
            index = max(self.enc_vocab.values()) + 1
            self.enc_vocab[word] = index
        else:
            self.add_to_file(word, self.decoder_vocabulary)
            index = max(self.dec_vocab.values()) + 1
            self.dec_vocab[word] = index
            self.dec_vecToSeg[index] = word
        return index

    def onlinelearning(self, input_strs, target_strs):
        input_seg = jieba.lcut(input_strs)
        target_seg = jieba.lcut(target_strs)

        input_vec = []
        for word in input_seg:
            if word not in self.enc_vocab.keys():
                vec = self.add_voc(word, "enc")
            else:
                vec = self.enc_vocab.get(word)
            input_vec.append(vec)

        target_vec = []
        for word in target_seg:
            if word not in self.dec_vocab.keys():
                vec = self.add_voc(word, "dec")
            else:
                vec = self.dec_vocab.get(word)
            target_vec.append(vec)

        with tf.Session() as sess:
            # 初始化变量
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            fd = self.data_iter([input_vec], [target_vec], 1, 1)
            for i in range(100):
                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)
                checkpoint_path = self.model_path + "nlp_chat.ckpt"
                # 保存模型
                self.model.saver.save(
                    sess, checkpoint_path, global_step=self.model.global_step)

                for i, (e_in, dt_pred) in enumerate(zip(
                        fd[self.model.decoder_targets].T,
                        sess.run(self.model.decoder_prediction_train, fd).T
                )):
                    print('    sample {}:'.format(i + 1))
                    print('    dec targets > {}'.format(e_in))
                    print('    dec predict > {}'.format(dt_pred))
                    if i >= 0:
                        break

    def segement(self, strs):
        return jieba.lcut(strs)

    def make_inference_fd(self, inputs_seq):
        sequence_lengths = [len(seq) for seq in inputs_seq]
        max_seq_length = max(sequence_lengths)

        inputs_time_major = []
        for sents in inputs_seq:
            inputs_batch_major = np.zeros(
                shape=[max_seq_length], dtype=np.int32)  # == PAD
            for index in range(len(sents)):
                inputs_batch_major[index] = sents[index]
            inputs_time_major.append(inputs_batch_major)

        inputs_time_major = np.array(inputs_time_major).swapaxes(0, 1)
        return {self.model.encoder_inputs: inputs_time_major,
                self.model.encoder_inputs_length: sequence_lengths}

    def predict(self):
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                print(ckpt.model_checkpoint_path)
                self.model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("没找到模型")

            action = False
            while True:
                if not action:
                    inputs_strs = input("me > ")
                if not inputs_strs:
                    continue

                inputs_strs = re.sub(
                    "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", inputs_strs)

                action = False
                segements = self.segement(inputs_strs)
                # inputs_vec = [enc_vocab.get(i) for i in segements]
                inputs_vec = []
                for i in segements:
                    inputs_vec.append(self.enc_vocab.get(i, self.model.UNK))
                fd = self.make_inference_fd([inputs_vec])
                inf_out = sess.run(self.model.decoder_prediction_inference, fd)
                inf_out = [i[0] for i in inf_out]

                outstrs = ''
                for vec in inf_out:
                    if vec == self.model.EOS:
                        break
                    outstrs += self.dec_vecToSeg.get(vec, self.model.UNK)
                print(outstrs)

    def clearModel(self, remain=3):
        try:
            filelists = os.listdir(self.model_path)
            re_batch = re.compile(r"nlp_chat.ckpt-(\d+).")
            batch = re.findall(re_batch, ",".join(filelists))
            batch = [int(i) for i in set(batch)]
            if remain == 0:
                for file in filelists:
                    if "nlp_chat" in file:
                        os.remove(self.model_path + file)
                os.remove(self.model_path + "checkpoint")
                return
            if len(batch) > remain:
                for bat in sorted(batch)[:-(remain)]:
                    for file in filelists:
                        if str(bat) in file and "nlp_chat" in file:
                            os.remove(self.model_path + file)
        except Exception as e:
            return

    def test(self):
        with tf.Session() as sess:

            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 获取输入输出
            train_src = [[2, 3, 5], [7, 8, 2, 4, 7], [9, 2, 1, 2]]
            train_targets = [[2, 3], [6, 4, 7], [7, 1, 2]]

            loss_track = []

            for batch in range(self.max_batches + 1):
                # 获取fd [time_steps, batch_size]
                fd = self.data_iter(train_src,
                                    train_targets,
                                    2,
                                    3)

                _, loss, _, _ = sess.run([self.model.train_op,
                                          self.model.loss,
                                          self.model.gradient_norms,
                                          self.model.updates], fd)
                loss_track.append(loss)

                if batch == 0 or batch % self.show_epoch == 0:
                    print("-" * 50)
                    print("epoch {}".format(sess.run(self.model.global_step)))
                    print('  minibatch loss: {}'.format(
                        sess.run(self.model.loss, fd)))

                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[self.model.decoder_targets].T,
                            sess.run(self.model.decoder_prediction_train, fd).T
                    )):
                        print('  sample {}:'.format(i + 1))
                        print('    dec targets > {}'.format(e_in))
                        print('    dec predict > {}'.format(dt_pred))
                        if i >= 3:
                            break


if __name__ == '__main__':
    seq_obj = Seq2seq()
    if sys.argv[1]:
        if sys.argv[1] == 'train':
            seq_obj.train()
        elif sys.argv[1] == 'infer':
            seq_obj.predict()
```
## 本章小结
1.本章从神经网络的基础理论开始，系统介绍了深度学习在NLP领域的应用。从基础的多层感知机的浅层网络，到调参、BP、word2vec词向量方法，再到RNN、LSTM、Attention机制、Seq2Seq方法，进行了系统介绍。
