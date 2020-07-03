# 第九章 NLP中用到的机器学习算法
## 简介
1. NLP领域中，应用了大量的机器学习模型和方法，可以说机器学习是NLP的基石。
2. 文本的分类、语音识别、文本的聚类都要用到机器学习中的方法。
3. 机器学习的核心在于建模和算法，学习得到的参数只是一个结果。
### 机器学习训练的要素
1. 成功训练一个模型需要四个要素：数据、转换数据的模型、衡量模型好坏的损失函数和一个调整模型权重以便最小化损失函数的算法。
2. 数据：对于数据，肯定是越多越好。事实上，数据是机器学习发展的核心，因为复杂的非线性模型比其他线性模型需要更多模型。  
3. 模型：通常数据和我们最终想要的相差很远，例如我们想知道照片中人是不是在高兴，所以我们需要把一千万像素变成一个高兴度的概率值。通常我们需要在数据上应用数个非线性函数(例如神经网络)。
4. 损失函数：我们需要对比模型的输出和真实值之间的误差。损失函数可以帮助我们平衡先验和后验的期望，以便我们做出决策。损失函数的选取，取决于我们想短线还是长线。  
5. 训练误差：这是模型在训练数据集上的误差。我们通过最小化损失函数来寻找最优参数。不幸的是，即使我们在训练集上面你和得很好，也不能保证在新的没见过的数据上我们可以仍然做得很好。  
6. 测试误差：这是模型在没见过的新数据上的误差，可能会跟训练误差不一样(统计上叫过拟合)。
### 机器学习的组成部分
1. 机器学习里最重要的四类问题(按学习结果分类)： 
```
1. 预测(Prediction): 一般用回归(Regression，Arima)等模型。  
2. 聚类(Clustering): K-means方法。
3. 分类(Classification): 如支持向量机法(Support Vector Machine，SVM), 逻辑回归(Logistic Regression)。
4. 降维(Dimensional reductional): 如主成分分析法(Principal Component Analysis，PCA，即纯矩阵运算)。
```
2. 按照学习方法，机器学习又可以分为如下几类：
```
1. 监督学习(Supervised Learning，如深度学习)  
2. 无监督学习(Un-supervised Learning，如聚类)  
3. 半监督学习(Semi-supervised Learning)  
4. 增强学习(Reinforced Learning)  
```
3. 监督学习描述的任务是，当给定输出x，如何通过在标注了输入和输出的数据上训练模型而预测输出y。从统计角度来说，监督学习主要关注如何估计条件概率p(y|x)。在实景场景中，监督学习最为常用。  
4. 监督学习的基本流程：  
```
1. 先准备训练数据，可以是文本、图像、音频、数字等，然后抽取所需要的特征，形成特征向量(Feature Vectors)  
2. 把这些特征连同对应的标记(label)一起喂给学习算法，训练出一个预测模型(Predictive Model)  
3. 采用同样的特征抽取方法作用于新数据，得到用于测试特征向量。
4. 使用预测模型对将来的数据进行预测。  
```
5. 除了分类问题以外，常用的还有回归预测。回归与分类的区别在于，预测的目标是连续变量。回归分析也是监督学习里最简单的一类任务，在该类的任务里，输入是任何离散或连续的、单一或多个的变量，而输出是连续的数值。  
6. 无监督学习即在没有人工标记的情况下，计算机进行预测、分类等工作。  
7. 半监督介于监督学习和无监督学习两者之间，增强学习牵扯到更深的运算、随机过程、博弈论基础，这里暂时不展开。  
8. 机器学习作为新创的学科或方法，被广泛应用于人工智能和数据科学等领域问题的求解。按行业的说法，神经网络、深度学习、增强学习等模型都属于机器学习的范畴。
## 几种常用的机器学习方法
### 文本分类
1. 文本分类技术在NLP领域有着举足轻重的地位。文本分类是指在给定分类体系，根据文本内容自动确定文本类别的过程。20世纪90年代以来，文本分类已经出现了很多应用，比如信息检索、Web文档自动分类、数字图书馆、自动文摘、分类新闻、文本过滤、单词语义辨析、情感分析等。  
2. 最基础的分类是归到两个类别中，称为二分类(Binary Classification)问题，即是判断是非问题。  
3. 分类过程主要分为两个阶段，训练阶段和预测阶段。训练阶段根据训练数据训练得到分类模型。预测阶段根据分类器推断出文本所属的类别。训练阶段一般需要先分词，然后提取文本为特征，提取特征的过程称为特征提取。  
4. 常见的分类器有逻辑回归(Logistic Regression，LR。名义上虽然是回归，其实是分类)、支持向量机(Support Vector Machines，SVM)、K近邻(K-Nearest，KNN)、决策树(Decision Tree，DT)、神经网络(Neural Network，NN)等。  
5. 如果特征数量很多，跟样本数量差不多，这是选择LR或者线性的SVM。如果特征数量比较少，样本数量一般，不大也不小，选择SVM的高斯核版本。如果数据量分非常大，又非线性，可以使用决策树的升级版本-随机森林。当数据达到巨量时，特征向量也非常大，则需要使用神经网络拓展到现在的深度学习模型。  
6. 交叉验证是用来验证分类器性能的一种统计方法。基本思想是在训练集的基础上将分为训练集和验证集，循环往复，提升性能。  
7. 文本分类大致分为如下几个步骤：
```
1. 定义阶段：定义数据以及分类体系，具体分为哪些类别，需要哪些数据。  
2. 数据预处理：对文档做分词，去停用词等准备工作。  
3. 数据提取特征：对文档矩阵进行降维，提取训练集中最有用的特征。  
4. 模型训练阶段：选择具体的分类模型以及算法，训练出文本分类器。  
5. 评测阶段：在测试集上测试并评价分类器的性能。  
6. 应用阶段：应用性能最高的分类模型对待分类文档进行分类。
```
### 特征提取
在使用分类器之前，需要对文本提取特征，而一般来说，提取特征有几种经典的方法：
```
1. Bag-of-words：最原始的特征集，一个单词/分词就是一个特征。往往一个数据集就会有成千上万的特征；有一些简单的指标可以帮助筛选掉一些对分类没帮助的词语，例如去停词、计算信息熵等。  
2. 统计特征：包括Term frequency(TF)、Inverse document frequency(IDF)，以及合并起来的TF-IDF。这种语言模型主要是用词汇的统计特征来作为特征集，每个特征都能够说得出物理意义，看起来会比bag-of-words效果好，但实际效果也差不多。  
3. N-Gram：一种考虑了词汇顺序的模型，就是N阶Markov链，每个样本转移成转移概率矩阵，也能取得不错效果。
```
### 标注
给定一个输入，输出不定量的类别，这个就叫作标注任务，这类任务有时候也叫多标签分类。
### 搜索与排序
互联网时代早期，谷歌研发出一个著名的网页排序算法-PageRank。该算法的排序结果并不取决于特定的用户检索条目，这些排序结果可以更好地为所包含的检索条目的网页进行排序。
### 推荐系统
推荐系统和搜索排序关系紧密，并且被广泛应用于电子商务、搜索引擎、新闻门户等。推荐系统的主要目标是把用户可能感兴趣的东西推荐给用户。
### 序列学习
1. 序列学习是一类近来备受关注的机器学习问题。在这类问题中，需要考虑顺序问题，输入和输出的长度不固定(如翻译，输入英文和翻译出来的中文长度都是不固定的)。这类模型通常可以处理任何长度的输入序列，或者输出任何长度的序列。当输入和输出都是不定长的序列时，我们把这类模型称为seq2seq，例如QA问答系统、语言翻译模型和语音转录文本模型。  
2. 语音识别：在语音识别的问题里，输入序列通常都是麦克风的声音，而输出是对通过麦克风所的话的文本转录。  
3. 文本转录语音：这是语音识别问题的逆问题。这里的输入是一个文本序列，而输出才是声音序列。  
4. 机器翻译：机器翻译的目标是把一段话从一种语言翻译成另一种语言。  
## 分类器方法
### 朴素贝叶斯(Naive Bayesian)
1. 朴素贝叶斯方法是基于贝叶斯定理与特征条件独立假设的分类方法，对于给定的训练集合，首先基于特征独立(所以叫做朴素版的贝叶斯)学习输入、输出的联合概率分布；然后基于此模型，对给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。
2. 朴素贝叶斯方法简单，学习与预测的效率都很高，是常用的方法。
### 逻辑回归
1. 逻辑回归(Logistic Regression)是统计机器学习中的经典方法，虽然简单，但是由于其模型复杂度低，不容易过拟合，计算复杂度小。  
2. 逻辑回归有很多优点，比如容易实现、分类时计算量小，速度快、存储资源低等；缺点也是明显的，比如容易过拟合、准确度欠佳等。
### 支持向量机
1. 支持向量机(SVM)的最终目的是在特征空间中寻找一个尽可能将两个数据集合分开的超级平面(hyper-plane)。  
2. SVM算法优点：可用于线性/非线性分类，也可以用于回归；低误差率；推导过程优美，容易解释；计算复杂度较低。缺点：对参数和核函数的选择比较敏感；原始的SVM只擅长处理二分类问题。
## 无监督学习的文本聚类
1. 无监督学习(Un-supervised Learning)希望能够发现数据本身的规律和模式，与监督学习相比，无监督学习不需要对数据进行标记。这样可以节省大量的人力、物力，也可以让数据的获取变得非常容易。某种程度上说，机器学习的终极目标就是无监督学习。
2. 从功能上看，无监督学习可以帮助我们发现数据的“簇”，同时也可以帮助我们找寻“离群点”(outlier)；此外，对于特征维度特别高的数据样本，我们同样可以通过无监督学习对数据降维，保留数据的主要特征，这样对高维空间的数据也可以进行处理。  
3. 一些常见的无监督学习任务：
```
1. 聚类问题通常研究如何把一堆数据点分成若干类，从而使得同类数据点相似而非同类数据点不相似。
2. 子空间估计问题通常研究如何将原始数据向量在更低维度下表示。理想情况下，子空间的表示要具有代表性才能与原始数据接近。一个常用的方法叫做主成分分析。  
3. 表征学习希望在欧几里德空间中找到原始对象的表示方式，从而能在欧几里德空间里表示出原始对象的符号性质。  
4. 生成对抗网络是最近很火的一个领域。这里描述数据的生成过程，并检查真实数据与生成的数据是否统计上相似。  
```
4. 聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个“簇”(cluster)。通过这样的划分，每个簇可能对应于一些潜在的类别。聚类常用于寻找数据内在的分布结构，也可以作为分类等其他学习任务的前驱过程。
5. 在NLP领域，一个重要的应用方向是文本聚类，文本聚类有很多算法，例如K-means、DBScan、BIRCH、CURE等。  
6. K-means算法是一种非监督学习的算法，它解决的是聚类问题。将一些数据通过无监督的方式，自动化聚集出一些簇。文本聚类存在大量的使用场景，比如数据挖掘、信息检索、主题检测、文本概括等。
7. 文本聚类对文档集合进行划分，使得同类别的文档聚合在一起，不同类别的文档相似度比较小。文本聚类不需要预先对文档进行标记，具有高度的自动化能力。  
8. K-means算法接收参数k，然后将事先输入的n个数据对象划分为k个聚类以便使得所获得的聚类满足聚类中的对象相似度较高，而不同聚类中的对象相似度较小。  
9. K-means算法思想：以空间中k个点为中心进行聚类，对最接近他们的对象归类，通过迭代的方法，逐次更新各聚类中心点的值，直到得到最好的聚类结果。  
10. K-means算法描述：
```
1. 适当选择c个类的初始中心  
2. 在第k次迭代中，对任意一个样本求其到c个中心的距离，将该样本归到距离最短的那个中心所在的类。  
3. 利用均值等方法更新该类的中心值。  
4. 对于所有的c个聚类中心，如果利用上述(2)和(3)的迭代法更新后，值保持不变，则迭代结束；否则继续优化。 
```
11. 初始的聚类点对后续的最终划分有非常大的影响，选择合适的初始点，可以加快算法的收敛速度和增强类之间的区分度。选择初始聚类点的方法有如下几种： 
```
1. 随机选择法。随机选择k个对象作为初始聚类点。  
2. 最小最大法。先选择所有对象中相距最遥远的两个对象作为聚类点。然后选择第三个点，使得它与确定的聚类点的最小距离是所有中心点中最大的，然后按照相同的原则选取。  
3. 最小距离法。选择一个正数r，把所有对象的中心作为第一个聚类点，然后依次输入对象，当前输入对象与已确认的聚点的距离大于r时，则该对象作为一个新的聚类点。 4. 最近归类法。划分方法就是决定当前对象应该分到哪一簇中。
```
## 文本分类实战：中文垃圾邮件分类
### 实现代码
1. 数据归整和预处理(09_normalization.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''


import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:

        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)

    return normalized_corpus
```
2. 特征提取(09_feature_extractors.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
```
3. 模型训练与评价(09_classifier.py)
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
from sklearn.model_selection import train_test_split


def get_data():
    '''
    获取数据
    :return: 文本数据，对应的labels
    '''
    with open("data/ham_data.txt", encoding="utf8") as ham_f, open("data/spam_data.txt", encoding="utf8") as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()

        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()

        corpus = ham_data + spam_data

        labels = ham_label + spam_label

    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    '''
    
    :param corpus: 文本数据
    :param labels: label数据
    :param test_data_proportion:测试数据占比 
    :return: 训练数据,测试数据，训练label,测试label
    '''
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=test_data_proportion, random_state=42)
    return train_X, test_X, train_Y, test_Y


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels


from sklearn import metrics


def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(
        metrics.accuracy_score(true_labels,
                               predicted_labels),
        2))
    print('精度:', np.round(
        metrics.precision_score(true_labels,
                                predicted_labels,
                                average='weighted'),
        2))
    print('召回率:', np.round(
        metrics.recall_score(true_labels,
                             predicted_labels,
                             average='weighted'),
        2))
    print('F1得分:', np.round(
        metrics.f1_score(true_labels,
                         predicted_labels,
                         average='weighted'),
        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions


def main():
    corpus, labels = get_data()  # 获取数据集

    print("总的数据量:", len(labels))

    corpus, labels = remove_empty_docs(corpus, labels)

    print('样本之一:', corpus[10])
    print('样本的label:', labels[10])
    label_name_map = ["垃圾邮件", "正常邮件"]
    print('实际类型:', label_name_map[int(labels[10])], label_name_map[int(labels[5900])])

    # 对数据进行划分
    train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus,
                                                                            labels,
                                                                            test_data_proportion=0.3)

    from normalization import normalize_corpus

    # 进行归一化
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)

    ''.strip()

    from feature_extractors import bow_extractor, tfidf_extractor
    import gensim
    import jieba

    # 词袋模型特征
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf 特征
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # tokenize documents
    tokenized_train = [jieba.lcut(text)
                       for text in norm_train_corpus]
    print(tokenized_train[2:10])
    tokenized_test = [jieba.lcut(text)
                      for text in norm_test_corpus]
    # build word2vec 模型
    model = gensim.models.Word2Vec(tokenized_train,
                                   size=500,
                                   window=100,
                                   min_count=30,
                                   sample=1e-3)

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', max_iter = 100)
    lr = LogisticRegression()

    # 基于词袋模型的多项朴素贝叶斯
    print("基于词袋模型特征的贝叶斯分类器")
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)

    # 基于词袋模型特征的逻辑回归
    print("基于词袋模型特征的逻辑回归")
    lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                      train_features=bow_train_features,
                                                      train_labels=train_labels,
                                                      test_features=bow_test_features,
                                                      test_labels=test_labels)

    # 基于词袋模型的支持向量机方法
    print("基于词袋模型的支持向量机")
    svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)


    # 基于tfidf的多项式朴素贝叶斯模型
    print("基于tfidf的贝叶斯模型")
    mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)
    # 基于tfidf的逻辑回归模型
    print("基于tfidf的逻辑回归模型")
    lr_tfidf_predictions=train_predict_evaluate_model(classifier=lr,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)


    # 基于tfidf的支持向量机模型
    print("基于tfidf的支持向量机模型")
    svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                         train_features=tfidf_train_features,
                                                         train_labels=train_labels,
                                                         test_features=tfidf_test_features,
                                                         test_labels=test_labels)



    import re

    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 0 and predicted_label == 0:
            print('邮件类型:', label_name_map[int(label)])
            print('预测的邮件类型:', label_name_map[int(predicted_label)])
            print('文本:-')
            print(re.sub('\n', ' ', document))

            num += 1
            if num == 4:
                break

    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
        if label == 1 and predicted_label == 0:
            print('邮件类型:', label_name_map[int(label)])
            print('预测的邮件类型:', label_name_map[int(predicted_label)])
            print('文本:-')
            print(re.sub('\n', ' ', document))

            num += 1
            if num == 4:
                break


if __name__ == "__main__":
    main()
-- 上述代码输出结果
总的数据量: 10001
样本之一: 北京售票员可厉害，嘿嘿，有专座的，会直接拉着脖子指着鼻子让上面的人站起来让 座的，呵呵，比较赞。。。 杭州就是很少有人给让座，除非司机要求乘客那样做。 五一去杭州一个景点玩，车上有两个不到一岁的小孩，就是没有人给让座，没办法家长只能在车上把小孩的推车打开让孩子坐进去，但是孩子还是闹，只能抱着，景点离市区很远，车上很颠，最后家长坐在地上抱孩子，就是没有一个人给让座，要是在北京，一上车就有人让座了

样本的label: 1.0
实际类型: 正常邮件 垃圾邮件

[['10156', '说', '的', '呵呵', '标题', 'Re', '我', '要是', '你', '女朋友', '绝对', '跟', '你', '分手', 'Re', '昨晚', '猫', '又', '闹', '了', '一夜', '嗯', '，', '谁', '说', '我', '养猫', '是因为', '有', '爱心', '我', '跟', '谁', '急', '是', '哈', '，', '不是', '用', '爱心', '区分', '的', '喜欢', '宠物', '的', '就', '养', '，', '不', '喜欢', '的', '就', '不养', '呗', '卷', '卷', '，', '你', '搞', '成', '这', '副', '鬼', '样子', '，', '还', '好意思', '来', '找', '我', '撒娇', '。', '。', '。'], ['中信', '（', '国际', '）', '电子科技', '有限公司', '推出', '新', '产品', '：', '升职', '步步高', '、', '做生意', '发大财', '、', '连', '找', '情人', '都', '用', '的', '上', '，', '详情', '进入', '网址', 'httpwwwusa5588comccc', '电话', '：', '02033770208', '服务', '热线', '：', '013650852999'], ['贵', '公司', '负责人', '：', '你好', '！', '本', '公司', '祥泰', '实业', '有限公司', '）', '具有', '进出口', '及', '国内贸易', '的', '企业', '承', '多家', '公司', '委托', '有', '广告', '建筑工程', '其它', '服务', '商品销售', '等', '的', '发票', '向', '外代', '开', '点数', '优惠', '本', '公司', '原则', '是', '满意', '后', '付款', '有意者', '请来', '电', '洽谈', '电话', '：', '013631690076', '邮箱', '：', 'shitailong8163com', '联系人', '：', '郭生', '如', '给', '贵', '公司', '带来', '不便', '请谅解'], ['李敖来', '大陆', '，', '轰轰烈烈', '，', '热热闹闹', '，', '国内', '有些', '人', '也', '坐不住', '了', '，', '总', '有人', '要', '跳', '出来', '显摆', '显摆', '，', '不过', '，', '没显', '好', '倒', '是', '现', '了', '眼', '。', '大家', '看看', '下面', '的', '材料', '，', '真', '为', '清华', '汗颜', '啊', '！', '资料', '一', '【', '清华大学', '学者', '评李敖', '演讲', '：', '李敖', '老', '矣', '尚能', '骂否', '？', '】', 'httpnewsanhuinewscomsystem20050923001359039shtml'], ['公司', '名称', '：', '北京', '优力', '维尔', '科技', '有限公司', '高科技', '公司', '招聘', '行政助理', '兼', '前台', '一名', '要求', '：', '25', '岁', '以下', '，', '形象', '良好', '，', '有', '责任心', '、', '诚实', '守信', '、', '有', '团队', '意识', '、', '敬业', '，', '可以', '尽快', '就位', '工作', '地点', '：', '中信', '国安', '数码港', '，', '地理位置', '在', '稻香', '园桥', '桥东', '，', '中国航天', '大厦', '西边', '待遇', '面议', '联系方式', '：', '简历', 'mailtowuwei99tsinghuaorgcn', '公司', '的', '新', '网站', '和', 'mail', '服务器', '都', '在建设中', '，', '投递', '的', '简历', '中', '最好', '附带', '照片', '联系电话', '：', '82652066', '工作日', '9001800', '欢迎', '投递', '简历', '，', '合则', '约见'], ['如果', '他', '只是', '想', '找个', '上床', '的', '，', '这招', '可能', '管用', '，', '但', '对', '他', '身边', '的', '女人', '，', '这招', '不', '适用', '，', '聪明', '的', '老板', '不吃', '窝边草', '。', '是否', '只', '上床', '从来不', '由', '男人', '单方面', '说了算', '。', '如果', '是', '的话', '，', '呵呵', '，', '恐怕', '结婚', '率会', '大大降低', '的', '。', '窝边草', '心理', '是', '个', '阻碍', '，', '克服', '它', '就是', '了', '。', '个人', '猜测', '，', '这种', '人', '一般', '喜欢', '能', '给', '他', '安全感', '的', '女人', '，', '即', '传统', '的', '贤妻良母', '。', '至少', '，', '他们', '更', '倾向', '于', '找', '此类', '女人', '做', '老婆', '。', '即使', '他们', '骨子里', '喜欢', '风骚', '的', '女人', '，', '呵呵', '。', '这个', '猜测', '才', '没道理', '。', '你', '在', '毫无道理', '地', '给', '楼主', '泄气', '。'], ['家园网', '提供', '12M', '免费', '主页', '空间', '，', '欢迎', '申请', '家园网', '空间', '特点', '：', '独立', '二级域名', '，', 'Web', '上传', '，', '马上', '申请', '立即', '开通', '，', '永久', '免费', '终生', '稳定', '安全', '无广告', '客服', '在线', '答疑', '。', '网址', 'httpwwwjayacn', '客服', '信箱', 'geduoyeahnet', '网络实名', '：', '家园网', '家园网', '客服', '中心'], ['中信', '（', '国际', '）', '电子科技', '有限公司', '推出', '新', '产品', '：', '升职', '步步高', '、', '做生意', '发大财', '、', '连', '找', '情人', '都', '用', '的', '上', '，', '详情', '进入', '网址', 'httpwwwusa5588comccc', '电话', '：', '02033770208', '服务', '热线', '：', '013650852999']]
基于词袋模型特征的贝叶斯分类器
准确率: 0.79
精度: 0.85
召回率: 0.79
F1得分: 0.78
基于词袋模型特征的逻辑回归
准确率: 0.96
精度: 0.96
召回率: 0.96
F1得分: 0.96
基于词袋模型的支持向量机
准确率: 0.97
精度: 0.97
召回率: 0.97
F1得分: 0.97
基于tfidf的贝叶斯模型
准确率: 0.79
精度: 0.85
召回率: 0.79
F1得分: 0.78
基于tfidf的逻辑回归模型

准确率: 0.94
精度: 0.94
召回率: 0.94
F1得分: 0.94
基于tfidf的支持向量机模型
准确率: 0.97
精度: 0.97
召回率: 0.97
F1得分: 0.97
邮件类型: 垃圾邮件
预测的邮件类型: 垃圾邮件
文本:-
中信（国际）电子科技有限公司推出新产品： 升职步步高、做生意发大财、连找情人都用的上，详情进入 网  址:  http://www.usa5588.com/ccc 电话：020-33770208   服务热线：013650852999 
邮件类型: 垃圾邮件
预测的邮件类型: 垃圾邮件
文本:-
您好！ 我公司有多余的发票可以向外代开！（国税、地税、运输、广告、海关缴款书）。 如果贵公司（厂）有需要请来电洽谈、咨询！ 联系电话: 013510251389  陈先生 谢谢 顺祝商祺! 
邮件类型: 垃圾邮件
预测的邮件类型: 垃圾邮件
文本:-
如果您在信箱中不能正常阅读此邮件，请点击这里 
邮件类型: 垃圾邮件
预测的邮件类型: 垃圾邮件
文本:-
以下不能正确显示请点此 IFRAME: http://bbs.ewzw.com/viewthread.php?tid=3790 
邮件类型: 正常邮件
预测的邮件类型: 垃圾邮件
文本:-
你好，我30，未婚，研究生毕业，想找一个在校的女孩做朋友，每月可以给她2000元零用 我不会走入她的生活中去，为她保密 qq ******* 回了一封信以后： 阿，不愿意跟在一群人后面买单，那会让人觉得特傻 只想一对一交往 【 在 leeann2002 的来信中提到: 】 啊,那您介意找一堆女孩子做朋友吗? 也不用给零用钱拉~~用来带我们一起玩就可以了~,唱歌跳舞什么的~~ (要不2000就把自己卖了...) 有照片连接吗? 
邮件类型: 正常邮件
预测的邮件类型: 垃圾邮件
文本:-
公司：集制作，创作以及宣传为一体的音乐制作文化发展公司 拥有录音棚，主要制作唱片， 也涉及电影、电视剧、广告等音乐制作及创作 职位：经理助理（说文秘也行） 要求：女   22～25 聪明本分 有一定的协调和办事能力 长相过得去 工作业务范围： 平时管理公司文件 接待、电话、上网 负责安排歌手及制作人的工作下达 待遇：试用期月薪1000 转正1500～2000，另有提成 公司地址：北京市朝阳区麦子店 电话留了会被封么？ 先信箱联系吧 
邮件类型: 正常邮件
预测的邮件类型: 垃圾邮件
文本:-
现已确定2006年度不会进行研究生培养机制改革试点，我校研究生招生类别、培养费用 等相关政策仍按现行规定执行，即我校绝大多数研究生属于国家计划内非定向培养生和 定向培养生，培养费由国家财政拨款。少数研究生（主要是除临床医学以外的专业学位 硕士生）属于委托培养生（培养费由选送单位支付）或自筹经费培养生（培养费由考生 本人自筹），其交纳培养费的标准详见我校财务处（网址：http://10.49.99.99）的公 示（2005.9.8）。 
邮件类型: 正常邮件
预测的邮件类型: 垃圾邮件
文本:-
提前征友K歌，只要是MM，但是还是有点小要求，age&lt;=23岁，相貌不在乎，身高不限。 K歌地点暂时选在交通大学附近的佰金KTV，因为ME手头有两张优惠券。 那里环境虽然比不上PARTYWORLD，但是比Melody稍微好那么一点点。 如果想报名的人请联系本人。 QQ：275738585 MSN：gao_520@hotmail.com 加我时请务必注明 SMTH 如果有意的MM，请联系我，组织好了大家以后，星期六午饭就在KTV吃。 
```
## 文本聚类实战：用K-means对豆瓣读书数据聚类
1. 数据获取(09_douban_spider.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

import ssl
import bs4
import re
import requests
import csv
import codecs
import time

from urllib import request, error

context = ssl._create_unverified_context()


class DouBanSpider:
    def __init__(self):
        self.userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        self.headers = {"User-Agent": self.userAgent}

    # 拿到豆瓣图书的分类标签
    def getBookCategroies(self):
        try:
            url = "https://book.douban.com/tag/?view=type&icn=index-sorttags-all"
            response = request.urlopen(url, context=context)
            content = response.read().decode("utf-8")
            return content
        except error.HTTPError as identifier:
            print("errorCode: " + identifier.code + "errrorReason: " + identifier.reason)
            return None

    # 找到每个标签的内容
    def getCategroiesContent(self):
        content = self.getBookCategroies()
        if not content:
            print("页面抓取失败...")
            return None
        soup = bs4.BeautifulSoup(content, "lxml")
        categroyMatch = re.compile(r"^/tag/*")
        categroies = []
        for categroy in soup.find_all("a", {"href": categroyMatch}):
            if categroy:
                categroies.append(categroy.string)
        return categroies

    # 拿到每个标签的链接
    def getCategroyLink(self):
        categroies = self.getCategroiesContent()
        categroyLinks = []
        for item in categroies:
            link = "https://book.douban.com/tag/" + str(item)
            categroyLinks.append(link)
        return categroyLinks

    def getBookInfo(self, categroyLinks):
        self.setCsvTitle()
        categroies = categroyLinks
        try:
            for link in categroies:
                print("正在爬取：" + link)
                bookList = []
                response = requests.get(link)
                soup = bs4.BeautifulSoup(response.text, 'lxml')
                bookCategroy = soup.h1.string
                for book in soup.find_all("li", {"class": "subject-item"}):
                    bookSoup = bs4.BeautifulSoup(str(book), "lxml")
                    bookTitle = bookSoup.h2.a["title"]
                    bookAuthor = bookSoup.find("div", {"class": "pub"})
                    bookComment = bookSoup.find("span", {"class": "pl"})
                    bookContent = bookSoup.li.p
                    # print(bookContent)
                    if bookTitle and bookAuthor and bookComment and bookContent:
                        bookList.append([bookTitle.strip(),bookCategroy.strip() , bookAuthor.string.strip(),
                                         bookComment.string.strip(), bookContent.string.strip()])
                self.saveBookInfo(bookList)
                time.sleep(3)

            print("爬取结束....")

        except error.HTTPError as identifier:
            print("errorCode: " + identifier.code + "errrorReason: " + identifier.reason)
            return None

    def setCsvTitle(self):
        csvFile = codecs.open("data/data.csv", 'a', 'utf_8_sig')
        try:
            writer = csv.writer(csvFile)
            writer.writerow(['title', 'tag', 'info', 'comments', 'content'])
        finally:
            csvFile.close()

    def saveBookInfo(self, bookList):
        bookList = bookList
        csvFile = codecs.open("data/data.csv", 'a', 'utf_8_sig')
        try:
            writer = csv.writer(csvFile)
            for book in bookList:
                writer.writerow(book)
        finally:
            csvFile.close()

    def start(self):
        categroyLink = self.getCategroyLink()
        self.getBookInfo(categroyLink)


douBanSpider = DouBanSpider()
douBanSpider.start()
```
2. 数据的归整和预处理(09_normalization.py(cluster文件夹内))
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:

        text =" ".join(jieba.lcut(text))
        normalized_corpus.append(text)

    return normalized_corpus
```
3. K-means聚类(09_cluster.py)
```
#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-12-06 19:48:21
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-12-06 20:04:16
'''

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


book_data = pd.read_csv('data/data.csv') #读取文件

print(book_data.head())

book_titles = book_data['title'].tolist()
book_content = book_data['content'].tolist()

print('书名:', book_titles[0])
print('内容:', book_content[0][:10])

from normalization import normalize_corpus

# normalize corpus
norm_book_content = normalize_corpus(book_content)


# 提取 tf-idf 特征
vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                  feature_type='tfidf',
                                                  min_df=0.2, max_df=0.90,
                                                  ngram_range=(1, 2))
# 查看特征数量
print(feature_matrix.shape)

# 获取特征名字
feature_names = vectorizer.get_feature_names()

# 打印某些特征
print(feature_names[:10])

from sklearn.cluster import KMeans


def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


num_clusters = 10
km_obj, clusters = k_means(feature_matrix=feature_matrix,
                           num_clusters=num_clusters)
book_data['Cluster'] = clusters

from collections import Counter

# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())


def get_cluster_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books

    return cluster_details


def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)


import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib.font_manager import FontProperties


def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, book_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'title': book_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()


cluster_data = get_cluster_data(clustering_obj=km_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=num_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))

from sklearn.cluster import AffinityPropagation


def affinity_propagation(feature_matrix):
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_
    return ap, clusters


# get clusters using affinity propagation
ap_obj, clusters = affinity_propagation(feature_matrix=feature_matrix)
book_data['Cluster'] = clusters

# get the total number of books per cluster
c = Counter(clusters)
print(c.items())

# get total clusters
total_clusters = len(c)
print('Total Clusters:', total_clusters)

cluster_data = get_cluster_data(clustering_obj=ap_obj,
                                book_data=book_data,
                                feature_names=feature_names,
                                num_clusters=total_clusters,
                                topn_features=5)

print_cluster_data(cluster_data)

plot_clusters(num_clusters=num_clusters,
              feature_matrix=feature_matrix,
              cluster_data=cluster_data,
              book_data=book_data,
              plot_size=(16, 8))

from scipy.cluster.hierarchy import ward, dendrogram


def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, book_data, figure_size=(8, 12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    book_titles = book_data['title'].values.tolist()
    # plot dendrogram
    ax = dendrogram(linkage_matrix, orientation="left", labels=book_titles)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=200)


# build ward's linkage matrix
linkage_matrix = ward_hierarchical_clustering(feature_matrix)
# plot the dendrogram
plot_hierarchical_clusters(linkage_matrix=linkage_matrix,
                           book_data=book_data,
                           figure_size=(8, 10))
-- 上述代码输出结果
    title  ...                                            content
0  ﻿解忧杂货店  ...  现代人内心流失的东西，这家杂货店能帮你找回——\r\n僻静的街道旁有一家杂货店，只要写下烦恼...
1   巨人的陨落  ...  在第一次世界大战的硝烟中，每一个迈向死亡的生命都在热烈地生长——威尔士的矿工少年、刚失恋的美...
2   我的前半生  ...  一个三十几岁的美丽女人子君，在家做全职家庭主妇。却被一个平凡女人夺走丈夫，一段婚姻的失败，让...
3    百年孤独  ...  《百年孤独》是魔幻现实主义文学的代表作，描写了布恩迪亚家族七代人的传奇故事，以及加勒比海沿岸...
4   追风筝的人  ...  12岁的阿富汗富家少爷阿米尔与仆人哈桑情同手足。然而，在一场风筝比赛后，发生了一件悲惨不堪的...

[5 rows x 5 columns]
书名: ﻿解忧杂货店
内容: 现代人内心流失的东西
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/rn/r6_l2xln77j0bv69j2c_5rg00000gp/T/jieba.cache
Loading model cost 0.655 seconds.
Prefix dict has been built successfully.
(2822, 16281)
['000', '01', '07', '09', '10', '100', '1000', '1001', '108758', '11']
dict_items([(0, 968), (2, 373), (4, 347), (9, 229), (6, 191), (5, 136), (8, 44), (7, 154), (1, 264), (3, 116)])
Cluster 0 details:
--------------------
Key features: ['本书', '艺术', '内容', 'the', '作品']
book in this cluster:
﻿解忧杂货店, 新名字的故事, 鱼王, 新名字的故事, 鱼王, 外婆的道歉信, 我的天才女友, 鱼王, 外婆的道歉信, 目送, 吃鲷鱼让我打嗝, 孤独六讲, 瓦尔登湖, 文学回忆录（全2册）, 爱你就像爱生命, 艺术的故事, ﻿解忧杂货店, 百鬼夜行 阳, 金色梦乡, 不思议图书馆, 目送, 瓦尔登湖, 此生多珍重, 冬牧场, 1Q84 BOOK 2, 大萝卜和难挑的鳄梨, 国境以南 太阳以西, 远方的鼓声, 舞！舞！舞！, 没有女人的男人们, 无比芜杂的心绪, 爱吃沙拉的狮子, 东京奇谭集, 遇到百分之百的女孩, ﻿一只狼在放哨, 万物静默如谜, 我的孤独是一座花园, 海子诗全集, 事物的味道，我尝得太早了, 唯有孤独恒常如新, 飞鸟集, 二十首情诗与绝望的歌, 荒原, 王尔德童话, 夜莺与玫瑰, 猜猜我有多爱你, 牧羊少年奇幻之旅, 银河铁道之夜, 狐狸的窗户, 爱你就像爱生命, 我的精神家园, 王小波全集, 白银时代, 假如你愿意你就恋爱吧, 万寿寺, 三十而立, 我的精神家园, 王小波全集, 门萨的娼妓, 这就是二十四节气, 猜猜我有多爱你, 草房子, 梦书之城, 柑橘与柠檬啊, 《噼里啪啦系列》, 深夜小狗神秘事件, 诗经, 闲情偶寄, 东京梦华录, 声律启蒙, 牡丹亭, 既见君子, 麦琪的礼物, 兄弟（上）, 兄弟, 兄弟（下）, 音乐影响了我的写作, 温暖和百感交集的旅程, 余华作品系列（共12册）, 我没有自己的名字, 我能否相信自己, 我胆小如鼠, 张爱玲文集, 色，戒, 异乡记, 红楼梦魇, 雷峰塔, 传奇(上下), 张看（上下）, 传奇, 阿城精选集, 额尔古纳河右岸, 棋王.樹王.孩子王, 人·兽·鬼, 七缀集, 宋诗选注, 钱钟书文集, 槐聚心史, 麦琪的礼物, 茶花女, 堂吉诃德, 丧钟为谁而鸣, 鲁迅全集（2005最新修订版）, 狂人日记, 野草, 彷徨, 中国小说史略, 两地书, 鲁迅小说全集, 诗经, 闲情偶寄, 东京梦华录, 声律启蒙, 牡丹亭, 既见君子, ﻿人类的群星闪耀时, 心灵的焦灼, 斯·茨威格中短篇小说选, 一个政治性人物的肖像, 变形的陶醉, 玛丽·斯图亚特, 不朽, 帷幕, 相遇, 米兰·昆德拉传, 广岛之恋, 中国北方的情人, 杜拉斯传, 毁灭，她说, 外面的世界, ﻿生死桥, 树犹如此, 夏宇詩集／Salsa, 孤獨，或類似的東西, 東課樓經變, 周夢蝶詩文集, 七声, 無愛紀, 她是女子，我也是女子, 简媜散文, 我城, ﻿疯狂艺术史：从达芬奇到伦勃朗, 切尔诺贝利之花, 伊甸园世界01：星之上, 卢浮宫的守护者, 金色梦乡, 玫瑰的名字, 长夜难明, 明镜之书, 希腊棺材之谜, 放学后, ﻿你今天真好看, 活了100万次的猫, 向左走·向右走, 猜猜我有多爱你, 这些都是你给我的爱, Yaxin, le faune Gabriel, 睡不着：Tango一日一画, 我不喜欢这世界，我只喜欢你, 从你的全世界路过, 坏一坏, 致我们终将逝去的青春, ﻿解忧杂货店, 放学后, 时生, 雪地殺機, 圣女的救济, 秘密, 拉普拉斯的魔女, 超·杀人事件, ﻿三体, 献给阿尔吉侬的花束, 火星救援, 华氏451, 摩天楼, 软件体的生命周期, 岛屿之书, 鹤唳华亭, 曾有一个人，爱我如生命, 夜旅人, 海棠依舊: 知否? 知否? 應是綠肥紅瘦 卷一, 三生三世 枕上书, 四月间事, 孤城闭, 原来你还在这里, 七根凶简, 长夜难明, 明镜之书, 锦衣行, 天龙八部, 神雕侠侣, 射雕英雄传（全四册）, 有匪2：离恨楼, 刀背藏身, 剑桥倚天屠龙史, 倚天屠龙记(共四册), 陆小凤传奇, 欢乐英雄, ﻿贝伦与露西恩, 无根之木, 九州·缥缈录, 有顶天家族, ﻿散步去, 大叔上等, 灌篮高手31, 机器猫哆啦A梦23, 非法侵入(01), 死亡筆記 1, 铳梦, 竹光侍 1, 天是红河岸, 20世紀少年01, PRIDE (下巻) (爆男COMICS), 我可以被擁抱嗎？因為太過寂寞而叫了蕾絲邊應召, PRIDE, ﻿1988：我想和这个世界谈谈, 我所理解的生活, 他的国, 光明与磊落, 长安乱, 所有人问所有人, 杂的文, 韩寒五年文集（上下）, 飄移中國, 桃花债 上, 義父～梅花日記～, 成化十四年, 麒麟之与子同袍, 同級生, 我爱摇滚乐, 不疯魔不成活, mother, 乱世为王（上卷）, 圆舞, 她比烟花寂寞, 开到荼蘼, 承欢记, 亦舒作品集(共60册), 人淡如菊, 朝花夕拾, 寒武纪, 她的二三事, 印度墨, 亦舒新经典, 乐未央, 不羁的风, 亦舒作品系列, 阿修罗, 金色梦乡, 长夜难明, 希腊棺材之谜, 放学后, 一朵桔梗花, 梦里花落知多少, 雨季不再来, 三毛典藏全集, 亲爱的三毛, 你是我不及的梦, 滚滚红尘, 流星雨, 傾城, ﻿悟空传, 明朝那些事儿（壹）, 盗墓笔记, 琅琊榜, 致我们终将逝去的青春, 临高启明, 鹤唳华亭, 海棠依舊: 知否? 知否? 應是綠肥紅瘦 卷一, 怨氣撞鈴1·食骨, ﻿莲花, 蔷薇岛屿, 清醒纪, 安妮宝贝系列, 大方 No.1, 古书之美, 月, 悲伤逆流成河, 幻城, 小时代2.0虚铜时代, 爵迹•燃魂书, 小时代3.0：刺金时代, 愿风裁尘, 小时代1.5·青木时代VOL.2, 临高启明, 夜旅人, 海棠依舊: 知否? 知否? 應是綠肥紅瘦 卷一, 大唐明月1·风起长安, 步步惊心, 不负如来不负卿1, 史上第一混乱·第一季神仙预备役, 扶摇皇后（上下）, 独步天下（上下卷）, 第一风华1·天赐良缘（上下）, 官居一品（全七册）, ﻿天龙八部, 神雕侠侣, 射雕英雄传（全四册）, 倚天屠龙记(共四册), 白马啸西风, 书剑恩仇录(上下), 飞狐外传(上下), ﻿不干了！我开除了黑心公司, 迟来的翅膀, 未聞花名（上）, 四叠半神话大系, 末日時在做什麼？有沒有空？可以來拯救嗎？ 01, 空之境界（上）, 奇诺之旅I, Fate/Zero Vol.1 [第四次圣杯战争秘话], 斬首循環, 言叶之庭, 冰菓, 刀剑神域 01, 古书堂事件手帖 01, 狼與辛香料, 少女不十分, 三日間的幸福, 尼罗河上的惨案, 罗杰疑案, 死亡之犬, 幕后凶手, 阿加莎·克里斯蒂侦探推理“波洛”系列（全32册）, 大侦探波洛精选集, 牙医谋杀案, 阳光下的罪恶, 捕鼠器, 啤酒谋杀案, 无尽长夜, 谋杀启事, 斯泰尔斯庄园奇案, ﻿向左走·向右走, 月亮忘記了, 世界別為我擔心, 又寂寞又美好, 我只能為你畫一張小卡片, 照相本子, 谢谢你毛毛兔，这个下午真好玩, ﻿三体, 火星救援, 软件体的生命周期, 少数派报告, 十六夜膳房, 夏洛克是我的名字, 我还是很喜欢你, 这世界正在遗忘不改变的人, 七根凶简, 大师和玛格丽特, 魔戒（第一部）, 冰与火之歌（卷三）, 冰与火之歌（卷二）, ﻿生死桥, 她是女子，我也是女子, 三十年细说从头 (上下), WKW, 别来无恙, 龙头凤尾, 無愛紀, 飞毡, 穿KENZO的女人, ﻿向左走·向右走, 月亮忘記了, 世界別為我擔心, 又寂寞又美好, 我只能為你畫一張小卡片, 照相本子, 谢谢你毛毛兔，这个下午真好玩, 偶发空缺, 蚕, 哈利·波特的哲学世界, Harry Potter Page to Screen, Fantastic Beasts and Where to Find Them, 一个人住第9年, 一個人去旅行, 一個人上東京, 一个人漂泊的日子①, 节日万岁！, 一個人的第一次, 150cm Life 2, 一個人去跑步, ﻿欢乐英雄, 陆小凤传奇, 楚留香传奇, 三少爷的剑, 古龙作品全集（全53卷）, 七种武器（全三册）, 绝代双骄（全三册）, 小李飞刀系列, 边城浪子（上下）, 武林外史（上下）, 谁来跟我干杯, 护花铃, 曼珠沙华, 听雪楼系列典藏版（全三册）, 忘川 上, 镜·双城, 帝都赋, 血薇, 大漠荒颜, 羽·赤炎之曈, 忘川 下, 七夜雪, 花镜, 沧海, 彼岸花, ﻿那些生命中温暖而美好的事情, 不朽, 千秋, 年华是无效信, 须臾, 万象, 尘埃星球, 文艺风象·假妆, 《文艺风象+文艺风赏》第一期合刊, 文艺风象·A梦, 文艺风象·消夏, 文艺风象·雨天爱好者, 文艺风象·TA和TA的猫, ﻿水仙已乘鲤鱼去, 鲤·写信, 鲤·上瘾, 鲤·来不及, 鲤·谎言, 十爱, 是你来检阅我的忧伤了吗, 昼若夜房间, 巧克力味的暑假/阳光姐姐小书房, 年华是无效信, 会有天使替我爱你, 高考战斗手册, 聲之形 01, 青春好似大脸狗, 做好学生有点累-伍美珍作品, 那些回不去的年少时光（套装上下册）, 丝绸之路, 明朝那些事儿（壹）, 史记（全十册）, 自控力, 逃避自由, 思考，快与慢, ﻿哥德尔、艾舍尔、巴赫, 失控, 苏菲的世界, 维特根斯坦传, 加缪手记, 逻辑学导论, 维特根斯坦传, 人类的群星闪耀时, 当呼吸化为空气, 阿拉伯的劳伦斯, 别闹了，费曼先生, 我心深处, 沈从文的后半生, 三十年细说从头 (上下), 艺术的故事, 文学回忆录（全2册）, 失控, 娱乐至死, 美的历程, 丝绸之路, 乡下人的悲歌, 天真的人类学家, 贸易的猜忌, ﻿艺术的故事, 疯狂艺术史：从达芬奇到伦勃朗, 脸的历史, 美的历程, 现代艺术150年, 如花在野, 梵高手稿, 孤独之间, 手绘的奇思妙想, 零ZEЯRO, Thoughts on Design, 色彩设计的原理, 何为美好生活：Beams at Home, 娱乐至死, 动物农场, 伯罗奔尼撒战争, 美丽灵魂, 贸易的猜忌, 北京的城墙与城门, 建筑家安藤忠雄, 建构文化研究, 空间的诗学, 建筑十书, 建筑，从那一天开始, 交往与空间, 负建筑, 贝聿铭全集, ﻿悉达多, 西藏生死书, 圣经, 科学与宗教的领地, 慕尼黑的清真寺, A Secular Age, 沉默, 千面英雄, 人间是剧场, 眼泪与圣徒, 故事, 三十年细说从头 (上下), 导演的诞生, 与奥逊·威尔斯共进午餐, 贾想1996—2008, 电影艺术（插图第8版）, 伟大的电影, 石挥谈艺录, WKW, 我心深处, 电影镜头设计, ﻿数学之美, 高数笔谈, 概率论与数理统计, 无言的宇宙, 古今数学思想（一）, 线性代数及其应用, 具体数学（英文版第2版）, 概率论沉思录, 高观点下的初等数学, 线性代数应该这样学, 贸易的猜忌, 君主论, 极权主义的起源, The King's Two Bodies, 变化社会中的政治秩序, 政治秩序的起源, 伯罗奔尼撒战争, 三十年细说从头 (上下), 寻找家园, 乡下人的悲歌, 写作这回事, 我坦言我曾历尽沧桑, 為了活下去：脫北女孩朴研美, 明朝那些事儿（壹）, 史记（全十册）, 哈佛中国史（全六卷）, 东晋门阀政治, 大秦帝国, 陈寅恪魏晋南北朝史讲演录, 三国志（全五册）, 潜规则, 失控, 加缪手记, 我的精神家园, 疯癫与文明, 传习录, 史记（全十册）, 声律启蒙, 易经, 论语, 菜根谭, 王阳明全集（全二册）, 禅说庄子：齐物论, 创造乡村音乐, 乐之本事, 怎样鉴别黄色歌曲, 西方文明中的音乐, 与小泽征尔共度的午后音乐时光, 有生之年非听不可的1001张唱片, 文学回忆录（全2册）, 人类的群星闪耀时, 天真的人类学家, 海错图笔记, 素履之往, 孤独之间, ﻿维特根斯坦传, 穷查理宝典, 人类的群星闪耀时, 鞋狗, 曾国藩（上中下）, 曾国藩家书, 我是你的男人, 梵高传, 小顾聊绘画·壹, 伯里曼人体结构绘画教学, 笔记大自然, ﻿长昼的安魂曲, 茶馆, 奥狄浦斯王, 石挥谈艺录, 莎士比亚全集（全8册）, 推销员之死, 芭巴拉少校, 雷雨, 萧伯纳戏剧选, ﻿艺术的故事, 疯狂艺术史：从达芬奇到伦勃朗, 脸的历史, 现代艺术150年, 孤独之间, 现代主义, 詹森艺术史（插图第7版）, 加德纳艺术通史, Sensuous Surfaces, 疯狂艺术史：从莫奈到毕加索, 照夜白, 风景与记忆, 艺术通史, 写给大家的西方美术史, 图说中国绘画史, 寂静之道, 佛祖都说了些什么?, 悉达多, 当和尚遇到钻石, 藏传佛教象征符号与器物图解, 五灯会元（全三册）, 西藏度亡经, 地藏菩萨本愿经, 第二次世界大战战史, 中国古代战争的地理枢纽, 滑铁卢, 苦难辉煌, 一战秘史, 麒麟之与子同袍, 竞逐富强, 野蛮大陆, 艾希曼在耶路撒冷, 巴黎烧了吗?, 邻人, 纳粹医生, 纯粹理性批判, 西方哲学史（上卷）, 逻辑哲学论, 存在与时间, 疯癫与文明, 悲剧的诞生, 叔本华思想随笔, 查拉图斯特拉如是说, 解体概要, 天国之秋, 袁氏当国, 剑桥中华民国史（上卷）, 科举改制与最后的进士, 记忆的政治, 花舞大唐春, 汉帝国的遗产, 汉代物质文化资料图说, 庞贝三日, 逝者的面具, 开放社会及其敌人（全二卷）, 致命的自负, 无政府、国家和乌托邦, 自由的伦理, 詹森艺术史（插图第7版）, 加德纳艺术通史, 梵高传, 伯里曼人体结构绘画教学, 清宫海错图, 世界美术名作二十讲, 陌生的经验, 如何看懂印象派, 剑桥艺术史：绘画观赏, 透视的艺术, 爱你就像爱生命, 永恒的时光之旅, 夜航西飞, 旅行与读书, 不去会死！, 小家，越住越大, 明天也是小春日和, 把小日子收进手帐里, 如花在野, 刻意练习, 自控力, 自醒：从被动努力到主动进步的转变, 异类, 你不用假装过得很好, 自控力, 逃避自由, ﻿永恒的时光之旅, 決鬥寫真論, 摄影构图学, 摄影师, 论摄影, 走来走去, 黑白, 一次・图片和故事, 家庭日记, 失焦, 新名字的故事, 灿烂千阳, 初夜, 夜莺, 爱这个世界, 高难度谈话, 小强升职记, 扛得住，世界就是你的, 演说之禅, 汪曾祺谈吃, 旅行者的早餐, 明日的便当, 随园食单, 日本料理神髓, 食物信息图, 鸭川食堂, 五味, 食物与厨艺, 文心, 正面管教, 如何说孩子才会听，怎么听孩子才肯说, 育儿基本, 傅雷家书, 永恒的时光之旅, 夜航西飞, 旅行与读书, 不去会死！, 新世界, 超越死亡, 生命之书, 当下的力量, 生命的轨迹, 人间是剧场, 简单冥想术：激活你的潜在创造力, 次第花开, 意识光谱, 事事本无碍, 性、生态、灵性, 脈輪全書, 人体的故事, 肠子的小心思, 女性健康私密书, 消失的微生物, 核心基础运动, 拉伸, 西尔斯健康育儿百科, 轻断食, ﻿外婆的道歉信, 情感暴力, 看不见的爱, 萤火虫小巷, 像我这样笨拙地生活, 折纸大全, 女孩子的手编小饰品111款, 手作族一定要会的缝纫基本功, 刺绣针法百种简史与示范, 橡皮章中毒, 大人的科学：流光幻彩折纸灯, 最详尽的缝纫教科书, 爱上有缝纫机的日子, 缝纫基础知识, haru的橡皮章生活, 从零开始学缝纫, 把刺绣变简单, 青木和子的唯美刺绣, 皮革工藝. vol.9 基礎技法篇, 大模型：改变世界的十大机械, 幸福的婚姻, 魔鬼聊天术, 性学观止（上下册）, 極致挑逗：雙人共撫全圖解120招, 搞定男人, 男人不屌女人不High, 爱的五种能力, 遵生八笺, 问中医几度秋凉, 人体使用手册, 营养圣经, 范志红：不懂健康，难做美丽女人, 中国健康调查报告, 养生书, 神奇的肌肤能量书, 瑜伽, 全注全译黄帝内经(上下), 谈话的力量, 高难度谈话, 关键对话, 爱是一种选择, 沃顿商学院最实用的谈判课（原书第2版）, 高效能人士的7个习惯（人际关系篇）, 人际沟通技巧, ﻿小家，越住越大, 何为美好生活：Beams at Home, BEAMS AT HOME 2, 最简单的小户型收纳整理术, 家事的撫慰（上冊）, 收纳全书, ﻿出国自助游教室, 路上有微光, 越南自助游, 台北旅遊全攻略, 雪山乌托邦, 青海/藏羚羊自助旅行手册, 明天就去尼泊尔, 国富论, 资本论（第一卷）, 致命的自负, 创业维艰, 卓有成效的管理者, 从0到1, 重新定义公司, 金字塔原理, 旁观者, 第五项修炼, 商业模式新生代, 国富论, 资本论（第一卷）, 致命的自负, 创业维艰, 从0到1, 重新定义公司, 优势谈判, 鞋狗, ﻿聪明的投资者, 金融炼金术, 货币战争, 货币金融学, 大而不倒, 门口的野蛮人, ﻿穷查理宝典, 聪明的投资者, 日本蜡烛图技术, 金融炼金术, 炒股的智慧, 随机漫步的傻瓜, 一个广告人的自白, 参与感, 跨越鸿沟, 不做无效的营销, 超级符号就是超级创意, ﻿创业维艰, 从0到1, 鞋狗, The Innovators, 商业模式新生代, 创新公司, 重新定义公司, 四步创业法, 创新与企业家精神, 三双鞋, ﻿聪明的投资者, 财务自由之路, 炒股的智慧, 巴菲特的护城河, 财报就像一本故事书, ﻿一个广告人的自白, Neil French, 超级符号就是超级创意, 文案训练手册, 奥格威谈广告, 蔚蓝诡计, 中兴百货广告作品全集, 洞见, 日本蜡烛图技术, 炒股的智慧, 公司价值分析, 海龟交易法则, 联想风云, 艰难的辉煌, 塞氏企业传奇, 活动策划完全手册, 疯传, 獨立策展人Independent Curators, 蔚蓝诡计, 赖声川的创意学, 广告媒体策划, 展览实践手册, 叶茂中的营销策划, 活动策划全攻略, 地理学与生活 (全彩插图第11版), 哥德尔、艾舍尔、巴赫, 花朵的秘密生命, 杂草记, 生命的跃升, 数学之美, ﻿失控, 创业维艰, 运营之光, 从0到1, 重新定义公司, 启示录, 信息简史, 长尾理论, 科技想要什么, 硅谷之谜, 代码大全（第2版）, 深入理解计算机系统, JavaScript高级程序设计（第3版）, C++ Primer 中文版（第 4 版）, Python编程：从入门到实践, Fluent Python, 程序员修炼之道, 哥德尔、艾舍尔、巴赫, 控制论与科学方法论, 七堂极简物理课, 数学之美, ﻿启示录, Designing Interactions, Web信息架构(第3版), Web表单设计, ﻿启示录, Designing Interactions, Web信息架构(第3版), Web表单设计, 数学之美, 深入理解计算机系统, 数据结构与算法分析, Pattern Recognition And Machine Learning, Algorithms to Live By, TCP/IP详解 卷1：协议, Fluent Python, ﻿Head First HTML与CSS、XHTML（中文版）, JavaScript语言精粹, 网络是怎样连接的, 图解HTTP, 你不知道的JavaScript（上卷）, Getting Real, Deep Web File #網絡奇談, PHP和MySQL Web开发(原书第4版), 深入剖析Tomcat, Web前端黑客技术揭秘, 从0到1, 信息简史, 重新定义公司, 科技想要什么, 硅谷之谜, 文明之光（第四册）, 星际唱片, 科技之巅, Web表单设计, Designing Interactions, ﻿通信之道——从微积分到5G, 大话无线通信, 大话通信, 信号与系统, 大话移动通信, 业余无线电通信, 通信原理, 通信原理, 通信行业求职宝典, 射频和无线技术入门, 通信新读, Fundamentals of Wireless Communication, 信息简史, 通信之美, 数据可视化之美, The Nature of Code, 交互式培训, Web表单设计, Understanding Your Users, Handbook of Usability Testing, 随心所欲, Letting Go of the Words, ﻿Neural Networks and Deep Learning, 神经网络与机器学习（原书第3版）, 意识的宇宙, Neural Networks for Pattern Recognition, MATLAB神经网络30个案例分析, 模式识别与神经网络, 神经网络在应用科学和工程中的应用, 神经网络与深度学习, Neural Networks, 神经网络模型及其MATLAB仿真程序设计, 深度学习, Embodiment and the Inner Life, 脑的高级功能与神经网络, MATLAB神经网络原理与实例精解, 计算机程序设计艺术（第1卷）, 高效程序的奥秘, Head First Python（中文版）, C专家编程, 机器学习系统设计, Dive Into Python 3, 代码大全（第2版）, Programming Ruby中文版, 梦断代码, 代码整洁之道, JAVASCRIPT语言精髓与编程实践
========================================
Cluster 1 details:
--------------------
Key features: ['本书', '历史', '生活', '爱情', '先生']
book in this cluster:
飘, 我们仨, 红楼梦, ﻿沉默的大多数, 我们仨, 时间的果, 民主的细节, 人间草木, 撒哈拉的故事, 当我谈跑步时我谈些什么, 沉默的大多数, 明朝那些事儿（1-9）, 时间的果, 人间草木, 红楼梦, 飘, 挪威的森林, ﻿我们仨, 撒哈拉的故事, 时间的果, 人间草木, 生活，是很好玩的, 当我谈跑步时我谈些什么, 挪威的森林, 诗的八堂课, 小毛驴与我, ﻿沉默的大多数, 红拂夜奔, 寻找无双·东宫西宫, ﻿沉默的大多数, 常识, 退步集, 我执, 写在人生边上 人生边上的边上 石语, 普通读者, ﻿红楼梦, 金瓶梅, 陶庵梦忆 西湖梦寻, 诗词会意：周汝昌评点中华好诗词, 大好河山可骑驴, 叶嘉莹说汉魏六朝诗, ﻿红楼梦, 飘, 傲慢与偏见, 荆棘鸟, 查特莱夫人的情人, 我们生活在巨大的差距里, ﻿夹边沟记事, 小说课, 平原上的摩西, 红拂夜奔, 写在人生边上 人生边上的边上 石语, 管锥编（全五册）, 谈艺录, 钱钟书散文, 围城 / 人·兽·鬼, 容安馆札记, 钱锺书英文文集, 钱锺书手稿集•外文笔记（第一辑）, 钱钟书杨绛散文, ﻿飘, 傲慢与偏见, 安娜·卡列尼娜（上下）, 朝花夕拾, 故事新编, 魏晋风度及其他, ﻿红楼梦, 金瓶梅, 陶庵梦忆 西湖梦寻, 诗词会意：周汝昌评点中华好诗词, 大好河山可骑驴, 叶嘉莹说汉魏六朝诗, 异端的权利, 巴西：未来之国, 被背叛的遗嘱, 杜拉斯谈杜拉斯, 抵挡太平洋的堤坝, 平静的生活, 夏夜十点半钟, 半小时漫画中国史（修订版）, 无人生还, 清明上河图密码, 一個人住第5年, 阿狸·梦之城堡, ﻿挪威的森林, 基地, 完美爱情标本店, 清明上河图密码, 清明上河图密码4, 剑桥简明金庸武侠史, 通稿2003, 走进大文豪的家, 清明上河图密码4, ﻿撒哈拉的故事, 荷西 我爱你, 温柔的夜, 稻草人手记, 送你一匹马, 闹学记, 我的灵魂骑在纸背上, 明朝那些事儿（1-9）, 鬼吹灯之精绝古城, 素年锦时, 七月与安生, 八月未央, 且以永日, 素年锦时之月棠记, 小时代1.0折纸时代, 左手倒影，右手年华。, 金庸随想录, 连城诀, 剑桥简明金庸武侠史, 无人生还, 听幾米唱歌, 基地, 每天早上和你一起醒来, 我执, 常识, 迷楼, 原来你非不快乐, 听幾米唱歌, ﻿一個人住第5年, 笑红尘, 你的怪兽男友, 草样年华, 夹边沟记事, 明朝那些事儿（1-9）, 亲密关系（第5版）, 爱的艺术, 进化心理学, 爱的艺术, 学习做一个会老的人, 狂热分子, 造房子, 造房子, 西文字体, 狂热分子, 民主的细节, 民主的细节, 通往奴役之路, ﻿造房子, 城记, 建筑：形式、空间和秩序, 中国古代建筑史, 正见, 耶路撒冷三千年, 思考的乐趣, 民主的细节, 狂热分子, 历史的终结及最后之人, ﻿我们仨, 被淹没和被拯救的, 时光列车, 记忆小屋, 夹边沟记事, 中国通史, 南明史, 姚著中国史, 魏晋之际的政治权力与家族网络, ﻿沉默的大多数, 进化心理学, 正见, 经典里的中国, 黄帝内经, 管锥编（全五册）, 来自民间的叛逆, 如果这可以是首歌, 人间草木, 诗的八堂课, 伊莎贝拉, 陈寅恪的最后20年, 中央公园西路, 哈姆莱特, 莎士比亚四大悲剧, 正见, 佛教常识答问, 布局天下, 隐形军队, 战争改变历史, 西洋世界军事史（全三卷）, 1944：腾冲之围, 像自由一样美丽, 单向度的人, 战争与革命中的西南联大, 帝国夕阳, ﻿神祇、陵墓与学者, ﻿通往奴役之路, 容忍与自由, 一课经济学, 法律、立法与自由(第一卷), 自由论, 资本主义与自由, 美国自由的故事, 飘, 亲密关系（第5版）, 爱的艺术, 挪威的森林, 傲慢与偏见, 荆棘鸟, 带一本书去巴黎, 撒哈拉的故事, 我已与一万亿株白桦相逢, 西班牙旅行笔记, 乖，摸摸头, 最好金龟换酒, 理想的下午, 当我谈跑步时我谈些什么, 学习做一个会老的人, 亲密关系（第5版）, 亲密关系（第5版）, 爱的艺术, 哈佛商学院谈判课, 进化心理学, 时间的果, 如何让女人免于心碎, 雅舍谈吃, 日日之食：家常菜的小秘密, 好妈妈胜过好老师, 孩子：挑战, 带一本书去巴黎, 撒哈拉的故事, 我已与一万亿株白桦相逢, 西班牙旅行笔记, 乖，摸摸头, 最好金龟换酒, 理想的下午, 一味, 吃货的生物学修养, 爱的艺术, 每天早上和你一起醒来, 爱的五种语言, 性别战争, 亲密关系（第5版）, 完美关系的秘密, 金赛性学报告, ﻿黄帝内经, 亲密关系（第5版）, 爱的五种语言, 收纳的艺术, 家, 家, 北欧风格小屋, 装修不上当，省心更省钱（新）, 通往奴役之路, 伟大的博弈, 定位, 通往奴役之路, 伟大的博弈, 哈佛商学院谈判课, 定位, 伟大的博弈, 定位, 定位, 大量流出, 众病之王, Java编程思想 （第4版）, UNIX环境高级编程, 交互设计之路, 交互设计之路, 什么是科学, 交互设计之路, 交互设计之路, UCD火花集2, 程序开发心理学
========================================
Cluster 2 details:
--------------------
Key features: ['一个', '故事', '女人', '小说', '世界']
book in this cluster:
巨人的陨落, 我的前半生, 百年孤独, 世界的凛冬, 月亮和六便士, 斯通纳, ﻿百年孤独, 巨人的陨落, 月亮和六便士, 世界的凛冬, 斯通纳, ﻿百年孤独, 巨人的陨落, 世界的凛冬, 月亮和六便士, 江城, 亲爱的安德烈, 皮囊, 白鹿原, ﻿百年孤独, 巨人的陨落, 世界的凛冬, 小王子, 刀锋, 永恒的边缘, 白夜行, 我的职业是小说家, 不可思议的朋友, 怒, 我口袋里的星辰如沙砾, 皮囊, 孩子你慢慢来, ﻿我的职业是小说家, 奥克诺斯, 小王子, 夏洛的网, 当世界年纪还小的时候, 蓝熊船长的13条半命, 天蓝色的彼岸, 爱德华的奇妙之旅, 彼得·潘, 安吉拉·卡特的精怪故事集, 格林童话全集, 思维的乐趣, 似水流年, 思维的乐趣, 野火集, ﻿不可思议的朋友, 窗边的小豆豆, 我亲爱的甜橙树, 夏洛的网, 蓝熊船长的13条半命, 当世界年纪还小的时候, 爱德华的奇妙之旅, 天蓝色的彼岸, 彼得·潘, 月亮和六便士, 百年孤独, 了不起的盖茨比, 简爱, 罪与罚, 老人与海, 约翰·克利斯朵夫, 第七天, 没有一条道路是重复的, 红玫瑰与白玫瑰, 金锁记, 小团圆, 张爱玲作品集, 尘埃落定, 蒙着眼睛的旅行者, 丙申故事集, 一个陌生女人的来信, 简爱, 老人与海, 双城记, 羊脂球, 等待戈多, 故事新编, 一个陌生女人的来信, 一个陌生女子的来信, 恐惧, 昨日之旅, 茨威格文集, 茨威格小说集, 一个陌生女人的来信, 蒙田, ﻿不能承受的生命之轻, 生活在别处, 好笑的爱, 玩笑, 庆祝无意义, 身份, 琴声如诉, 劳儿之劫, 直布罗陀水手, 爱, 无耻之徒, 重读, 纽约客, ﻿白夜行, 东方快车谋杀案, 猎豹, 雪人, 半落, 不可思议的朋友, 我怎样毁了我的一生, 松风, 你看起来好像很好吃, 我口袋里的星辰如沙砾, 一个人的好天气, 陪安东尼度过漫长岁月, 少年巴比伦, 1995-2005夏至未至, 白夜行, 沉睡的人鱼之家, 祈祷落幕时, 单恋, 永恒的终结, 球状闪电, 三生三世 十里桃花, 华胥引（全二册）, 那个不为人知的故事, 他知道风从哪个方向来, 浮生若梦1, ﻿白夜行, 猎豹, 雪人, 怒, 香水, 达·芬奇密码, 夜色人生, 鹿鼎记（全五册）, 英雄志（全三册）, 华胥引（全二册）, 異變者1, 一座城池, 很高兴见到你, 像少年啦飞驰, 三重门, 零下一度, ﻿魔道祖师, 桃花债, 又一春, 山河表里, ﻿我的前半生, 流金岁月, 如果墙会说话, 吃南瓜的人, ﻿白夜行, 东方快车谋杀案, 怒, 夜色人生, 半落, 首无·作祟之物, 哭泣的骆驼, 万水千山走遍, 我的宝贝, 倾城, 三生三世 十里桃花, 那个不为人知的故事, 华胥引（全二册）, 魔道祖师, 彼岸花, 春宴, 二三事, 告别薇安, 瞬间空白, ﻿1995-2005夏至未至, 爱与痛的边缘, 天亮说晚安, 夏至未至, 爵迹：雾雪零尘, 庆余年·壹, 醉玲珑（上卷）, 凤囚凰（上中）, 新宋, 新宋·十字1, 窃明, 鹿鼎记（全五册）, 侠客行, ﻿东方快车谋杀案, ABC谋杀案, 悬崖山庄奇案, 我不是完美小孩, 地下铁, 我的心中每天开出一朵花, 走向春天的下午, 躲进世界的角落, 微笑的鱼, 遗失了一只猫, 永恒的终结, 球状闪电, 索拉里斯星, 我口袋里的星辰如沙砾, 我偏爱那些不切实际的浪漫, 少年巴比伦, 愿你迷路到我身旁, 那个不为人知的故事, 1995-2005夏至未至, 停靠，一座城, 请你永远记得我, 松风, 魔戒前传, 我的前半生, 喜宝, 霸王别姬 青蛇, 我不是完美小孩, 地下铁, 我的心中每天开出一朵花, 走向春天的下午, 躲进世界的角落, 微笑的鱼, 遗失了一只猫, 哈利·波特与被诅咒的孩子, 罪恶生涯, 一个人旅行2, 人气绘本天后高木直子作品典藏（全6册）, 一个人的第一次 第一次一个人旅行, 一个人去跑步, 一个人住第几年？, 一个人漂泊的日子2, 一个人的美食之旅2, 天涯·明月·刀, 楚留香传奇系列, 白玉老虎（上下）, 羽·青空之蓝, 镜·织梦者, 剩者为王Ⅱ, 文艺风象·毕业即胜利, 17, 鲤·嫉妒, 鲤·变老, 誓鸟, 鲤·荷尔蒙, 聋哑时代, 三重门, 忽而今夏, 我在等，等风等你来, 巨人的陨落, 世界的凛冬, 枪炮、病菌与钢铁, 天才在左 疯子在右, 如果没有今天，明天会不会有昨天？, 如果没有今天，明天会不会有昨天？, 禅与摩托车维修艺术, 历史深处的忧虑, 枪炮、病菌与钢铁, 枪炮、病菌与钢铁, 日本新中产阶级, 做衣服, 枪炮、病菌与钢铁, 江城, 历史深处的忧虑, 日本新中产阶级, 历史深处的忧虑, 独裁者手册, 穆斯林的葬礼, 达·芬奇密码, 剧本, 异形全书：经典四部曲终极档案, 魔鬼数学, 独裁者手册, 巨流河, 上学记, 中国大历史, 辛丰年音乐笔记, 中国大历史, 褚时健传, 一个无政府主义者的意外死亡, 等待戈多, 莎乐美, 欲望号街车, 玩偶之家, 自制美学, 亮剑, 魔鬼之师SAS, ﻿世界的凛冬, 零年：1945, 五个人的战争, 如果没有今天，明天会不会有昨天？, 奢华之色, 哈耶克文选, 常识, ﻿我的前半生, 一个陌生女人的来信, 平如美棠, 穆斯林的葬礼, 喜宝, 爱情和其他魔鬼, 江城, 岛上书店, 如果没有今天，明天会不会有昨天？, 平如美棠, 我口袋里的星辰如沙砾, 皮囊, 遇见未知的自己, 我口袋里的星辰如沙砾, 我的前半生, 皮囊, 喜宝, 偷影子的人, 所谓会说话，就是会换位思考, 富爸爸，穷爸爸, 精进, 滚蛋吧!肿瘤君, 别让将来的你，讨厌现在不理智的自己, 活法, 天才在左 疯子在右, 美国纽约摄影学院摄影教材（上）, 被人遗忘的人：中国精神病人生存状况, 直到长出青苔, ﻿我的前半生, 长恨歌, 另一座城, ﻿深度工作, 杜拉拉升职记, 至味在人间, 亲爱的安德烈, 窗边的小豆豆, 孩子的宇宙, 江城, ﻿遇见未知的自己, 观呼吸, 这书能让你戒烟, 乳房, 喜宝, 我偏爱那些不切实际的浪漫, 爱得太多的女人, 原谅石, 请你永远记得我, 世上另一个我(修订版), ﻿巧手折一折, 最简单的家庭木工, ﻿如何在爱中修行, 吸引力是这样炼成的, 卡内基沟通与人际关系, 最简单的家庭木工, 青海全攻略, 我把欧洲塞进背包, 沙发旅行, Lonely Planet旅行指南系列：韩国（2013年全新版）, 精益创业, 精益创业, 富爸爸，穷爸爸, 世界上最伟大的推销员, 品牌洗脑, 精益创业, 富爸爸，穷爸爸, 小狗钱钱, 工作前5年，决定你一生的财富, 通向财务自由之路, 广告文案训练手册, 创意, 创意的生成, 史丹·温斯坦称傲牛熊市的秘密, 光变：一个企业及其工业史, 下一个倒下的会不会是华为, 汽车和我, 迪斯尼战争, 绩效致死, 非常营销：娃哈哈--中国成功的实战教案, ﻿创意, 会讲故事，让世界听你的, 时尚：幕后的策动, 精益创业, 程序员的自我修养, 上帝的手术刀, 那些古怪又让人忧心的问题what if?, 程序员的自我修养, 文明之光（第一册）, 支付战争, 程序员的自我修养
========================================
Cluster 3 details:
--------------------
Key features: ['设计', '用户', 'web', '本书', '体验']
book in this cluster:
认识电影, 经济学原理（上下）, 认识电影, 纽约无人是客, 写给大家看的设计书（第3版）, 素描的诀窍, 写给大家看的设计书（第3版）, 简约至上, 版式设计原理, 设计的觉醒, 深泽直人, 平面设计中的网格系统, 认知与设计, 街道的美学, ﻿认识电影, 离散数学及其应用（原书第5版）, 认识电影, ﻿素描的诀窍, ﻿素描的诀窍, 纽约无人是客, 纽约无人是客, 摄影构图与色彩设计, Excel图表之道, 纽约无人是客, 住宅设计解剖书, 装修设计解剖书, 家的模样, ﻿经济学原理（上下）, ﻿经济学原理（上下）, 产品经理的第一本书, 编码, C程序设计语言, 算法导论（原书第2版）, 集体智慧编程, 简约至上, 点石成金, About Face 3 交互设计精髓, 认知与设计, 破茧成蝶：用户体验设计师的成长之路, 人人都是产品经理, 设计师要懂心理学, Web界面设计, 交互设计沉思录, 触动人心, 瞬间之美, 简约至上, 点石成金, About Face 3 交互设计精髓, 认知与设计, 破茧成蝶：用户体验设计师的成长之路, 人人都是产品经理, 设计师要懂心理学, Web界面设计, 交互设计沉思录, 触动人心, 瞬间之美, 编码, 算法导论（原书第2版）, 集体智慧编程, C程序设计语言, 编译原理, HTML & CSS设计与构建网站, JAVASCRIPT权威指南(第四版), Flask Web开发：基于Python的Web应用开发实战, JavaScript DOM编程艺术, 高性能JavaScript, 构建高性能Web站点, 形式感+：网页视觉设计创意拓展与快速表现, 锦绣蓝图, Web之困：现代Web应用安全指南, ﻿点石成金, About Face 3 交互设计精髓, 破茧成蝶：用户体验设计师的成长之路, 简约至上, 交互设计沉思录, Web界面设计, Designing Interfaces中文版, 最佳设计100细则, 移动应用UI设计模式, 用户体验度量, 形式感+：网页视觉设计创意拓展与快速表现, 用户体验面面观, 写给大家看的设计书, 无线通信, 数据通信与网络, ﻿About Face 3 交互设计精髓, 认知与设计, 破茧成蝶：用户体验设计师的成长之路, 交互设计沉思录, 触动人心, 用户体验草图设计, 在你身边，为你设计, 用户体验面面观, 决胜UX, 设计调研, 移动设计, 亲爱的界面, UCD火花集, 用户体验草图设计, 用户体验面面观, 设计沟通十器, UCD火花集, 一目了然, 设计网事, 人机交互：以用户为中心的设计和评估, 就这么简单, 重塑用户体验, 体验设计白书, 神经计算原理, 神经网络, Matlab神经网络与应用, JavaScript DOM编程艺术, C程序设计语言, Effective java 中文版（第2版）, 高性能网站建设指南, CSS禅意花园
========================================
Cluster 4 details:
--------------------
Key features: ['我们', '作者', '自己', '本书', 'isbn']
book in this cluster:
追风筝的人, 囚鸟, 追风筝的人, 囚鸟, 活着为了讲述, 追风筝的人, 囚鸟, 活着为了讲述, 送你一颗子弹, 浮生六记, 这些人，那些事, 送你一颗子弹, 海边的卡夫卡, 来自新世界 （上下）, 咖啡未冷前, 这些人，那些事, 浮生六记, 穿过圩场, 悲伤与理智, 海边的卡夫卡, 鲍勃·迪伦诗歌集 (1961-2012), 博尔赫斯诗选, 现实与欲望, 给孩子的诗, 失物之书, 绿毛水怪, 理想国与哲人王, 呐喊, 角儿, 自由在高处, 写在人生边上, 理想国与哲人王, 浮生六记, 三国演义（全二册）, 唐宋词十七讲, 三国演义（全二册）, 呼啸山庄, 在细雨中呼喊, 现实一种, 我没有自己的名字, 黄昏里的男孩, ﻿半生缘, 棋王, 苍老的指甲和宵遁的猫, 人情之美, 城门开, 写在人生边上, 呼啸山庄, ﻿呐喊, 鲁迅杂文全集, 热风, 浮生六记, 三国演义（全二册）, 唐宋词十七讲, 昨日的世界, 一个陌生女人的来信, 慢, 物质生活, 爱，谎言与写作, 寂寞的十七岁, 我在伊朗长大, 再一次, 海贼王, 塔希里亚故事集, S., ZOO, 秘密花园, 朝花惜时, 最好的我们, 此间的少年, 那些回不去的年少时光, 余生，请多指教, 微微一笑很倾城, 幻夜, 虚无的十字架, 你一生的故事, 来自新世界 （上下）, 微微一笑很倾城, S., 七夜雪, 吉尔莫·德尔·托罗的奇思妙想, 塔希里亚故事集, 精灵宝钻, 海贼王, 浪客行 1, 可爱的洪水猛兽, 告白与告别, 不可抗力 上, 關照你的花蕾, 玫瑰的故事, S., 13．67, 敲响密室之门, 此间的少年, 眠空, 大方 No.2, 刻下来的幸福时光, 绾青丝Ⅰ, 歌尽桃花, 渴望死亡的小丑, 告诉我，你怎样去生活, 奉命谋杀, 星空, 幾米創作10年精選, 布瓜的世界, 你一生的故事, ﻿朝花惜时, 最好的我们, 那些回不去的年少时光, 我知道你会来，所以我等, 拂过冬季到云来, 最初不过你好, 精灵宝钻, 13．67, 任你行, 青蛇, 星空, 幾米創作10年精選, 布瓜的世界, 神奇的魁地奇球, 一个人住的每一天, 30分老妈, ﻿七夜雪, 剩者为王, 文艺风象·闪开！让专业不对口的来, 鲤·暧昧, 鲤·最好的时光, 鲤·因爱之名, 葵花走失在1890, 鲤·偶像, ﻿最好的我们, 此间的少年, 微微一笑很倾城, 拂过冬季到云来, 我们纯真的青春, 大国的崩溃, 十二幅地图中的世界史, 天朝的崩溃, 非暴力沟通, 人生的智慧, ﻿活着为了讲述, 昨日的世界, 奇特的一生, 伍迪·艾伦谈话录, 送你一颗子弹, 十二幅地图中的世界史, 知觉之门, 忧郁的热带, 你一定爱读的极简欧洲史, ﻿未来简史, 最好的告别, 忧郁的热带, ﻿设计中的设计, 设计心理学, 最好的告别, 大国的崩溃, 美丽新世界, 我的应许之地, 利维坦, 毛泽东选集 第一卷, 我的应许之地, 金刚经说什么, 雕刻时光, 伍迪·艾伦谈话录, 吉尔莫·德尔·托罗的奇思妙想, 怎样解题, 从一到无穷大, 利维坦, 联邦党人文集, 活着为了讲述, 昨日的世界, 人情之美, 见证, 天朝的崩溃, 穿过圩场, 智识分子, 哲学的故事, 毛泽东选集 第一卷, 莫扎特与纳粹, 未来简史, 世界因你不同, 秘密花园, 30天学会绘画, 蒋公的面子, 佛教的见地与修道, 学佛三书（共3册）, 雪域求法记, 进击的局座, 战略, 奥斯维辛, 第三帝国的兴亡（上中下）, ﻿人生的智慧, 哲学的故事, 人性的，太人性的, 谈谈方法, ﻿天朝的崩溃, 自由选择, 自由秩序原理, 联邦党人文集, 扭曲的人性之材, 30天学会绘画, 如何看一幅画, 泛若不系之舟, 旅行的艺术, 无声告白, 人生的智慧, 囚徒健身, ﻿追风筝的人, 无声告白, 把时间当作朋友, 被讨厌的勇气, 钓鱼的男孩, 把时间当作朋友, 活出生命的意义, 布鲁克林有棵树, 拆掉思维里的墙, 把力气花在你想要的生活上, 秘密, 非暴力沟通, 24个比利, 我们时代的神经症人格, 醒来的女性, 只有医生知道!, 他们说，我是幸运的, 提问的力量, 职场动物进化手册, 学会提问, 品尝的科学, 四季便当, 厨房里的人类学家, 优秀的绵羊, 情感依附, 泛若不系之舟, 旅行的艺术, 改变，从心开始, 秘密, 亲密关系, 一念之转, ﻿只有医生知道!, 囚徒健身, 无器械健身, 硬派健身, 病者生存, 非暴力沟通, 莉莉和章鱼, 恋人絮语, 我有个恋爱想和你谈下, 彼得·科恩木工基础, 玩转手工100招, 我们为什么会分手？, 如何让你爱的人爱上你, 求医不如求己, 零起点学中医, 我要内脏变年轻, 黄帝内经·四气调神, 顾中一说：我们到底应该怎么吃？, 脸要穷养，身要娇养（全新图文修订版）, 呼吸之间, 瑜伽之光, 不要用爱控制我, 人情与面子, 我好-你好, 家庭收纳1000例, 老火车的时光慢游, 意大利, 伦敦, 激荡三十年（上）, 行动的勇气, 自由选择, 怪诞行为学, 激荡三十年（下）, MBA教不了的创富课, 激荡三十年（上）, 行动的勇气, 自由选择, 怪诞行为学, 激荡三十年（下）, 激荡三十年（上）, 智能时代, MBA教不了的创富课, 激荡三十年（下）, 成为乔布斯, 行动的勇气, 投资最重要的事, 公司理财, 金融的逻辑, 彼得·林奇的成功投资, 投资最重要的事, 手把手教你读财报, 怎样选择成长股, 说谎者的扑克牌, 引爆点, 无价, 顾客为什么购买, MBA教不了的创富课, 街头生意经, 精益创业实战, 彼得·林奇的成功投资, 投资最重要的事, 手把手教你读财报, 怎样选择成长股, 股市进阶之道, 广告文案, 我们在为什么样的广告买单, ﻿彼得·林奇的成功投资, 手把手教你读财报, 怎样选择成长股, 股市进阶之道, 笑傲股市, 走出幻觉走向成熟, ﻿激荡三十年（上）, 激荡三十年（下）, 跌荡一百年（下）, 规模与范围, 广告与促销, 創意筆記本: 概念力, 时间简史, 上帝掷骰子吗, 从一到无穷大, 超越时空, 自私的基因, 未来简史, 智能时代, 游戏改变世界, 大数据时代, 上瘾, Python编程快速上手, 时间简史, 上帝掷骰子吗, 从一到无穷大, 超越时空, 自私的基因, 万物简史, 万万没想到, 时间的形状, 万物起源, 设计心理学, 设计心理学, 编程之美, ﻿智能时代, 迷人的材料, 无人驾驶, 设计心理学, Facebook效应, 随意搜寻, 应需而变, 实用程序育儿法
========================================
Cluster 5 details:
--------------------
Key features: ['法国', '地球', '国家', '著名', '一部']
book in this cluster:
1984, 1984, 二手时间, 1984, 二手时间, 看见, 黄金时代, 冷暴力, 國史大綱（上下）, 智惠子抄, 智惠子抄, 恶之花, ﻿最后的精灵, 黄金时代, 古文观止, 基督山伯爵, 悲惨世界（上中下）, 红与黑, 基督山伯爵, 悲惨世界（上中下）, 红与黑, 巴黎圣母院, 包法利夫人, 古文观止, 笑忘录, 无知, 告别圆舞曲, 雅克和他的主人, ﻿情人, 长别离, 玛格丽特·杜拉斯, 封闭式车库, 波丽娜, 地图（人文版）, 三体Ⅲ, 三体Ⅱ, 银河系漫游指南, 沙丘, 海伯利安, 神们自己, 封闭式车库, 三体全集, 11处特工皇妃, 直到那一天, 笑傲江湖（全四册）, 11处特工皇妃, 临界·爵迹 I, 临界·爵迹 Ⅱ, ﻿11处特工皇妃, 笑傲江湖（全四册）, 三体Ⅱ, 银河系漫游指南, 三体Ⅲ, 沙丘, 海伯利安, 神们自己, 三体全集, 仿生人会梦见电子羊吗？, 神奇动物在哪里, 神奇动物在哪里, ﻿人类简史, 二手时间, 國史大綱（上下）, 大英博物馆世界简史（全3册）, 冷暴力, 乌合之众, 对伪心理学说不, 人类简史, 理想国, 第二性, ﻿人类简史, 北欧，冰与火之地的寻真之旅, 智惠子抄, 大英博物馆世界简史（全3册）, 人类简史, 二手时间, 乌合之众, 看见, 中国国家治理的制度逻辑, 荒野之歌, 大英博物馆世界简史（全3册）, 渴望风流, ﻿人类简史, 二手时间, 乌合之众, 1984, 北欧，冰与火之地的寻真之旅, ﻿1984, 论美国的民主, 旧制度与大革命, 僧侣与哲学家, ﻿论美国的民主, 中国国家治理的制度逻辑, 旧制度与大革命, 国家的常识, 我的前半生, 國史大綱（上下）, 僧侣与哲学家, 國史大綱（上下）, 古文观止, ﻿人类简史, 北欧，冰与火之地的寻真之旅, 我的前半生, 渴望风流, 僧侣与哲学家, 无畏之海, 特拉法尔加战役, 命运攸关的抉择, 所有我们看不见的光, 我是女兵,也是女人, 理想国, 智惠子抄, 情人, 北欧，冰与火之地的寻真之旅, 在漫长的旅途中, 冷暴力, 乌合之众, 对伪心理学说不, 荒野之歌, 第二性, 爱弥儿, 北欧，冰与火之地的寻真之旅, 在漫长的旅途中, 巴黎, 投资中最简单的事, 投资中最简单的事, 投资中最简单的事, 投资中最简单的事, 地球之美, 人类简史, 地图（人文版）, 万物：创世, 对伪心理学说不, 万物：创世, 对伪心理学说不, 地球之美
========================================
Cluster 6 details:
--------------------
Key features: ['中国', '公司', '社会', '著名', '本书']
book in this cluster:
霍乱时期的爱情, 围城, 霍乱时期的爱情, 霍乱时期的爱情, 我为你洒下月光, 围城, 平凡的世界（全三部）, 北鸢, 活着本来单纯, 我为你洒下月光, 倾城之恋, 围城, 平凡的世界（全三部）, 山海经全译, 中国历代政治得失, 活着本来单纯, 我为你洒下月光, 摇摇晃晃的人间, 青铜时代, 东宫·西宫, 佛祖在一号线, 给孩子的故事, 山海经全译, 人间词话, 山海经, 四世同堂, 倾城之恋, 流言, 第一炉香, 怨女, 北鸢, 望春风, ﻿围城, 钱锺书手稿集•中文笔记, 钱钟书选集·散文卷，小说诗歌卷, 旧文四篇, 阿Q正传, 伤逝, 鲁迅与当代中国, 鲁迅全集(1), 山海经全译, 人间词话, 山海经, 天工开物·栩栩如真, 画的秘密, 嫌疑人X的献身, 福尔摩斯探案全集（上中下）, 画的秘密, 塞拉菲尼抄本, 嫌疑人X的献身, 偏爱你的甜, 嫌疑人X的献身, 多情剑客无情剑(上、中、下), 塞拉菲尼抄本, 青春, 嫌疑人X的献身, 金庸散文集, 我有一切的美妙, 多情剑客无情剑(上、中、下), 男生贾里  女生贾梅, 万历十五年, 中国历代政治得失, 叫魂, 邓小平时代, 乡土中国, 乡土中国, 叫魂, 画的秘密, 日本的八个审美意识, 乡土中国, 寻路中国, 中国历代政治得失, 邓小平时代, 穿墙透壁, 中国建筑史, 华夏意匠, 图像中国建筑史, 空谷幽兰, 金刚经, 八万四千问, 论自由, ﻿万历十五年, 中国历代政治得失, 中国近代史, 中国历代政治得失, 中国文化的深层结构, 论自由, 中国哲学简史, ﻿人间词话, 中国古代文化常识, 中国文化要义, 说文解字, 给孩子的故事, 李鸿章传, 硅谷钢铁侠, 如何读中国画, 沿着塞纳河到翡冷翠, 隔江山色, 如何读中国画, 隔江山色, ﻿金刚经, 楞严经, 维摩诘经, 活着回来的男人, 中国近代史, 中国1945, 李鸿章传, 中国近代史, 近代中国社会的新陈代谢, 潮来潮去, 档案中的历史, 美术、神话与祭祀, 宗子维城, 从历史中醒来, 黄泉下的美术, 中国天文考古学, 中国古代物质文化, 中国青铜时代, 唐风吹拂撒马尔罕, 暗流, 何以中国, 白沙宋墓, 中国古代壁画 唐代, 论自由, 自由与繁荣的国度, 中国美术史讲座, 霍乱时期的爱情, 倾城之恋, ﻿寻路中国, 沿着塞纳河到翡冷翠, 我为你洒下月光, 高效能人士的七个习惯（精华版）, 火车上的中国人, Photoshop修色圣典, 我爱这哭不出来的浪漫, 昨天的中国, 记忆的性别, 给孩子的故事, ﻿寻路中国, 沿着塞纳河到翡冷翠, 中国居民膳食指南, 中国古代房内考, 人情、面子与权力的再生产, 这样装修不后悔（插图修订版）, 100元狂走中国, 中国自助游, 2011中国自助游全新彩色升级版, 中国古镇游, 大国大城, 中央帝国的财政密码, 高效能人士的七个习惯（精华版）, 大败局, 大国大城, 中央帝国的财政密码, ﻿浪潮之巅, 腾讯传, 大败局, 硅谷钢铁侠, 信用创造、货币供求与经济结构, 价值评估, 巴菲特致股东的信, 价值评估, 股市真规则, 消费者行为学 （第8版·中国版）, 市场营销原理, 增长黑客, 史玉柱自述, 增长黑客, 硅谷钢铁侠, 腾讯传, 解读基金, 巴菲特致股东的信, 股市真规则, 巴菲特致股东的信, 腾讯传, 大败局, 跌荡一百年（上）, 华为的世界, 中国的大企业, 解构德隆, 广告人手记, 电视节目策划笔记, 浪潮之巅, 必然, 腾讯传, 增长黑客, Web性能权威指南, 必然, 硅谷钢铁侠, 文明之光 （第三册）, 文明之光（第二册）
========================================
Cluster 7 details:
--------------------
Key features: ['小说', '日本', '作家', '奇幻', '影响力']
book in this cluster:
双峰: 神秘史, 戴上手套擦泪, 戴上手套擦泪, 灯塔, 双峰: 神秘史, 下雨天一个人在家, 繁花, 台北人, 一句顶一万句, 恋情的终结, 东京本屋, 人间失格, 下雨天一个人在家, 火花, 世界尽头与冷酷仙境, 一个人的村庄, 所谓好玩的事，我再也不做了, 世界尽头与冷酷仙境, 没有色彩的多崎作和他的巡礼之年, 且听风吟, 奇鸟行状录, 二十亿光年的孤独, 黑铁时代, 黑铁时代, 幻想图书馆, 绿山墙的安妮, 许三观卖血记, 一句顶一万句, 繁花, 象棋的故事, 情感的迷惘, 小说的艺术, 阿涅丝的最后一个下午, 欲望玫瑰, 写作, 小说稗类, 鳄鱼手记, 荒人手记, 遗产, 深夜食堂 01, 方向, 睡魔1：前奏与夜曲, 阿兰的战争, 守望者 上, 一日谈, 幽灵, 恶意, 夏日启示录, 恶意, 浪花少年侦探团, 新参者, 伽利略的苦恼, 流星之绊, 神经漫游者, ﻿有匪1, 双峰: 神秘史, 幽灵, 螺旋之谜, ﻿有匪1, 有匪3, 冰与火之歌（卷一）, 冰与火之歌, 草原动物园, 睡魔1：前奏与夜曲, 克苏鲁神话, 魔戒, 二代目归来, 睡魔2：玩偶之家, 魔法坏女巫, 萤火之森, 大哥, 恶意, 夏日启示录, 占星术杀人魔法, 有匪1, 天行健（全4册）, 深白, 梦里花落知多少, 小时代1.5·青木时代VOL.1, 金庸江湖志, 金庸十二钗, 雪山飞狐, 春宵苦短，少女前進吧！, 圖書館戰爭, 我的错都是大人的错, 我的世界都是你, 神经漫游者, 深渊上的火, ﻿冰与火之歌（卷一）, 冰与火之歌, 魔法坏女巫, 魔戒, 我的错都是大人的错, 我的世界都是你, 一个人的42公里, 一个人的美食之旅, 镜·神寂, 镜·龙战, 镜·辟天, 如果声音不记得, 鲤·孤独, 樱桃之远, 鲤·逃避, 影响力, 灯塔, 方向, 天外有天, 天才的编辑, 东京本屋, 菊与刀, 菊与刀, 走进建筑师的家, 一九八四·动物农场, 走进建筑师的家, 外部空间设计, 天外有天, 山月记, 菊与刀, 天空的另一半, 毛姆传, 天才的编辑, 一幅画开启的世界, 名画之谜·希腊神话篇, 佛陀, 再度觉醒, 阿兰的战争, 拥抱战败, 若非此时，何时？, 一幅画开启的世界, 恋情的终结, 情书, 罗摩桥, 东京本屋, 有匪1, 你自以为的极限，只是别人的起点, 影响力, 使女的故事, 天空的另一半, 四季家之味, 深夜食堂 01, 罗摩桥, 你是我的命运, 大人的科学：浪漫四季星空灯, 女醫師教你真正愉悅的性愛, 走进世界最美的家, 住宅读本, ﻿影响力, 影响力, 股票作手回忆录, 股票作手回忆录, ﻿影响力, 股票作手回忆录, 编码的奥秘, 编码的奥秘
========================================
Cluster 8 details:
--------------------
Key features: ['沟通', '有时', '是否', '不是', '因为']
book in this cluster:
活着, 杀死一只知更鸟, 杀死一只知更鸟, 活着, 杀死一只知更鸟, ﻿活着, 杀死一只知更鸟, 强风吹拂, 爷爷变成了幽灵, ﻿活着, 爷爷变成了幽灵, 何以笙箫默, 何以笙箫默, 何以笙箫默, 少有人走的路, 沟通的艺术（插图修订第14版）, 杀死一只知更鸟, 安野光雅的十二堂绘画课, 安野光雅的十二堂绘画课, ﻿活着, 杀死一只知更鸟, 少有人走的路, 沟通的艺术（插图修订第14版）, ﻿少有人走的路, 强风吹拂, 少有人走的路, 沟通的艺术（插图修订第14版）, 沟通圣经, 说话就是生产力, 杀死一只知更鸟, 游戏力, ﻿沟通的艺术（插图修订第14版）, 沟通圣经, 强势, 非暴力沟通实践篇, 用户体验的要素, 赢在用户, 用户体验的要素, 赢在用户, 用户体验的要素, 赢在用户, 赢在用户, ﻿用户体验的要素, 赢在用户
========================================
Cluster 9 details:
--------------------
Key features: ['美国', '推荐', '哈利', '编辑', '波特']
book in this cluster:
房思琪的初戀樂園, 雪落香杉树, 雪落香杉树, 房思琪的初戀樂園, 雪落香杉树, 步履不停, 一只特立独行的猪, 心理学与生活, 1Q84 BOOK 1, 1Q84 BOOK 1, 1Q84 BOOK 3, 月光落在左手上, 七夜物语, 一只特立独行的猪, 一只特立独行的猪, 七夜物语, 动物凶猛, 我可以咬一口吗, 我可以咬一口吗, 生活三部曲：生活的样子, 动物凶猛, 雪落香杉树, 金庸师承考, 碟形世界：猫和少年魔笛手, 美国众神, 金庸师承考, 雪落香杉树, 碟形世界：猫和少年魔笛手, 哈利·波特与魔法石, 哈利·波特与阿兹卡班的囚徒, 哈利·波特与火焰杯, 哈利·波特与死亡圣器, 哈利·波特与密室, 哈利·波特与混血王子, 哈利·波特与凤凰社, ﻿哈利·波特与魔法石, 哈利·波特与阿兹卡班的囚徒, 哈利·波特与火焰杯, 哈利·波特与死亡圣器, 哈利·波特与密室, 哈利·波特与混血王子, 哈利·波特与凤凰社, 诗翁彼豆故事集, 哈利•波特(共6册) (精装), 哈利·波特百科全书, 哈利·波特百科全书, 有生之年, 黎明破晓的世界, 海洋与文明, ﻿社会心理学, 心理学与生活, 社会性动物, 为什么学生不喜欢上学?, 大问题, 规训与惩罚, 社会契约论, 作为意志和表象的世界, 史蒂夫·乔布斯传, 只是孩子, 音乐使人自由, 梵高传, 社会心理学, 社会性动物, 社会学的想像力, 大都会艺术博物馆指南, 聆听音乐, 认识艺术（全彩插图第8版）, 超越平凡的平面设计, 房思琪的初戀樂園, 社会性动物, 社会心理学, 社会学的想像力, 我心狂野, 世界电影史, 什么是数学, 数学：确定性的丧失, 微积分学教程（第一卷）, 数学, 社会契约论, 文明的冲突与世界秩序的重建, 山居杂忆（插图精装版）, 黎明破晓的世界, 一只特立独行的猪, 规训与惩罚, 作为意志和表象的世界, 庄子, ﻿音乐与情感, 聆听音乐, 音乐使人自由, 听音乐（插图第6版）, 怎样欣赏音乐, 古典音乐简单到不行！, 地下乡愁蓝调, 认识乐理, 古尔德读本, 古典风格, 全球通史（第7版 上册）, 大问题, 我也有一个梦想, 梵高传, 富兰克林自传, 永山裕子的水彩课, 像艺术家一样思考, 大都会艺术博物馆指南, 阿部智幸的水彩笔记, 永山裕子的水彩课, 水彩入门, 简·海恩斯的写意水彩, 大都会艺术博物馆指南, 认识艺术（全彩插图第8版）, 金刚经 心经 坛经, ﻿血战钢锯岭, 大问题, 作为意志和表象的世界, 规训与惩罚, 什么是批判？/ 自我的文化, 重说中国近代史, 与废墟为伴, 永山裕子的水彩课, 像艺术家一样思考, 梵高传, 面纱, 厨艺的常识, 心理学与生活, 靠谱, 人性的弱点全集, ﻿社会心理学, 心理学与生活, 社会性动物, 为什么学生不喜欢上学?, 房思琪的初戀樂園, 向前一步, 面纱, 靠谱, 向前一步, 零秒工作, 横向领导力, 斜杠创业家, 权力与领导（第5版）, 内向者沟通圣经, 12个工作的基本, ﻿厨艺的常识, 食帖·真的，烤箱什么都能做, 民国太太的厨房, ﻿为什么学生不喜欢上学?, 大学之路（套装）, 孩子，把你的手给我, 与神对话（第一卷）, 酸痛拉筋解剖书, 吃的真相, 植物性饮食革命, 男人来自火星 女人来自金星, 男人来自火星 女人来自金星, 亚当夏娃在拂晓, 诫律, 植物性饮食革命, 这样装修省大钱（插图修订版）, 认识商业, 合适, 经济学通识, 策略思维, 经济学的思维方式（第11版）, 摩根财团, 认识商业, 创新者的窘境, 营销管理, 靠谱, 管理的实践, 10人以下小团队管理手册, 认识商业, 合适, 经济学通识, 策略思维, 经济学的思维方式（第11版）, 摩根财团, 认识商业, 创新者的窘境, 摩根财团, 证券分析, 期货市场技术分析, 漫步华尔街, 一本书读懂财报, 证券分析, 期货市场技术分析, 漫步华尔街, 营销管理, 畅销的原理, 文案创作完全手册, 零售的哲学：7-Eleven便利店创始人自述, 黑客与画家, 创新者的窘境, 重来, 中产阶级如何保护自己的财富, 学会花钱, 个人理财, The Copy Book 全球32位顶尖广告文案的写作之道, 文案创作完全手册, 计算广告, 文案发烧, 我的广告生涯&科学的广告, 战胜华尔街, 专业投机原理, 股票大作手操盘术, 证券分析, 看得见的手, 文案发烧, 编辑力, 编辑人的世界, ﻿极简宇宙史, 什么是数学, 黑客与画家, ﻿计算机程序的构造和解释, 算法（第4版）, 编程珠玑, 黑客与画家, ﻿极简宇宙史, 什么是数学, ﻿计算机程序的构造和解释, 机器学习, 算法（第4版）, 编程珠玑, 人工智能的未来, 极简宇宙史, 月亮, 信息论基础, 数字通信, 数字通信, 数字通信, ﻿"""笨办法""学Python"
========================================
```

