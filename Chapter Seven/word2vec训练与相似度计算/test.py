#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-28 11:08:36
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-28 11:08:51
'''


import gensim
def my_function():

    model = gensim.models.Word2Vec.load('./data/zhiwiki_news.word2vec')
    print(model.similarity('西红柿','番茄'))  #相似度为0.63
    print(model.similarity('西红柿','香蕉'))  #相似度为0.44

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))

if __name__ == '__main__':
    my_function()
