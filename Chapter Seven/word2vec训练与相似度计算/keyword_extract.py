#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2020-02-28 11:06:53
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-28 11:07:09
'''


import jieba.posseg as pseg
from jieba import analyse

def keyword_extract(data, file_name):
   tfidf = analyse.extract_tags
   keywords = tfidf(data)
   return keywords

def getKeywords(docpath, savepath):

   with open(docpath, 'r') as docf, open(savepath, 'w') as outf:
      for data in docf:
         data = data[:len(data)-1]
         keywords = keyword_extract(data, savepath)
         for word in keywords:
            outf.write(word + ' ')
         outf.write('\n')
