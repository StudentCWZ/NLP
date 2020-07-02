# -*- coding: utf-8 -*-
# @Author: StudentCWZ
# @Date:   2019-12-09 15:46:56
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2020-02-03 12:02:26



# 导入第三方库
import glob
import random
import jieba



def get_content(path):
    '''读取语料'''
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l
        return content


def get_TF(words, topK=10):
    '''计算高频词'''
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key = lambda x: x[1], reverse=True)[:topK]




def stop_words(path):
    '''读取停用词'''
    with open(path) as f:
        return [l.strip() for l in f]



def main():
    files = glob.glob('/Users/mac/Desktop/Python自然语言处理实战/data/news/C000013/*.txt')
    corpus = [get_content(x) for x in files]

    sample_inx = random.randint(0, len(corpus)) # 获取随机样本语料

    # split_words = list(jieba.cut(corpus[sample_inx]))
    split_words = [x for x in jieba.cut(corpus[sample_inx]) if x not in stop_words('/Users/mac/Desktop/Python自然语言处理实战/data/stop_words.utf8')]# 过滤停用词，切分语料
    print('样本之一：'+corpus[sample_inx])
    print('样本分词效果：'+ '/ '.join(split_words))
    print('样本的topk(10)词：' + str(get_TF(split_words)))

if __name__ == '__main__':
    main()

