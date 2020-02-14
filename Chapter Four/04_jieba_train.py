#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-11-28 10:33:34
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-11-28 10:46:39
'''
import jieba.posseg as psg

sent = r'中文分词是文本处理不可或缺的一步！'
seg_list = psg.cut(sent)
print(''.join(['{0}/{1}'.format(w, t) for w,t in seg_list]))
