#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
# @Author: StudentCWZ
# @Date:   2019-11-28 10:33:34
# @Last Modified by:   StudentCWZ
# @Last Modified time: 2019-11-28 10:46:39
'''


class RMM(object):
	def __init__(self):
		self.window_size = 3

	def cut(self, text):
		result = []
		index = len(text)
		dic = ['研究','研究生','生命','命','的','起源']
		while index > 0:
			for size in range(index - self.window_size, index):
				piece = text[size:index]
				if piece in dic:
					index = size + 1
					break
			index = index - 1
			result.append(piece+'----')
		result.reverse()
		return result


if __name__ == '__main__':
	text = '研究生命的起源'
	tokenizer = RMM()
	print(tokenizer.cut(text))
