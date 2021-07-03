#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 - 2021/7/2
# @Author  : VieleZMaya（杨诗晓 1120192621）
# @FileName: GettingEmbeddings.py
# @Version : 0.0.8
""" import os
os.chdir("4/Files") """
import numpy as np
# 提取词向量
def getEmbedding() -> dict:
    print("Get Embedding...")
    with open(r"word2vec.txt","r") as f:
        fds = f.readlines()
    embeddings = {}
    # 对文件中每一行提取对应单词的向量
    for i, line in enumerate(fds):
        line = line[:-1].split(' ')
        if i == 0:
            length = eval(line[1])
        if i != 0:
            word = ''
            vec = np.zeros(length)
            for j, item in enumerate(line):
                if j == 0:
                    word = item
                else:
                    vec[j - 1] = eval(item)
            embeddings[word] = vec
    return embeddings
