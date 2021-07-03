#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 - 2021/7/2
# @Author  : VieleZMaya（杨诗晓 1120192621）
# @FileName: Extracting.py
# @Version : 1.5.5
import os
from torch.nn.modules.pooling import MaxPool1d
from GettingEmbeddings import getEmbedding
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
os.chdir("4/Files")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract():
    # SemEval2010 T8 中要预测的19种关系
    relations = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)', 'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)', 'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)', 'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)', 'Product-Producer(e1,e2)',
                 'Product-Producer(e2,e1)', 'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)', 'Content-Container(e1,e2)', 'Content-Container(e2,e1)', 'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)', 'Other']
    # 提取词向量
    embeddings = getEmbedding()
    print("Extract Processing...")
    files = ["trainFile.txt", "testFile.txt"]
    maxlength = 0
    for file in files:
        with open(file, 'r') as f:
            fds = f.readlines()
        # 提取单词，相对位置，实体及其位置，关系和最大句子长度
        sentences, indexs, entities, labels, max_length = readIn(
            fds, embeddings, relations)
        if maxlength < max_length:
            maxlength = max_length
        in_size = len(embeddings['UNK'])
        total_size = len(sentences)
        wf_pf3 = np.zeros((total_size, maxlength, int(in_size * 3 + 2)))
        lexi3 = np.zeros((total_size, int(6 * in_size)))
        for i, sentence in enumerate(sentences):
            # len sentence:10~32
            # len sentences:8000
            wf_pf2 = np.zeros((maxlength, int(in_size * 3 + 2)))
            lexi2 = np.zeros(int(6 * in_size))
            # 提取每一句话对应的WF-PF向量
            for j, word in enumerate((sentence)):
                wf_pf1 = np.zeros(int(in_size * 3 + 2), dtype=float)
                wf_pf1[:len(embeddings[word])] = embeddings[word]
                wf_pf1[len(embeddings[word]): int(2 * len(embeddings[word]))] = (indexs[i][word][0] / len(sentences[i])) * embeddings[entities[i][0]]
                wf_pf1[int(2 * len(embeddings[word])): int(3 * len(embeddings[word]))] = (indexs[i][word][1] / len(sentences[i])) * embeddings[entities[i][1]]
                wf_pf1[int(3 * len(embeddings[word])): int(3 * len(embeddings[word])) + 2] = list(indexs[i][word])
                wf_pf2[j] = wf_pf1
            wf_pf3[i] = wf_pf2
            # 提取每一句话对应的Lexical Features
            lexi2[: in_size] = embeddings[entities[i][0]]
            lexi2[in_size: int(2 * in_size)] = embeddings[entities[i][1]]
            for j in range(2, 4):
                index = eval(entities[i][j])
                if index > 0:
                    lexi2[int((j * 2 - 2) * in_size): int((j * 2 - 1)* in_size)] = embeddings[sentences[i][index - 1]]
                else:
                    lexi2[int((j * 2 - 2) * in_size) : int((j * 2 - 1) * in_size)] = embeddings[sentences[i][index]]
                if index < len(sentences[i]) - 1:
                    lexi2[int((j * 2 - 1) * in_size) : int((j * 2) * in_size)] = embeddings[sentences[i][index + 1]]
                else:
                    lexi2[int((j * 2 - 1) * in_size) : int((j * 2) * in_size)] = embeddings[sentences[i][index]]
            lexi3[i] = lexi2
        # 转换成pytorch张量
        sentence_x = torch.FloatTensor(wf_pf3)
        lexical_x = torch.FloatTensor(lexi3)
        if file == 'trainFile.txt':
            # torch.Size([8000, 85, 17])
            train_sentence_x = sentence_x
            train_lexical_x = lexical_x
            train_y = torch.LongTensor(labels)
        else:
            # torch.Size([2717, 60, 17])
            test_sentence_x = sentence_x
            test_lexical_x = lexical_x
            test_y = torch.LongTensor(labels)
    return train_sentence_x, train_lexical_x, train_y, test_sentence_x, test_lexical_x, test_y


def readIn(fds, embeddings, relations):
    # 对从处理好的文本中提取相应数据
    sentences = []
    indexs = []
    entities = []
    count_relation = 0
    len_line = 5  # 一个周期为5行
    labels = []
    max_length = 0
    for i, line in enumerate(fds):
        line = line[:-1]
        # 获取每个词，以及对应实体的相对位置
        if i % len_line == 2:
            items = line.split(' ')
            word = ''
            index = {}
            sentence = []
            for i, item in enumerate(items):
                if i % 2 == 0:
                    word = item
                    if word != '':
                        sentence.append(word)
                else:
                    index[word] = eval(item)
            indexs.append(index)
            max_length = max(max_length, len(sentence))
            sentences.append(sentence)
        # 实体1和实体2和对应位置
        elif i % len_line == 3:
            items = line.split(' ')
            temp = []
            for item in items:
                temp.extend(item.split(','))
            entities.append(tuple(temp))
        # 关系
        elif i % len_line == 4:
            labels.append(relations.index(line))
    # len of training, testing : 8000, 2717
    return sentences, indexs, entities, labels, max_length


#extract()
