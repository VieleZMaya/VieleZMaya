#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/30 - 2021/7/2
# @Author  : VieleZMaya（杨诗晓 1120192621）
# @FileName: Pretreatment.py
# @Version : 0.4.1
import os
from nltk.corpus import wordnet as wn
os.chdir("4/Files")
# 对训练集文件和测试集文件均做处理
files = ["TRAIN_FILE.TXT", "TEST_FILE_FULL.TXT"]
with open("wordList.txt", 'w') as wf:
    wordList = []
    hypernyms = {}
    for file in files:
        with open(file, 'r') as f:
            fds = f.readlines()
        if "TRAIN" in file:
            outFile = 'trainFile.txt'
        else:
            outFile = 'testFile.txt'
        with open(outFile, "w") as f:
            word_dic = {}
            # 对于输入的每一行
            for i, line in enumerate(fds):
                # 句子所在的一行
                if(i % 4) == 0:
                    # 将句子编号写入文件
                    f.write("{}\n".format(eval(line[:line.index('\t')])))
                    # 替换实体中的空格，以防发生错误
                    e1 = line[line.index("<e1>")+4: line.index("</e1>")]
                    e2 = line[line.index("<e2>")+4: line.index("</e2>")]
                    if ' ' in e1:
                        word = e1.replace(' ', '*')
                        line = line[: line.index("<e1>")+4] + \
                            word + line[line.index("</e1>"):]
                    if ' ' in e2:
                        word = e2.replace(' ', '*')
                        line = line[: line.index("<e2>")+4] + \
                            word + line[line.index("</e2>"):]
                    # 将完整英文句子写入文件
                    sentence = line[line.index("\""): -3].replace('""','"').replace(':','').replace('." ', '').replace('"', '').replace("<e1>", '').replace("</e1>\'s", '').replace("</e1>\'", '').replace(
                        "</e1>", "").replace("</e2>\'s", '').replace("<e2>", '').replace("</e2>", '').replace(',','').replace(';','') + '\n'
                    if sentence[-2] == '.':
                        sentence = sentence[:-2] + '\n'
                    f.write(sentence)
                    # 去除双引号和句号并分词
                    line = line[: -3].replace(':','').replace('(', '').replace(')', '').replace("\"",'').split(' ')
                    sentence = []
                    entity = []
                    index = []
                    words = []
                    index_e1 = 0
                    index_e2 = 0
                    # 对于句子中的每一个词
                    for i, word in enumerate(line):
                        # 剔除第一个词之前的Tab和前双引号
                        if "\t" in word:
                            word = word[word.index("\t")+1:]
                        # 是实体
                        if '<e' in word:
                            # 对实体1 进行操作
                            if "e1" in word:
                                if '.' in word:
                                    word = word.replace('.', ' ')
                                word = word[word.index("<e1>")+4:word.index("</e1>")].replace(' ','*')
                                # 将实体加入列表并标记位置
                                entity.append(word)
                                index_e1 = i
                                wnitem = wn.synsets(word)
                                if len(wnitem) > 1:
                                    if len(wnitem[1].hypernyms()) > 0:
                                        strr = str(wnitem[1].hypernyms())
                                        hypernyms[word] = strr[strr.index('\'')+1:strr.index('.')]
                            # 实体2 操作同上
                            elif "e2" in word:
                                if '.' in word:
                                    word = word.replace('.', ' ')
                                word = word[word.index(
                                    "<e2>")+4:word.index("</e2>")].replace(' ','*')
                                entity.append(word)
                                index_e2 = i
                                wnitem = wn.synsets(word)
                                if len(wnitem) > 1:
                                    if len(wnitem[1].hypernyms()) > 0:
                                        strr = str(wnitem[1].hypernyms())
                                        hypernyms[word] = strr[strr.index('\'')+1:strr.index('.')]
                        # 将词加入列表以便后续记录位置
                        word = word.replace(',','').replace('(','').replace(')','')
                        if word != '':
                            if word[0] in ['\'', '.']:
                                word = word[1:]
                            word = word.replace('"','').replace(';','').replace('!','').replace(':','')
                        if word != '':
                            words.append(word)
                    # 记录相对位置并写入文件
                    for i, word in enumerate(words):
                        index.append("{} {},{} ".format(word, int(i - index_e1), int(i - index_e2)))
                    for word in index:
                        f.write("{}".format(word))
                        wordList.append(word[: word.index(' ')])
                    f.write('\n')
                    # 将实体1和实体2写入文件
                    f.write('{},{} {},{}\n'.format(entity[0], entity[1],index_e1, index_e2))
                # 将关系写入文件
                elif(i % 4) == 1:
                    f.write(line)
    for word in wordList:
        wf.write(word + '\n')
    with open('hypernyms.txt', 'w') as hf:
        for word in hypernyms:
            hf.write("{} {}\n".format(word, hypernyms[word]))

        
