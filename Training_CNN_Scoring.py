#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/7/2
# @Author  : VieleZMaya（杨诗晓 1120192621）
# @FileName: Training_CNN_Scoring.py
# @Version : 7.6.12
from torch._C import dtype
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from Extracting import extract
import torch
import torch.nn as nn
import numpy as np
import os
import re
import gc
import time
from torch.nn import init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.metrics import f1_score, accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Cnn(nn.Module):
    # 初始化模型及其参数
    def __init__(self, in_channels, out_channels, num_output, max_sentence, hidden_size=10, num_layers=1):
        super(Cnn, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.conv1d = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.mp1 = nn.MaxPool1d(kernel_size=max_sentence, stride=1)
        self.lstm = nn.LSTM(out_channels, hidden_size,
                            num_layers, batch_first=True)
        self.linear1 = nn.Linear(out_channels, 150)
        self.linear2 = nn.Linear(int(2 * (in_channels - 2) + 150), num_output)

    def forward(self, sentence_x, lexical_x):
        # torch.Size([8000, 85, 150])
        sentence_x = sentence_x.permute(0, 2, 1)
        # torch.Size([8000, 150, 85])
        '''第一步 卷积'''
        out_conv = self.conv1d(sentence_x)
        # torch.Size([8000, 150, 85])
        '''第二步 池化'''
        out_pooling = self.mp1(out_conv).permute(2, 0, 1).squeeze(0)
        # torch.Size([8000, 150])
        '''第三步 LSTM（加了反而精确度下降，放弃）'''
        """ out_pooling = out_pooling.reshape(-1, 1, self.out_channel)
        out_lstm, _ = self.lstm(out_pooling)  # size (seq_len, batch, input_size)
        seq, btc, hds = out_lstm.shape  # out_lstm is output, size (seq_len, batch, hidden_size)
        out_lstm = out_lstm.view(seq*btc, hds) """
        '''第四步 全连接 #1 + tanh'''
        out_w2 = self.linear1(out_pooling)
        # torch.Size([8000, 150])
        out_sentence = torch.tanh(out_w2)
        '''第五步 sentence features 和 lexical features拼接'''
        extrac_feature = torch.cat((out_sentence, lexical_x), dim=1)
        '''第六步 全连接 #2'''
        out_union = F.relu(self.linear2(extrac_feature))
        return out_union


def main():
    # 开始时间
    print("Start at:", end="")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 读入训练集和测试集
    train_sentence_x, train_lexical_x, train_y, test_sentence_x, test_lexical_x, test_y = extract()
    # 设置参数
    in_channels = train_sentence_x.shape[2]  # 特征向量大小
    max_sentence = train_sentence_x.shape[1]  # 最长句子长度
    # train_sentence_x.size : torch.Size([8000, 85, 17])
    out_channels = 112 # 顶着6G显存上限跑
    learning_rate = 5e-5
    num_output = int(max(train_y).item()) + 1  # 19
    num_epochs = 150000
    # 实例化CNN模型
    model = Cnn(in_channels, out_channels, num_output, max_sentence)
    # 初始化模型参数
    init.normal_(model.linear2.weight, mean=1, std=0.001)
    init.constant_(model.linear2.bias, val=0)
    # 定义损失函数（交叉熵损失函数）
    criterion = CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # 开始训练
    print("Start training...")
    f1 = train(model, train_sentence_x, train_lexical_x, train_y,
               test_sentence_x, test_lexical_x, test_y, criterion, optimizer, num_epochs)
    print("Final test f1 {:.4f}%".format(f1 * 100))
    print("End at:", end = "")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 以下开始训练过程


def train(model, train_sentence_x, train_lexical_x, train_y, test_sentence_x, test_lexical_x, test_y, criterion, optimizer, num_epochs):
    # 初始化用于作图的记录表
    epoch_list = []
    loss_list = []
    train_f1_list = []
    test_f1_list = []
    max_f1 = 0
    # 把东西丢到显卡上整
    model = model.to(device)
    train_sentence_x = train_sentence_x.to(device)
    train_lexical_x = train_lexical_x.to(device)
    train_y = train_y.to(device)
    test_sentence_x = test_sentence_x.to(device)
    test_lexical_x = test_lexical_x.to(device)
    test_y = test_y.to(device)
    # 训练
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_sentence_x, train_lexical_x)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        predict = model(train_sentence_x, train_lexical_x)
        f1_train, acc = f1_measure(predict, train_y)
        predict = model(test_sentence_x, test_lexical_x)
        f1_test, acc = f1_measure(predict, test_y)
        epoch_list.append(epoch)
        loss_list.append(train_loss)
        train_f1_list.append((f1_train))
        test_f1_list.append(f1_test)
        if f1_test > max_f1:
            max_f1 = f1_test
        if (epoch + 1) % 100 == 0:
            print("epoch {} train loss {} train f1 {:.4f}% test f1 {:.4f}% test acc {:.4f}%".format(
                epoch + 1, train_loss, f1_train * 100, f1_test * 100, acc * 100))
            if f1_train > 0.9:
                """ outs = F.log_softmax(model(test_sentence_x, test_lexical_x), dim=1)
                outs = outs.data.max(1)[1].tolist()
                print(outs)
                print(test_y.tolist()) """
                break
    # 作出F1变化图和Loss变化图
    plt.figure()
    plt.plot(epoch_list, train_f1_list, 'r', linewidth=2, label='f1_train')
    plt.plot(epoch_list, test_f1_list, 'b', linewidth=2, label='f1_test')
    # 设置图例
    plt.legend()
    title = 'CNN F1-measure'
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("F1-measure")
    plt.savefig('{}.png'.format(title))
    plt.close()
    print("F1 Diagram Created.")
    plt.figure()
    plt.plot(epoch_list, loss_list, 'r', linewidth=2, label="loss")
    title = "CNN Loss"
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig('{}.png'.format(title))
    plt.close()
    print("Loss Diagram Created.")
    return max_f1


def f1_measure(out, targets):
    out = F.log_softmax(out, dim=1)
    out = out.data.max(1)[1].tolist()
    f1 = f1_score(targets.cpu(), out, average='weighted')
    acc = accuracy_score(targets.cpu(), out)
    return f1, acc


main()
