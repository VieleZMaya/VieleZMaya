import os
import collections
import math
import random
import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.chdir("4/Files")


# 读取词表，得到词典
def read_data():
    with open("wordList.txt", "r")as f:
        fds = f.readlines()

    raw_word_list = []
    for item in fds:
        item = item[:-1]
        raw_word_list.append(item)
    return raw_word_list


words = read_data()
print('Size of words ', len(words))

# 词典大小（不能多也不能少）
vocabulary_size = 26888


def build_dataset(words):
    # 词汇编码
    count = [['UNK', -1]]
    # 找到词频最大的前vocabulary_size个词，count中存储的是单词和数量
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count", len(count))
    dictionary = dict()
    # 给每个词一个编号
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary[word]
        # data和words对应，把词转换为下标
        data.append(index)
    # 反转字典 key为词汇编码 values为词汇本身
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 索引、要用于训练的词和对应数量、字典的反转
data, count, dictionary, reverse_dictionary = build_dataset(words)
data_index = 0
# 为skip-gram模型产生一个batch训练样本


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # (1, batch_size)
    labels = np.ndarray(shape=(batch_size, 1),
                        dtype=np.int32)  # (batch_size, 1)
    span = 2 * skip_window + 1  # [ left target right ]
    # 定义缓冲队列
    buffer = collections.deque(maxlen=span)

    # 遍历滑动窗口内的所有词语   左+中心词+右
    for i in range(span):
        # 依次取span个词放入缓冲队列中
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        # 缓冲区中心的目标标签
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# 构造模型
batch_size = 128
embedding_size = 150
skip_window = 2
num_skips = 2
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    # 输入要训练的词包
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # 权重矩阵
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 选取张量embeddings中对应train_inputs索引的值
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 转化变量输入，适配NCE
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    # 计算NCE损失的函数
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         inputs=embed,
                                         labels=train_labels,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 使用所学习的词向量来计算一个给定的 minibatch 与所有单词之间的相似度（余弦距离）
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # 添加变量初始值设定项
    init = tf.global_variables_initializer()

# 开始训练
with tf.Session(graph=graph) as session:
    # 将参数传入模型
    init.run()
    print("训练词向量中...")
    average_loss = 0
    for step in xrange(10):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # 通过评估优化器op来执行一个更新步骤
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
    final_embeddings = normalized_embeddings.eval()

# 输出词向量
with open('word2vec.txt', "w") as f:
    f.write(str(vocabulary_size) + ' ' + str(embedding_size) + '\n')
    for i in xrange(final_embeddings.shape[0]):
        sWord = reverse_dictionary[i]
        sVector = ''
        for j in xrange(final_embeddings.shape[1]):
            sVector = sVector + ' ' + str(final_embeddings[i, j])
        f.write(sWord + sVector + '\n')
print('词向量输出完成.')
