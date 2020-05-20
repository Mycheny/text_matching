import csv
import os
import sys

import gensim
import jieba

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import re
from esim_tencent.graph import Graph
import tensorflow as tf
import numpy as np
from esim_tencent import args
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import platform


def jieba_add_words(f):
    for ln in tqdm(f, desc="jieba add words"):
        if len(ln) < 2 or len(ln) > 5:
            continue
        line = ln.strip()
        if not isinstance(line, jieba.text_type):
            try:
                line = line.decode('utf-8').lstrip('\ufeff')
            except UnicodeDecodeError:
                raise ValueError('dictionary file %s must be utf-8')
        if not line:
            continue
        # match won't be None because there's at least one character
        word, freq, tag = jieba.re_userdict.match(line).groups()
        if freq is not None:
            freq = freq.strip()
        if tag is not None:
            tag = tag.strip()
        jieba.add_word(word, freq, tag)


if platform.system() == 'Windows':
    ChineseEmbedding_path = r"E:/DATA/tencent/ChineseEmbedding.bin"
elif platform.system() == 'Linux':
    ChineseEmbedding_path = r"/stor/wcy/tencent/ChineseEmbedding.bin"
else:
    sys.exit(0)


def w2v(word, model):
    try:
        return model.wv[word]
    except:
        return np.zeros(args.word_embedding_size)


def shuffle(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences

    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。

    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值

    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def word_index(p_sentences, h_sentences, idx2word):
    word2idx = {v: i for i, v in enumerate(idx2word)}

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.seq_length)
    h_list = pad_sequences(h_list, maxlen=args.seq_length)

    return p_list, h_list


# 加载char_index、静态词向量、动态词向量的训练数据
def load_all_data(path, index2word, data_size=None):
    df = pd.read_csv(path, sep='\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    loc1 = label == 0
    loc2 = label == 1
    p = np.concatenate((p[loc1], p[loc2]))
    h = np.concatenate((h[loc1], h[loc2]))
    label = np.concatenate((label[loc1], label[loc2]))
    pro = label.sum() / (1 - label).sum()
    if pro < 0.5:
        p = p[int(len(p[loc1]) * (1 - pro)):]
        h = h[int(len(h[loc1]) * (1 - pro)):]
        label = label[int(len(label[loc1]) * (1 - pro)):]

    p, h, label = shuffle(p, h, label)

    p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p))
    h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h))

    p_w_index, h_w_index = word_index(p_seg, h_seg, index2word)

    return p_w_index, h_w_index, label


if __name__ == '__main__':
    data_types = ["ATEC", "CCKS", "LCQMC"]
    data_type = data_types[0]
    print(data_type)
    data_path = f"../input/data/{data_type}/processed"
    vectors_model = gensim.models.KeyedVectors.load(ChineseEmbedding_path, mmap='r')
    num = 500000
    index2word = vectors_model.index2word[:num]
    vectors = vectors_model.vectors[:num]
    del vectors_model  # 删除变量 释放内存
    jieba_add_words(index2word)
    p, h, y = load_all_data(f'{data_path}/train.tsv', index2word, data_size=None)
    p_eval, h_eval, y_eval = load_all_data(f'{data_path}/dev.tsv', index2word, data_size=1000)

    p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
    h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
    y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

    dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
    dataset = dataset.batch(args.batch_size).repeat(args.epochs)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    model = Graph(embedding=vectors)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config)as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})
        steps = int(len(y) / args.batch_size)
        for epoch in range(args.epochs):
            for step in range(steps):
                p_batch, h_batch, y_batch = sess.run(next_element)
                _ = sess.run([model.train_op, ], feed_dict={model.p: p_batch,
                                                            model.h: h_batch,
                                                            model.y: y_batch,
                                                            model.keep_prob: args.keep_prob})
                if step % 10 == 0:
                    loss, acc, confusion_matrix = sess.run([model.loss, model.acc, model.confusion_matrix],
                                         feed_dict={model.p: p_batch,
                                                    model.h: h_batch,
                                                    model.y: y_batch,
                                                    model.keep_prob: args.keep_prob})
                    print('epoch:', epoch, ' step:', f"{step}/{steps}", ' loss: ', loss, ' acc:', acc, 'cm\n', confusion_matrix)

            loss_eval, acc_eval, confusion_matrix_eval = sess.run([model.loss, model.acc, model.confusion_matrix],
                                           feed_dict={model.p: p_eval,
                                                      model.h: h_eval,
                                                      model.y: y_eval,
                                                      model.keep_prob: 1})
            print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval, 'cm\n', confusion_matrix_eval)
            print('\n')
            saver.save(sess, f'../output/esim_tencent/{data_type}/esim_tencent_{data_type}_{epoch}.ckpt')
