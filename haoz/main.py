# -*- coding: utf-8 -*- 
# @Time 2020/4/13 14:46
# @Author wcy

from sklearn.decomposition import TruncatedSVD
import numpy as np
from wordfreq import word_frequency
import jieba

import time
import json
from collections import OrderedDict

from pathlib import Path


def init_dict(dictfile='tc_min.dict'):
    # jieba 加载自定义词典
    jieba.load_userdict(dictfile)
    jieba.enable_paddle()

    # 加载词频数据并返回
    domain = int(2 ** 31 - 1)
    freq_dict = {}
    with open(dictfile, 'r', encoding='utf8') as f:
        for line in f:
            segs = line.split(' ')
            token = segs[0]
            freq = int(segs[1])
        freq_dict[token] = float(freq / domain)

    return freq_dict

freq_dict = init_dict()


def get_word_frequency(word, freq_dict=freq_dict):
    if word in freq_dict:
        return freq_dict[word]
    else:
        return word_frequency(word, 'zh')


from annoy import AnnoyIndex


def init_index(annoy_indexfile='tc_index_build10.ann.index', word2indexfile='tc_word_index.json'):
    # 我们用保存好的索引文件重新创建一个Annoy索引, 单独进行加载
    annoy_index = AnnoyIndex(200)
    annoy_index.load(annoy_indexfile)

    with open(word2indexfile, 'r') as fp:
        word_index = json.load(fp)

    # 准备一个反向id==>word映射词表
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return annoy_index, word_index, reverse_word_index


annoy_index,word_index,reverse_word_index = init_index()


def AVG_embedding(line, embed_index=annoy_index, word2index=word_index, dim=200, pc=0):
    # start = time.time()

    word_list = [token for token in list(jieba.cut(line, use_paddle=True))
                 if token in word2index.keys()]

    # stop = time.time()

    # print("time for cut words = %.2f s" % (float(stop - start)))

    # start = time.time()
    sent_length = len(word_list)
    vs = np.zeros(dim)
    if not sent_length:
        return vs
    for token in word_list:
        vs += embed_index.get_item_vector(word2index[token])

    # stop = time.time()
    # print("time for calc avg vector = %.2f s" % (float(stop - start)))

    return vs / sent_length

AVG_embedding("喜欢打篮球的男生喜欢什么样的女生").shape

from numpy import array


def FREQ_embedding(line, embed_index=annoy_index, word2index=word_index, dim=200, a=1e-3, pc=0):
    # start = time.time()
    word_list = [token for token in list(jieba.cut(line, use_paddle=True))
                 if token in word2index.keys()]
    # stop = time.time()

    # print("time for cut words = %.2f s" % (float(stop - start)))

    sent_length = len(word_list)
    vs = np.zeros(dim)

    # start = time.time()

    if not sent_length:
        return vs
    for token in word_list:
        token_freq = get_word_frequency(token)
        a_value = a / (a + token_freq)
        vs += a_value * array(embed_index.get_item_vector(word2index[token]))

    # stop = time.time()
    # print("time for calc weighted vector = %.2f s" % (float(stop - start)))

    return vs / sent_length

FREQ_embedding("无线上网卡和无线路由器怎么用").shape