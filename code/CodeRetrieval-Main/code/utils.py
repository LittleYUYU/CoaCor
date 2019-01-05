import numpy as np
import time
import math
import torch

def cos_np(data1, data2):
    """numpy implementation of cosine similarity for matrix"""
    dotted = np.dot(data1, np.transpose(data2))
    norm1 = np.linalg.norm(data1, axis=1)
    norm2 = np.linalg.norm(data2, axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data / np.linalg.norm(data, axis=1).reshape((data.shape[0], 1))
    return normalized_data


def dot_np(data1, data2):
    """cosine similarity for normalized vectors"""
    return np.dot(data1, np.transpose(data2))


#######################################################################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s' % (asMinutes(s), asMinutes(rs))


#######################################################################

def sent2indexes(sentence, vocab):
    '''sentence: a string
       return: a numpy array of word indices
    '''
    return np.array([vocab[word] for word in sentence.strip().split(' ')])


########################################################################

use_cuda = torch.cuda.is_available()


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


########################
# Metric Calculations ##
########################

def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def MAP(real, predict):
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))


def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg