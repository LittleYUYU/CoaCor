import sys
import os
import random
import numpy as np
import pickle

import torch
import torch.nn.functional as F

from models import JointEmbeder
from configs import get_config

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


def limit_vocab(old_vocab_dict, vocab_length):
    new_vocab_dict = {}
    for word, key in old_vocab_dict.iteritems():
        if key < vocab_length:
            new_vocab_dict[word] = key
    return new_vocab_dict


def processed_dataset(list_of_strings):
    preprocessed_text = []
    for string in list_of_strings:
        words = map(int, string.split())
        preprocessed_text.append(np.array(words))
    return preprocessed_text


def create_model_name_string(c):
    string1 = 'qtlen_{}_qblen_{}_codelen_{}_qtnwords_{}_qbnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
        format(c['qt_len'], c['code_len'], c['qb_len'], c['qt_n_words'], c['qb_n_words'], c['code_n_words'],
               c['batch_size'], c['optimizer'], str(c['lr'])[2:])
    string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}_simmeasure_{}'. \
        format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:], c['sim_measure'])
    string = string1 + string2
    return string


def _reload_model(model, conf, model_string):
    conf['model_directory'] = conf['workdir'] + 'model/' + model_string + '/'
    assert os.path.exists(conf['model_directory'] + 'best_model.h5'), 'Weights for saved best model not found'
    model.load_state_dict(torch.load(conf['model_directory'] + 'best_model.h5'))


def pad_seq(seq, maxlen, pad_token=0):
    if len(seq) < maxlen:
        seq = np.append(seq, [pad_token] * maxlen)
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
    return seq


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def process_text(input_string, vocab, unk_token=3):
    """
    Convert a given text string into np array for indices
    :param input_string: text input
    :param vocab: vocabulary
    :param unk_token: index for unk token
    :return:
    """
    words = input_string.split()
    sentence = []
    for word in words:
        if word in vocab:
            sentence.append(vocab[word])
        else:
            sentence.append(unk_token)

    return_array = np.array(sentence)
    return return_array.astype('int64')


def get_mrr_score(real, predict):
    total = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            total = total + 1.0 / float(index + 1)
    return total / float(len(real))


class CrCritic(object):
    def __init__(self):
        print("Loading Config")
        self.conf = get_config()

        print("Loading Negative sample dataset")
        self.qts, self.max_qt_len, self.qt_vocab = self._load_data_set("qt")
        self.qbs, self.max_qb_len, self.qb_vocab = self._load_data_set("qb")
        self.codes, self.max_code_len, self.code_vocab = self._load_data_set("code")

        self.data_len = len(self.qts)
        print("Data Length", self.data_len)

        print("Loading Model")
        self.model = self._load_model()

    def _load_data_set(self, data_set):
        conf = self.conf
        if data_set == "qt":
            data_filename, max_len, vocab_file, vocab_limit = conf['test_qt'], conf['qt_len'], conf['vocab_qt'], \
                                                              conf['qt_n_words']
        elif data_set == "qb":
            data_filename, max_len, vocab_file, vocab_limit = conf['test_qb'], conf['qb_len'], conf['vocab_qb'], \
                                                              conf['qb_n_words']
        else:
            data_filename, max_len, vocab_file, vocab_limit = conf['test_code'], conf['code_len'], conf['vocab_code'], \
                                                              conf['code_n_words']
        path = conf['workdir']
        list_of_data = pickle.load(open(path + data_filename, 'rb'))
        vocab = limit_vocab(load_dict(path + vocab_file), vocab_limit)
        processed_data = processed_dataset(list_of_data)
        return processed_data, max_len, vocab

    def _load_model(self):
        conf = self.conf
        model_string = create_model_name_string(conf)
        model = JointEmbeder(conf)
        _reload_model(model, conf, model_string)
        if torch.cuda.is_available():
            model = model.cuda()

        model = model.eval()
        return model

    def _get_negative_pool(self, poolsize, processed_code, processed_qt, processed_annotation, processed_qb):
        """
        :param poolsize: poolsize for negative sampling
        :param processed_code: code converted to indices array
        :param processed_qt: qt converted to indices array
        :param processed_annotation: Annotation converted to indices array
        :param processed_qb: qb converted to indices array
        :return:
        """
        negative_qts, negative_qbs, negative_codes = [], [], []
        for i in range(poolsize-1):
            negative_index = random.randint(0, self.data_len - 1)
            neg_qt, neg_qb, neg_code = self.qts[negative_index], self.qbs[negative_index], self.codes[negative_index]
            while np.array_equal(neg_qt, processed_qt) or np.array_equal(neg_code, processed_code) or \
                    np.array_equal(neg_qb, processed_annotation) or np.array_equal(neg_qb, processed_qb):
                negative_index = random.randint(0, self.data_len - 1)
                neg_qt, neg_qb, neg_code = self.qts[negative_index].astype('int64'), \
                                           self.qbs[negative_index].astype('int64'), \
                                           self.codes[negative_index].astype('int64')

            neg_code = pad_seq(neg_code, self.max_code_len)
            neg_qt = pad_seq(neg_qt, self.max_qt_len)
            neg_qb = pad_seq(neg_qb, self.max_qb_len)

            negative_qts.append(neg_qt)
            negative_qbs.append(neg_qb)
            negative_codes.append(neg_code)

        return negative_qts, negative_qbs, negative_codes

    def get_reward(self, code, annotation, qt, qb, poolsize=50, number_of_runs=20, top_n_results=-1):
        """
        :param code: Cleaned code string
        :param annotation: Annotation produced by the Code Annotation Model
        :param qt: Cleaned QT
        :param qb: Cleaned actual Question Body
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param top_n_results: If required restrict to top n results, -1 use full pool
        :return: Mean MRR score for the given inputs
        """

        # Process Input data
        processed_code = process_text(code, self.code_vocab)
        processed_qt = process_text(qt, self.qt_vocab)
        processed_annotation = process_text(annotation, self.qb_vocab)
        processed_qb = process_text(qb, self.qb_vocab)

        processed_code = pad_seq(processed_code, self.max_code_len)
        processed_qt = pad_seq(processed_qt, self.max_qt_len)
        processed_annotation = pad_seq(processed_annotation, self.max_qb_len)
        processed_qb = pad_seq(processed_qb, self.max_qb_len)

        # print(processed_code.shape)
        # print(processed_qt.shape)
        # print(processed_annotation.shape)
        # print(processed_qb.shape)
        # print('*******')

        mrrs = []
        for _ in range(number_of_runs):
            neg_qts, neg_qbs, neg_codes = self._get_negative_pool(poolsize, processed_code, processed_qt,
                                                               processed_annotation, processed_qb)
            # print("Total negative pool size", len(neg_qts))

            # print(processed_code.shape)
            # for a in neg_codes:
            #     print(a.shape)
            #     break
            #
            # print(processed_qt.shape)
            # for a in neg_qts:
            #     print(a.shape)
            #     break
            #
            # print(processed_annotation.shape)
            # for a in neg_qbs:
            #     print(a.shape)
            #     break

            combined_qts = np.stack([processed_qt] + neg_qts, axis=0)
            combined_codes = np.stack([processed_code] + neg_codes, axis=0)
            combined_qbs = np.stack([processed_annotation] + neg_qbs, axis=0)

            # print(combined_qts.shape)
            # print(combined_codes.shape)
            # print(combined_qbs.shape)

            combined_qts, combined_codes, combined_qbs = gVar(combined_qts), gVar(combined_codes), gVar(combined_qbs)

            code_repr = self.model.code_encoding(combined_codes, combined_qbs)

            qt = gVar(combined_qts[0].expand(poolsize, -1))
            qt_repr = self.model.qt_encoding(qt)

            sims = F.cosine_similarity(code_repr, qt_repr).data.cpu().numpy()
            neg_sims = np.negative(sims)
            predict = np.argsort(neg_sims)

            if top_n_results > 0:
                n_results = top_n_results
                predict = predict[:n_results]

            predict = [int(k) for k in predict]
            real = [0]
            # print("Calculating MRR")
            mrrs.append(get_mrr_score(real, predict))

        mean_mrr = np.mean(mrrs)
        print(" MRR", mean_mrr)
        return mean_mrr
