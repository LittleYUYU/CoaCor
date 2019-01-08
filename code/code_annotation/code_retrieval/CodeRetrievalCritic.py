import sys
import os
import random
import numpy as np
import pickle
import pdb

import torch
import torch.nn.functional as F

from models import JointEmbeder
from configs import get_config

GPU_ID = 0

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.set_device(GPU_ID)
torch.cuda.manual_seed_all(42)


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


def limit_vocab(old_vocab_dict, vocab_length):
    new_vocab_dict = {}
    for word, key in old_vocab_dict.iteritems():
        if key < vocab_length:
            new_vocab_dict[word] = key
    return new_vocab_dict


def processed_dataset(list_of_strings, PAD_ID):
    preprocessed_text = []
    idx2text = {}
    strings = all(isinstance(n, str) for n in list_of_strings)
    assert not strings
    for data_point in list_of_strings:
        # if strings:
        #     string = data_point
        # else:
        string = data_point[1]
        idx = data_point[0]
        if len(string) == 0:
            string = "%s" % str(PAD_ID)
        words = map(int, string.split())
        preprocessed_text.append(np.array(words))
        idx2text[idx] = np.array(words)
    return preprocessed_text, idx2text


def _reload_model(model, conf):
    print("Loading from %s..." % conf['checkpoint'])
    model.load_state_dict(torch.load(conf['checkpoint']))


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    if torch.cuda.is_available():
        tensor = tensor.cuda(GPU_ID)
    return tensor


def process_text(input_string, vocab, unk_token):
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
        sentence.append(vocab.get(word, unk_token))

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
        self.conf = get_config()

        # loading vocab
        path = self.conf['workdir']
        self.qt_vocab = limit_vocab(load_dict(path + self.conf['vocab_qt']), self.conf['qt_n_words'])
        self.code_vocab = limit_vocab(load_dict(path + self.conf['vocab_code']), self.conf['code_n_words'])

        self.qt_vocab_inv = {v:k for k,v in self.qt_vocab.items()}
        self.code_vocab_inv = {v:k for k,v in self.code_vocab.items()}

        # load data
        list_of_data = pickle.load(open(path + self.conf['train_qt'], 'rb'))
        self.qts, _ = processed_dataset(list_of_data, self.conf['PAD'])
        list_of_data = pickle.load(open(path + self.conf['train_code'], 'rb'))
        self.codes, _ = processed_dataset(list_of_data, self.conf['PAD'])

        print("Loading Model")
        self.model = self._load_model(self.conf)

    def _load_model(self, conf):
        model = JointEmbeder(conf)
        _reload_model(model, conf)
        if torch.cuda.is_available():
            model = model.cuda(GPU_ID)

        model = model.eval()
        return model

    def _get_negative_pool(self, data_name, poolsize, processed_code, processed_qt):
        """
        :param data_name: should be "train"
        :param poolsize: poolsize for negative sampling
        :param processed_code: code converted to indices array, 2D
        :param processed_qt: qt converted to indices array, 2D
        :return:
        """

        negative_qts, negative_codes = [], []
        for i in range(poolsize-1):
            negative_index = random.randint(0, len(self.qts) - 1)
            neg_qt, neg_code = self.qts[negative_index], self.codes[negative_index]

            while np.array_equal(neg_code, processed_code.reshape([-1])) or \
                np.array_equal(neg_qt, processed_qt.reshape([-1])):
                negative_index = random.randint(0, len(self.qts) - 1)
                neg_qt, neg_code = self.qts[negative_index].astype('int64'), \
                                    self.codes[negative_index].astype('int64')

            negative_qts.append(neg_qt)
            negative_codes.append(neg_code)

        return negative_qts, negative_codes

    def _batchify(self, data, align_right=False, include_lengths=False, max_length=200):
        data = [torch.from_numpy(x) for x in data]
        lengths = [x.size(0) for x in data]
        # max_length = min(max(lengths), max_length)
        lengths = [i if i < max_length else max_length for i in lengths]

        out = data[0].new(len(data), max_length).fill_(self.conf['PAD'])
        for i in range(len(data)):
            data_length = data[i].size(0)
            if data_length == 0:
                data[i] = torch.LongTensor(np.array([self.conf['PAD']]))
                data_length = 1
            if data_length > max_length:
                data[i] = data[i][:max_length]
                data_length = max_length
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def get_reward_batch_eval(self, codes, annotations, qts):
        """
        Get rewards when taking this batch as a pool.
        :param codes: a batch of cleaned codes.
        :param annotations: corresponding annotations.
        :param qts: corresponding QTs.
        :return:
        """
        #annotations = qts # debug
        processed_qts, processed_codes, processed_annotations = \
            gVar(self._batchify(qts, max_length=self.conf['qt_len'])), \
            gVar(self._batchify(codes, max_length=self.conf['code_len'])), \
            gVar(self._batchify(annotations, max_length=self.conf['qt_len']))

        annos_repr = self.model.qt_encoding(processed_annotations)
        codes_repr = self.model.cand_encoding(processed_codes)

        rr_per_pos = []
        for pos_idx in range(len(codes)):
            anno_repr = annos_repr[pos_idx].expand(1, -1)
            pos_code_repr = codes_repr[pos_idx].expand(1, -1)
            positive_score = self.model.scoring(anno_repr, pos_code_repr).data.cpu().numpy()
            positive_score = positive_score[0]

            _neg_codes_repr = torch.cat([codes_repr[:pos_idx], codes_repr[pos_idx+1:]])
            _neg_sims_scores = self.model.scoring(anno_repr, _neg_codes_repr).data.tolist()

            sims = [positive_score] + _neg_sims_scores

            neg_sims = np.negative(np.array(sims, dtype=np.float32))
            predict = np.argsort(neg_sims)
            predict = [int(k) for k in predict]
            real = [0]
            rr_per_pos.append(get_mrr_score(real, predict))

        return rr_per_pos

    def get_reward(self, data_name, code, annotation, qt,
                   poolsize=50, number_of_runs=1,
                   bool_processed=False):
        """
        :param code: Cleaned code string or tokens
        :param annotation: Annotation produced by the Code Annotation Model
        :param qt: Cleaned QT string or tokens
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param bool_processed: Set to True if the inputs are all processed
        :return: Mean MRR score for the given inputs
        """

        # Process Input data
        processed_code = np.expand_dims(process_text(code, self.code_vocab, unk_token=self.conf['UNK']) if not bool_processed else code, axis=0)
        processed_annotation = np.expand_dims(process_text(annotation, self.qb_vocab, unk_token=self.conf['UNK']) if not bool_processed else annotation, axis=0)
        processed_qt = np.expand_dims(process_text(qt, self.qt_vocab, unk_token=self.conf['UNK']) if not bool_processed else qt, axis=0)

        _processed_code, _processed_annotation = \
            gVar(self._batchify(processed_code, max_length=self.conf['code_len'])), \
            gVar(self._batchify(processed_annotation, max_length=self.conf['qt_len']))

        _, neg_codes = self._get_negative_pool(data_name,
            (poolsize-1) * number_of_runs + 1, processed_code, processed_qt)

        _neg_codes = gVar(self._batchify(neg_codes, max_length=self.conf['code_len']))

        anno_repr = self.model.qt_encoding(_processed_annotation)
        code_repr = self.model.cand_encoding(_processed_code)
        positive_score = F.cosine_similarity(code_repr, anno_repr).data.cpu().numpy()
        positive_score = positive_score[0]

        mrrs_in_runs = []
        neg_codes_repr = self.model.cand_encoding(_neg_codes)
        sims_scores = F.cosine_similarity(neg_codes_repr, anno_repr).data.tolist()
        assert len(sims_scores) == (poolsize - 1) * number_of_runs

        for run_idx in range(number_of_runs):
            sims = [positive_score] + sims_scores[run_idx * (poolsize - 1): (run_idx + 1) * (poolsize - 1)]
            assert len(sims) == poolsize

            neg_sims = np.negative(np.array(sims, dtype=np.float32))
            predict = np.argsort(neg_sims)
            predict = [int(k) for k in predict]
            real = [0]
            mrrs_in_runs.append(get_mrr_score(real, predict))

        mrr = np.mean(mrrs_in_runs)

        return mrr
