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
        # print("Loading Config")
        self.conf = get_config()

        print("Loading Negative sample dataset")
        self.qts, self.max_qt_len, self.qt_vocab = self._load_data_set("qt")
        self.qbs, self.max_qb_len, self.qb_vocab = self._load_data_set("qb")
        self.codes, self.max_code_len, self.code_vocab = self._load_data_set("code")

        self.data_len = len(self.qts)
        print("Sample Data Size", self.data_len)

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

            negative_qts.append(neg_qt)
            negative_qbs.append(neg_qb)
            negative_codes.append(neg_code)

        return negative_qts, negative_qbs, negative_codes

    def _batchify(self, data, align_right=False, include_lengths=False):
        data = [torch.from_numpy(x) for x in data]
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(self.conf['PAD'])
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def get_reward(self, code, annotation, qt, qb,
                   poolsize=50, number_of_runs=1, top_n_results=-1,
                   bool_processed=False):
        """
        :param code: Cleaned code string or tokens
        :param annotation: Annotation produced by the Code Annotation Model
        :param qt: Cleaned QT string or tokens
        :param qb: Cleaned actual Question Body string or tokens
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param top_n_results: If required restrict to top n results, -1 use full pool
        :param bool_processed: Set to True if the inputs are all processed
        :return: Mean MRR score for the given inputs
        """

        # Process Input data
        processed_code = np.expand_dims(process_text(code, self.code_vocab) if not bool_processed else code, axis=0)
        processed_qt = np.expand_dims(process_text(qt, self.qt_vocab) if not bool_processed else qt, axis=0)
        processed_annotation = np.expand_dims(process_text(annotation, self.qb_vocab) if not bool_processed else annotation, axis=0)
        processed_qb = np.expand_dims(process_text(qb, self.qb_vocab) if not bool_processed else qb, axis=0)

        processed_qt, processed_code, processed_annotation = gVar(processed_qt), gVar(processed_code), \
                                                             gVar(processed_annotation)
        qt_repr = self.model.qt_encoding(processed_qt)
        code_repr = self.model.code_encoding(processed_code, processed_annotation)
        positive_score = F.cosine_similarity(code_repr, qt_repr).data.cpu().numpy()
        positive_score = positive_score[0]

        mrrs_in_runs = []
        _, neg_qbs, neg_codes = self._get_negative_pool(
            (poolsize-1) * number_of_runs + 1, processed_code, processed_qt,
            processed_annotation, processed_qb)
        neg_qbs, neg_codes = gVar(self._batchify(neg_qbs)), gVar(self._batchify(neg_codes))
        neg_codes_repr = self.model.code_encoding(neg_codes, neg_qbs)
        sims_scores = F.cosine_similarity(neg_codes_repr, qt_repr).data.tolist()
        assert len(sims_scores) == (poolsize - 1) * number_of_runs

        for run_idx in range(number_of_runs):
            sims = [positive_score] + sims_scores[run_idx * (poolsize - 1): (run_idx + 1) * (poolsize - 1)]
            assert len(sims) == poolsize

            neg_sims = np.negative(np.array(sims, dtype=np.float32))
            predict = np.argsort(neg_sims)

            if top_n_results > 0:
                n_results = top_n_results
                predict = predict[:n_results]

            predict = [int(k) for k in predict]
            real = [0]
            mrrs_in_runs.append(get_mrr_score(real, predict))

        mrr = np.mean(mrrs_in_runs)
        return mrr

    def get_reward_in_batch(self, codes, annotations, qts, qbs,
                            poolsize=50, number_of_runs=1, top_n_results=-1):
        """
        :param codes: A list of cleaned-up processed codes
        :param annotations: A list of cleaned-up processed annotations produced by the Code Annotation Model
        :param qts: A list of cleaned-up processed QTs
        :param qbs: A list of cleaned-up processed actual Question Bodies
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param top_n_results: If required restrict to top n results, -1 use full pool
        :return: Mean MRR score for the given inputs
        """

        processed_qts, processed_codes, processed_annotations = gVar(self._batchify(qts)), gVar(self._batchify(codes)), \
                                                             gVar(self._batchify(annotations))
        qts_repr = self.model.qt_encoding(processed_qts)
        codes_repr = self.model.code_encoding(processed_codes, processed_annotations)
        positive_scores = F.cosine_similarity(codes_repr, qts_repr).data.tolist()

        # collect negative pools
        all_neg_qbs, all_neg_codes = [], []
        for code, annotation, qt, qb in zip(codes, annotations, qts, qbs):
            _, neg_qbs, neg_codes = self._get_negative_pool(
                (poolsize - 1) * number_of_runs + 1, code, qt, annotation, qb)
            all_neg_qbs.extend(neg_qbs)
            all_neg_codes.extend(neg_codes)

        all_neg_qbs, all_neg_codes = gVar(self._batchify(all_neg_qbs)), gVar(self._batchify(all_neg_codes))
        all_neg_codes_repr = self.model.code_encoding(all_neg_codes, all_neg_qbs)

        mrrs = []
        for example_idx, (qt_repr, positive_score) in enumerate(zip(list(qts_repr), positive_scores)):
            neg_codes_repr = all_neg_codes_repr[example_idx * (poolsize - 1) * number_of_runs:
                                                (example_idx + 1) * (poolsize - 1) * number_of_runs]
            assert len(neg_codes_repr) == (poolsize - 1) * number_of_runs
            sims_scores = F.cosine_similarity(neg_codes_repr, qt_repr.view(1, -1)).data.tolist()

            mrrs_in_runs = []
            for run_idx in range(number_of_runs):
                sims = [positive_score] + sims_scores[run_idx * (poolsize - 1): (run_idx + 1) * (poolsize - 1)]
                assert len(sims) == poolsize

                neg_sims = np.negative(np.array(sims, dtype=np.float32))
                predict = np.argsort(neg_sims)

                if top_n_results > 0:
                    n_results = top_n_results
                    predict = predict[:n_results]

                predict = [int(k) for k in predict]
                real = [0]
                mrrs_in_runs.append(get_mrr_score(real, predict))
            mrrs.append(np.average(mrrs_in_runs))

        return mrrs