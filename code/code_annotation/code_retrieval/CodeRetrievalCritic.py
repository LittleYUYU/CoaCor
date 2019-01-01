import sys
import os
import random
import numpy as np
import pickle
import pdb

import torch
import torch.nn.functional as F

from models import JointEmbeder, JointEmbederTP
from configs import get_config, get_config_noQB

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


# def create_model_name_string_old(c):
#     string1 = 'qtlen_{}_qblen_{}_codelen_{}_qtnwords_{}_qbnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
#         format(c['qt_len'], c['code_len'], c['qb_len'], c['qt_n_words'], c['qb_n_words'], c['code_n_words'],
#                c['batch_size'], c['optimizer'], str(c['lr'])[2:])
#     string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}_simmeasure_{}'. \
#         format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:], c['sim_measure'])
#     string = string1 + string2
#     return string


def create_model_name_string(c):
    string1 = 'qtlen_{}_qblen_{}_codelen_{}_qtnwords_{}_qbnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
        format(c['qt_len'], c['qb_len'], c['code_len'], c['qt_n_words'], c['qb_n_words'], c['code_n_words'],
               c['batch_size'], c['optimizer'], str(c['lr'])[2:])
    string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}_simmeasure_{}'. \
        format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:], c['sim_measure'])
    string3 = '_maxpool'
    string4 = '_xent' if c['loss'] == 'xent' else ''
    string5 = '_margin%s' % str(c['margin'])[2:] if c['margin'] != 0.05 else ''
    if c['cr_setup'] == "tp_qt_new_cleaned_rl_mrr_qb" and c['tp_weight'] not in {1.0, 0.0, -1}:
        string6 = '_tpW%s' % str(c['tp_weight'])[2:]
    else:
        string6 = ""
    string = string1 + string2 + string3 + string4 + string5 + string6
    return string


def _reload_model(model, conf, model_string):
    conf['model_directory'] = conf['workdir'] +\
                              'model%s/' % (conf['cr_setup'] if conf['cr_setup'] is not "default" else "") +\
                              model_string + '/'
    print("Loading from %s..." % conf['model_directory'])
    assert os.path.exists(conf['model_directory'] + 'best_model.h5'), 'Weights for saved best model not found'
    model.load_state_dict(torch.load(conf['model_directory'] + 'best_model.h5'))


def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    if torch.cuda.is_available():
        tensor = tensor.cuda(GPU_ID)
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


def ids2text(ids, vocab):
    return " ".join([vocab.get(id, "<unk>") for id in ids])


class CrCritic(object):
    def __init__(self, cr_setup, qt_dict_map=None):
        self.conf = get_config(cr_setup)
        self.conf_noQB = get_config_noQB("noqb")
        # self.conf = self.conf_noQB

        print("Loading Negative sample dataset")
        self.qts, self.qbs, self.codes, self.data_len = {}, {}, {}, {}
        # loading pos_to_negs pairs (fixed negative candidates comparison)
        # self.pos2negs = {}
        # self.qt_idx2ids, self.qb_idx2ids, self.code_idx2ids = {}, {}, {}
        # for data_name in ["train"]: #"valid", "test"
        #     qts, idx2qt, self.max_qt_len, self.qt_vocab = self._load_data_set("qt", data_name)
        #     qbs, idx2qb, self.max_qb_len, self.qb_vocab = self._load_data_set("qb", data_name)
        #     codes, idx2code, self.max_code_len, self.code_vocab = self._load_data_set("code", data_name)
        #     # process into dict
        #     self.qt_idx2ids.update(idx2qt)
        #     self.qb_idx2ids.update(idx2qb)
        #     self.code_idx2ids.update(idx2code)
        #
        #     self.qts[data_name] = qts
        #     self.qbs[data_name] = qbs
        #     self.codes[data_name] = codes
        #     self.data_len[data_name] = len(qts)
        #     if data_name == "train":
        #         continue
        #     self.pos2negs[data_name] = pickle.load(open(self.conf['workdir'] + self.conf['%s_pos2negs' % data_name]))

        # loading vocab
        path = self.conf['workdir']
        self.qt_vocab = limit_vocab(load_dict(path + self.conf['vocab_qt']), self.conf['qt_n_words'])
        self.qb_vocab = limit_vocab(load_dict(path + self.conf['vocab_qb']), self.conf['qb_n_words'])
        self.code_vocab = limit_vocab(load_dict(path + self.conf['vocab_code']), self.conf['code_n_words'])

        self.qt_vocab_inv = {v:k for k,v in self.qt_vocab.items()}
        self.qb_vocab_inv = {v:k for k,v in self.qb_vocab.items()}
        self.code_vocab_inv = {v:k for k,v in self.code_vocab.items()}

        self.qt_dict_map = qt_dict_map

        # print("Sample Data Size: valid %d, test %d." % (self.data_len["valid"], self.data_len["test"]))

        print("Loading Model")
        self.model = self._load_model(self.conf)
        self.model_noQB = self._load_model(self.conf_noQB)

    def _load_data_set(self, data_set, data_name):
        conf = self.conf
        if data_set == "qt":
            data_filename, max_len, vocab_file, vocab_limit = conf['%s_qt' % data_name], conf['qt_len'],\
                                                              conf['vocab_qt'], conf['qt_n_words']
        elif data_set == "qb":
            data_filename, max_len, vocab_file, vocab_limit = conf['%s_qb' % data_name], conf['qb_len'],\
                                                              conf['vocab_qb'], conf['qb_n_words']
        else:
            data_filename, max_len, vocab_file, vocab_limit = conf['%s_code' % data_name], conf['code_len'],\
                                                              conf['vocab_code'], conf['code_n_words']
        path = conf['workdir']
        list_of_data = pickle.load(open(path + data_filename, 'rb'))
        vocab = limit_vocab(load_dict(path + vocab_file), vocab_limit)
        processed_data, processed_idx2data = processed_dataset(list_of_data, self.conf['PAD'])
        return processed_data, processed_idx2data, max_len, vocab

    def _load_model(self, conf):
        model_string = create_model_name_string(conf)
        if conf['cr_setup'] == "tp_qt_new_cleaned_rl_mrr_qb":
            model = JointEmbederTP(conf)
        else:
            model = JointEmbeder(conf)
        _reload_model(model, conf, model_string)
        if torch.cuda.is_available():
            model = model.cuda(GPU_ID)

        model = model.eval()
        return model

    def _get_negative_pool(self, data_name, poolsize,
                           processed_code, processed_qt, processed_annotation, processed_qb,
                           bool_empty_qb=False):
        """
        :param data_name: valid or test
        :param poolsize: poolsize for negative sampling
        :param processed_code: code converted to indices array, 2D
        :param processed_qt: qt converted to indices array, 2D
        :param processed_annotation: Annotation converted to indices array, 2D
        :param processed_qb: qb converted to indices array, 2D
        :return:
        """

        negative_qts, negative_qbs, negative_codes = [], [], []
        for i in range(poolsize-1):
            negative_index = random.randint(0, self.data_len[data_name] - 1)
            neg_qt, neg_qb, neg_code = self.qts[data_name][negative_index], self.qbs[data_name][negative_index], \
                                       self.codes[data_name][negative_index]

            while np.array_equal(neg_qt, processed_qt.reshape([-1])) or \
                np.array_equal(neg_code, processed_code.reshape([-1])) or \
                np.array_equal(neg_qb, processed_annotation.reshape([-1])) or \
                np.array_equal(neg_qb, processed_qb.reshape([-1])):
                negative_index = random.randint(0, self.data_len[data_name] - 1)
                neg_qt, neg_qb, neg_code = self.qts[data_name][negative_index].astype('int64'), \
                                           self.qbs[data_name][negative_index].astype('int64'), \
                                           self.codes[data_name][negative_index].astype('int64')

            negative_qts.append(neg_qt)
            if bool_empty_qb:
                negative_qbs.append(np.array([self.conf['PAD']]).astype('int64'))
            else:
                negative_qbs.append(neg_qb)
            negative_codes.append(neg_code)

        return negative_qts, negative_qbs, negative_codes

    def _get_negative_pool_by_id(self, data_name, pos_id, bool_empty_qb=False):
        neg_indices = self.pos2negs[data_name][pos_id]
        negative_qts, negative_qbs, negative_codes = [], [], []
        for qt_idx, qb_idx, code_idx in neg_indices:
            neg_qt, neg_qb, neg_code = self.qt_idx2ids[qt_idx], self.qb_idx2ids[qb_idx], \
                                       self.code_idx2ids[code_idx]
            negative_qts.append(neg_qt)
            if bool_empty_qb:
                negative_qbs.append(np.array([self.conf['PAD']]).astype('int64'))
            else:
                negative_qbs.append(neg_qb)
            negative_codes.append(neg_code)

        return negative_qts, negative_qbs, negative_codes, neg_indices

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

    def _qt_dict_convert(self, qts, mapping):
        mapped_qts = []
        for qt in qts:
            mapped_qt = [mapping[i] for i in qt]
            mapped_qts.append(np.array(mapped_qt))
        return mapped_qts

    def get_reward_batch_eval(self, codes, annotations, qts, qbs,
                              reward_mode="cr", top_n_results=-1, bool_empty_qb=False,
                              replace_all=False):
        """
        Get rewards taking this batch as a pool.
        :param codes:
        :param annotations:
        :param qts:
        :param qbs:
        :param reward_mode: {cr (default) | cr_diff}
        :param bool_empty_qb: Set to True for feeding empty annotations
        :param replace_all: Set to True for replacing neg_qbs with generated neg_annos.
        :return:
        """
        if reward_mode == "cr_noqb":
            cr_models = [self.model_noQB]
            qts = annotations
            if self.qt_dict_map is not None:
                qts = self._qt_dict_convert(qts, self.qt_dict_map)

        else:
            cr_models = [self.model]
            if reward_mode == "cr_diff":
                cr_models.append(self.model_noQB)

        if replace_all:
            qbs = annotations

        if bool_empty_qb:
            qbs = [np.array([self.conf['PAD']]).astype('int64')] * len(qbs)

        processed_qts, processed_codes, processed_annotations, processed_qbs = \
            gVar(self._batchify(qts, max_length=self.conf['qt_len'])), \
            gVar(self._batchify(codes, max_length=self.conf['code_len'])), \
            gVar(self._batchify(annotations, max_length=self.conf['qb_len'])), \
            gVar(self._batchify(qbs, max_length=self.conf['qb_len']))

        cr_model_mrrs = []
        for cr_model in cr_models:
            pos_qts_repr = cr_model.qt_encoding(processed_qts)
            pos_codes_repr = cr_model.code_encoding(processed_codes, processed_annotations)
            neg_codes_repr = cr_model.code_encoding(processed_codes, processed_qbs)
            if cr_model.name == "tp":
                pos_qts_repr2 = cr_model.qt_encoding2(processed_qts)
                pos_qbs_repr = cr_model.qb_encoding(processed_annotations)
                neg_qbs_repr = cr_model.qb_encoding(processed_qbs)

            rr_per_pos = []
            for pos_idx in range(len(codes)):
                qt_repr = pos_qts_repr[pos_idx].expand(1, -1)
                pos_code_repr = pos_codes_repr[pos_idx].expand(1, -1)
                # positive_score = F.cosine_similarity(pos_code_repr, qt_repr).data.cpu().numpy()
                if cr_model.name == "tp":
                    pos_qb_repr = pos_qbs_repr[pos_idx].expand(1, -1)
                    qt_repr2 = pos_qts_repr2[pos_idx].expand(1, -1)
                else:
                    pos_qb_repr, qt_repr2 = None, None
                positive_score = cr_model.scoring(qt_repr, pos_code_repr, pos_qb_repr, qt_repr2).data.cpu().numpy()
                positive_score = positive_score[0]

                _neg_codes_repr = torch.cat([neg_codes_repr[:pos_idx], neg_codes_repr[pos_idx+1:]])
                # _neg_sims_scores = F.cosine_similarity(_neg_codes_repr, qt_repr).data.tolist()
                if cr_model.name == "tp":
                    _neg_qbs_repr = torch.cat([neg_qbs_repr[:pos_idx], neg_qbs_repr[pos_idx+1:]])
                else:
                    _neg_qbs_repr = None
                _neg_sims_scores = cr_model.scoring(qt_repr, _neg_codes_repr, _neg_qbs_repr, qt_repr2).data.tolist()

                sims = [positive_score] + _neg_sims_scores

                neg_sims = np.negative(np.array(sims, dtype=np.float32))
                predict = np.argsort(neg_sims)

                if top_n_results > 0:
                    n_results = top_n_results
                    predict = predict[:n_results]

                predict = [int(k) for k in predict]
                real = [0]
                rr_per_pos.append(get_mrr_score(real, predict))

            cr_model_mrrs.append(rr_per_pos)

        if reward_mode in ["cr", "cr_noqb"]:
            assert len(cr_model_mrrs) == 1
            #return cr_model_mrrs[0]
        elif reward_mode == "cr_diff":
            assert len(cr_model_mrrs) == 2
            # return cr_model_mrrs[0] - cr_model_mrrs[1]
        else:
            raise Exception("Invalid reward mode %s!" % reward_mode)
        return cr_model_mrrs

    def get_reward_batch_eval_qt(self, codes, annotations, qts, qbs,
                              reward_mode="cr", top_n_results=-1):
        """
        Get rewards taking this batch as a pool.
        NOTE: this function takes QTs as candidates, and annotations must be used for all code snippets.
        :param codes:
        :param annotations:
        :param qts:
        :param qbs: a placeholder that would not be used
        :param reward_mode: {cr (default) | cr_diff}
        :return:
        """
        assert reward_mode in {"cr", "cr_diff"}
        cr_models = [self.model]
        if reward_mode == "cr_diff":
            cr_models.append(self.model_noQB)

        processed_qts, processed_codes, processed_annotations = \
            gVar(self._batchify(qts, max_length=self.conf['qt_len'])), \
            gVar(self._batchify(codes, max_length=self.conf['code_len'])), \
            gVar(self._batchify(annotations, max_length=self.conf['qb_len']))

        cr_model_mrrs = []
        for cr_model in cr_models:
            qts_repr = cr_model.qt_encoding(processed_qts)
            codes_repr = cr_model.code_encoding(processed_codes, processed_annotations)
            if cr_model.name == "tp":
                qts_repr2 = cr_model.qt_encoding2(processed_qts)
                qbs_repr = cr_model.qb_encoding(processed_annotations)

            rr_per_pos = []
            for pos_idx in range(len(codes)):
                code_repr = codes_repr[pos_idx].expand(1, -1)
                pos_qt_repr = qts_repr[pos_idx].expand(1, -1)
                if cr_model.name == "tp":
                    qb_repr = qbs_repr[pos_idx].expand(1, -1)
                    pos_qt_repr2 = qts_repr2[pos_idx].expand(1, -1)
                else:
                    qb_repr, pos_qt_repr2 = None, None
                positive_score = cr_model.scoring(pos_qt_repr, code_repr, qb_repr, pos_qt_repr2).data.cpu().numpy()
                positive_score = positive_score[0]

                _neg_qts_repr = torch.cat([qts_repr[:pos_idx], qts_repr[pos_idx+1:]])
                if cr_model.name == "tp":
                    _neg_qts_repr2 = torch.cat([qts_repr2[:pos_idx], qts_repr2[pos_idx+1:]])
                else:
                    _neg_qts_repr2 = None
                _neg_sims_scores = cr_model.scoring(_neg_qts_repr, code_repr, qb_repr, _neg_qts_repr2).data.tolist()

                sims = [positive_score] + _neg_sims_scores

                neg_sims = np.negative(np.array(sims, dtype=np.float32))
                predict = np.argsort(neg_sims)

                if top_n_results > 0:
                    n_results = top_n_results
                    predict = predict[:n_results]

                predict = [int(k) for k in predict]
                real = [0]
                rr_per_pos.append(get_mrr_score(real, predict))

            cr_model_mrrs.append(rr_per_pos)

        if reward_mode == "cr":
            assert len(cr_model_mrrs) == 1
            #return cr_model_mrrs[0]
        elif reward_mode == "cr_diff":
            assert len(cr_model_mrrs) == 2
            # return cr_model_mrrs[0] - cr_model_mrrs[1]
        else:
            raise Exception("Invalid reward mode %s!" % reward_mode)
        return cr_model_mrrs


    def get_reward(self, data_name, code, annotation, qt, qb, idx=None,
                   reward_mode="cr", poolsize=50, number_of_runs=1, top_n_results=-1,
                   bool_processed=False, bool_empty_qb=False):
        """
        :param code: Cleaned code string or tokens
        :param annotation: Annotation produced by the Code Annotation Model
        :param qt: Cleaned QT string or tokens
        :param qb: Cleaned actual Question Body string or tokens
        :param idx: the <question id, qb id, code id> index of this positive example.
        :param reward_mode: {cr (default) | cr_diff}.
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param top_n_results: If required restrict to top n results, -1 use full pool
        :param bool_processed: Set to True if the inputs are all processed
        :param bool_empty_qb: Set to True for feeding empty annotations
        :return: Mean MRR score for the given inputs
        """
        if reward_mode == "cr_noqb":
            cr_models = [self.model_noQB]
            qt = annotation
            if self.qt_dict_map is not None:
                qt = self._qt_dict_convert([qt], self.qt_dict_map)[0]

        else:
            cr_models = [self.model]
            if reward_mode == "cr_diff":
                cr_models.append(self.model_noQB)

        # Process Input data
        processed_code = np.expand_dims(process_text(code, self.code_vocab, unk_token=self.conf['UNK']) if not bool_processed else code, axis=0)
        processed_qt = np.expand_dims(process_text(qt, self.qt_vocab, unk_token=self.conf['UNK']) if not bool_processed else qt, axis=0)
        processed_annotation = np.expand_dims(process_text(annotation, self.qb_vocab, unk_token=self.conf['UNK']) if not bool_processed else annotation, axis=0)
        processed_qb = np.expand_dims(process_text(qb, self.qb_vocab, unk_token=self.conf['UNK']) if not bool_processed else qb, axis=0)

        _processed_qt, _processed_code, _processed_annotation = \
            gVar(self._batchify(processed_qt, max_length=self.conf['qt_len'])), \
            gVar(self._batchify(processed_code, max_length=self.conf['code_len'])), \
            gVar(self._batchify(processed_annotation, max_length=self.conf['qb_len']))

        # sample negative examples
        if idx is not None:
            assert number_of_runs == 1
            _, neg_qbs, neg_codes, neg_indices = self._get_negative_pool_by_id(data_name, idx, bool_empty_qb)
        else:
            _, neg_qbs, neg_codes = self._get_negative_pool(data_name,
                (poolsize-1) * number_of_runs + 1, processed_code, processed_qt,
                processed_annotation, processed_qb, bool_empty_qb=bool_empty_qb)

        _neg_qbs, _neg_codes = gVar(self._batchify(neg_qbs, max_length=self.conf['qb_len'])), \
                             gVar(self._batchify(neg_codes, max_length=self.conf['code_len']))

        cr_model_mrrs = []
        for cr_model in cr_models:
            qt_repr = cr_model.qt_encoding(_processed_qt)
            code_repr = cr_model.code_encoding(_processed_code, _processed_annotation)
            positive_score = F.cosine_similarity(code_repr, qt_repr).data.cpu().numpy()
            positive_score = positive_score[0]

            mrrs_in_runs = []
            neg_codes_repr = cr_model.code_encoding(_neg_codes, _neg_qbs)
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

                # print for debugging
                if False and idx is not None and number_of_runs == 1:
                    print("-" * 50)
                    print("Pos: score %.3f. rr = %.3f" % (positive_score, mrrs_in_runs[-1]))
                    print("QT: %s\nCode: %s\nAnno: %s\n" % (
                        ids2text(processed_qt[0], self.qt_vocab_inv),
                        ids2text(processed_code[0], self.code_vocab_inv),
                        ids2text(processed_annotation[0], self.qb_vocab_inv)
                    ))

                    # print negative examples
                    for neg_count, (neg_idx, neg_code, neg_qb) in enumerate(zip(neg_indices, neg_codes, neg_qbs)):
                        print("Neg %d: %s, score %.3f" % (neg_count, str(neg_idx), sims_scores[neg_count]))
                        print("Code: %s\nQB: %s\n" % (
                            ids2text(neg_code, self.code_vocab_inv),
                            ids2text(neg_qb, self.qb_vocab_inv)
                        ))
                    print("-" * 5)

            mrr = np.mean(mrrs_in_runs)
            cr_model_mrrs.append(mrr)

        if reward_mode in ["cr", "cr_noqb"]:
            assert len(cr_model_mrrs) == 1
            #return cr_model_mrrs[0]
        elif reward_mode == "cr_diff":
            assert len(cr_model_mrrs) == 2
            # return cr_model_mrrs[0] - cr_model_mrrs[1]
        else:
            raise Exception("Invalid reward mode %s!" % reward_mode)
        return cr_model_mrrs

    def get_reward_in_batch(self, data_name, codes, annotations, qts, qbs,
                            poolsize=50, number_of_runs=1, top_n_results=-1):
        """
        NOTE: Need to double check.
        :param codes: A list of cleaned-up processed codes
        :param annotations: A list of cleaned-up processed annotations produced by the Code Annotation Model
        :param qts: A list of cleaned-up processed QTs
        :param qbs: A list of cleaned-up processed actual Question Bodies
        :param poolsize: Poolsize for negative sampling
        :param number_of_runs: Total number of runs of negative sampling
        :param top_n_results: If required restrict to top n results, -1 use full pool
        :return: Mean MRR score for the given inputs
        """

        processed_qts, processed_codes, processed_annotations = gVar(self._batchify(qts)), \
                                                                gVar(self._batchify(codes)), \
                                                                gVar(self._batchify(annotations))
        qts_repr = self.model.qt_encoding(processed_qts)
        codes_repr = self.model.code_encoding(processed_codes, processed_annotations)
        positive_scores = F.cosine_similarity(codes_repr, qts_repr).data.tolist()

        # collect negative pools
        all_neg_qbs, all_neg_codes = [], []
        for code, annotation, qt, qb in zip(codes, annotations, qts, qbs):
            _, neg_qbs, neg_codes = self._get_negative_pool(data_name,
                (poolsize - 1) * number_of_runs + 1, code, qt, annotation, qb)
            all_neg_qbs.extend(neg_qbs)
            all_neg_codes.extend(neg_codes)

        all_neg_qbs, all_neg_codes = gVar(self._batchify(all_neg_qbs)), \
                                     gVar(self._batchify(all_neg_codes))
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