import sys
import torch
import torch.utils.data as data
# import tables
import random
import numpy as np
import pickle

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
use_cuda = torch.cuda.is_available()


class StaQCDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, conf, dataset):
        cand_str = 'anno' if conf['use_anno'] else 'code'

        f_qt, qt_len, f_cand, cand_len = \
            conf['%s_qt' % dataset], conf['qt_len'], conf['%s_%s' % (dataset, cand_str)], \
            conf['%s_len' % cand_str]

        self.qt_len = qt_len
        self.cand_len = cand_len

        # Load Vocab Files
        self.path = conf['workdir']
        self.vocab_qt = limit_vocab(load_dict(self.path + conf['vocab_qt']), conf['qt_n_words'])
        self.vocab_cand = limit_vocab(load_dict(self.path + conf['vocab_%s' % cand_str]), conf['%s_n_words' % cand_str])

        self.PAD_token = conf['<pad>']

        # Processing Text
        # 1. Load Preprocessed Dataset
        print("loading data...")
        print("Loading qt from %s..." % (data_dir + f_qt))
        self.list_of_qt_strings = pickle.load(open(data_dir + f_qt))
        print("Loading candidate from %s..." % (data_dir + f_cand))
        self.list_of_cand_strings = pickle.load(open(data_dir + f_cand))

        # Convert string of indices to list of indices
        self.processed_qt = self.get_preprocessed_text(self.list_of_qt_strings)
        self.processed_cand = self.get_preprocessed_text(self.list_of_cand_strings)

        # Data Length
        self.data_len = len(self.processed_qt)
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = np.append(seq, [self.PAD_token] * maxlen)
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen

        return seq, act_len

    def get_preprocessed_text(self, list_of_strings):
        preprocessed_text = []
        bool_strings = all(isinstance(n, str) for n in list_of_strings)
        for data_point in list_of_strings:
            if bool_strings:
                string = data_point
            else:
                string = data_point[1]
            words = map(int, string.split())
            if self.PAD_token in words:
                words = words[:words.index(self.PAD_token)]
            preprocessed_text.append(np.array(words))
        return preprocessed_text

    def __getitem__(self, offset):
        # Question Title
        qt = self.processed_qt[offset].astype('int64')
        qt, qt_len = self.pad_seq(qt, self.qt_len)

        # < QT,Candidate>
        good_cand = self.processed_cand[offset].astype('int64')
        good_cand, good_cand_len = self.pad_seq(good_cand, self.cand_len)

        # < QT,~Candidate>
        rand_offset = random.randint(0, self.data_len - 1)
        while rand_offset == offset:
            rand_offset = random.randint(0, self.data_len - 1)

        bad_cand = self.processed_cand[rand_offset].astype('int64')
        bad_cand, bad_cand_len = self.pad_seq(bad_cand, self.cand_len)

        return qt, good_cand, bad_cand

    def __len__(self):
        return self.data_len


class CodennDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self, data_dir, conf, dataset):
        cand_str = 'anno' if conf['use_anno'] else 'code'

        f_qt, qt_len, f_cand, cand_len = \
            conf['%s_qt' % dataset], conf['qt_len'], conf['%s_%s' % (dataset, cand_str)], \
            conf['%s_len' % cand_str]

        self.qt_len = qt_len
        self.cand_len = cand_len

        # Load Vocab Files
        self.path = conf['workdir']
        self.vocab_qt = limit_vocab(load_dict(self.path + conf['vocab_qt']), conf['qt_n_words'])
        self.vocab_cand = limit_vocab(load_dict(self.path + conf['vocab_%s' % cand_str]), conf['%s_n_words' % cand_str])

        self.PAD_token = conf['<pad>']

        # Processing Text
        # 1. Load Preprocessed Dataset
        print("loading data...")
        print("Loading qt from %s..." % (data_dir + f_qt))
        self.list_of_qt_strings = pickle.load(open(data_dir + f_qt))
        print("Loading candidate from %s..." % (data_dir + f_cand))
        self.list_of_cand_strings = pickle.load(open(data_dir + f_cand))

        # Convert string of indices to list of indices
        self.processed_qt = self.get_preprocessed_text(self.list_of_qt_strings)
        self.processed_cand = self.get_preprocessed_text(self.list_of_cand_strings)

        # Data Length
        self.data_len = len(self.processed_qt)
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if isinstance(seq, list):
            seq = [self.pad_seq(sub_seq, maxlen) for sub_seq in seq]
        else:
            assert isinstance(seq, np.ndarray)
            if len(seq) < maxlen:
                seq = np.append(seq, [self.PAD_token] * maxlen)
                seq = seq[:maxlen]
            else:
                seq = seq[:maxlen]
        return seq

    def get_preprocessed_text(self, list_of_strings):
        preprocessed_text = []
        bool_strings = all(isinstance(n, str) for n in list_of_strings)
        for data_point in list_of_strings:
            if bool_strings:
                string = data_point
                words = map(int, string.split())
                preprocessed_text.append(np.array(words))
            elif len(data_point) == 3:
                string_list = data_point
                words_list = []
                for string in string_list:
                    words_list.append(np.array(map(int, string.split())))
                preprocessed_text.append(words_list)
            else:
                string = data_point[1]
                words = map(int, string.split())
                preprocessed_text.append(np.array(words))
        return preprocessed_text

    def __getitem__(self, offset):
        # Question Title
        qt = self.processed_qt[offset]
        if isinstance(qt, np.ndarray):
            qt = qt.astype('int64')
        elif isinstance(qt, list):
            assert len(qt) == 3
            for i in range(3):
                qt[i] = qt[i].astype('int64')
        qt = self.pad_seq(qt, self.qt_len)

        # < QT,Candidate>
        good_cand = self.processed_cand[offset].astype('int64')
        good_cand = self.pad_seq(good_cand, self.cand_len)

        return qt, good_cand

    def __len__(self):
        return self.data_len


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


def limit_vocab(old_vocab_dict, vocab_length):
    new_vocab_dict = {}
    for word, key in old_vocab_dict.iteritems():
        if key < vocab_length:
            new_vocab_dict[word] = key
    return new_vocab_dict
