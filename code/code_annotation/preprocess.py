# -*- coding: utf-8 -*-
import lib
import argparse
import torch
import codecs
import lib.data.Constants as Constants
import ast, asttokens
import sys
# from lib.data.Tree import *
import re
import gensim
import pickle
import copy
import numpy as np
import pdb
import random


def get_opt():
    parser = argparse.ArgumentParser(description='preprocess.py')
    parser.add_argument("-token_src", required=True, help="Path to tokenized source data")
    parser.add_argument("-token_tgt", required=True, help="Path to tokenized target data")
    parser.add_argument("-split_indices", required=True, help="Path to the indices of train/valid/test split")
    parser.add_argument('-save_data', required=True, help="Output file for the prepared data")
    # parser.add_argument('-src_vocab_size', type=int, default=50000, help="Size of the source vocabulary")
    # parser.add_argument('-tgt_vocab_size', type=int, default=50000, help="Size of the target vocabulary")
    parser.add_argument('-src_min_freq', type=int, default=2, help="Minimum word frequency for source")
    parser.add_argument('-tgt_min_freq', type=int, default=2, help="Minimum word frequency for target")
    parser.add_argument('-src_word2id', default=None, help="Given source vocabulary")
    parser.add_argument('-tgt_word2id', default=None, help="Given target vocabulary")
    parser.add_argument('-src_seq_length', type=int, default=0, help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=0, help="Maximum target sequence length to keep.")
    parser.add_argument('-seed', type=int, default=3435, help="Random seed")

    opt = parser.parse_args()
    return opt


def makeVocabulary(name, tokenized_data, size=0, min_freq=1):
    "Construct the word and feature vocabs."
    print(Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, Constants.UNK_WORD)
    print("Build vocabulary for %s ..." % name)

    vocab = lib.Dict([Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, Constants.UNK_WORD])
    for sent in tokenized_data:
        for token in sent:
            vocab.add(token)

    originalSize = vocab.size()

    if size != 0:
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d, limit size %d)' % (vocab.size(), originalSize, size))
    elif min_freq > 1:
        vocab = vocab.prune_by_freq(min_freq)
        print('Created dictionary of size %d (pruned by freq from %d, limit freq %d)' % (vocab.size(), originalSize, min_freq))
    else:
        print('Created dictionary of size %d' % (vocab.size()))

    return vocab


def transformVocabulary(name, word2id):
    "Transform a given word_to_id dict to lib.Dict vocabulary."
    print("Build vocabulary for %s ..." % name)

    vocab = lib.Dict([Constants.PAD_WORD, Constants.BOS_WORD, Constants.EOS_WORD, Constants.UNK_WORD])
    vocab.labelToIdx = word2id
    vocab.idxToLabel = {idx:label for label, idx in word2id.items()}

    print('Created dictionary of size %d' % vocab.size())

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(token_src, token_tgt, indices, srcDicts, tgtDicts, bool_ignore=True):

    src_ids, tgt_ids, pos_indices = [], [], []
    ignored, exceps, empty = 0, 0, 0

    for sent_src, sent_tgt, sent_qt, sent_idx in zip(token_src, token_tgt, token_qt, indices):
        if len(sent_src) == 0 or len(sent_tgt) == 0:
            empty += 1
            continue

        if len(sent_src) <= opt.src_seq_length and len(sent_tgt) <= opt.tgt_seq_length:
            src_ids += [srcDicts.convertToIdx(sent_src, Constants.UNK_WORD)]
            tgt_ids += [tgtDicts.convertToIdx(sent_tgt, Constants.UNK_WORD, eosWord=Constants.EOS_WORD)]
            pos_indices += [sent_idx]
        else:
            if bool_ignore:
                ignored += 1
            else:
                src_ids += [srcDicts.convertToIdx(sent_src[:opt.src_seq_length], Constants.UNK_WORD)]
                tgt_ids += [tgtDicts.convertToIdx(sent_tgt[:opt.tgt_seq_length], Constants.UNK_WORD, eosWord=Constants.EOS_WORD)]
                pos_indices += [sent_idx]

    print(('Prepared %d sentences ' +
           '(%d ignored due to src len > %d or tgt len > %d)' +
           '(%d ignored due to empty.)') %
          (len(src_ids), ignored, opt.src_seq_length, opt.tgt_seq_length, empty))
    # print(('Prepared %d sentences ' + '(%d ignored due to Exception)') % (len(src_ids), exceps))
    return src_ids, tgt_ids, pos_indices


def makeDataGeneral(name, token_src, token_tgt, indices, dicts, bool_ignore=True):
    print('Preparing ' + name + '...')
    res = {}
    res['src'], res['tgt'], res['indices'] = makeData(token_src, token_tgt, indices,
                                                      dicts['src'], dicts['tgt'],
                                                      bool_ignore=bool_ignore)
    return res


def main():
    torch.manual_seed(opt.seed)

    # load meta data
    idx2tokenized_src = pickle.load(open(opt.token_src)) #src=code
    idx2tokenized_tgt = pickle.load(open(opt.token_tgt)) #tgt=annotation
    split_indices = pickle.load(open(opt.split_indices)) # a dict of {train/valid/test: iids}
    split_indices["valid"] = split_indices["valid"][:(len(split_indices["valid"]) // 50 * 50)]
    split_indices["test"] = split_indices["test"][:(len(split_indices["test"]) // 50 * 50)] #poolsize = 50

    print("Data loaded!")

    if opt.src_word2id is not None:
        src_word2id = pickle.load(open(opt.src_word2id))
        print("src vocab loaded!")
    if opt.tgt_word2id is not None:
        tgt_word2id = pickle.load(open(opt.tgt_word2id))
        print("tgt vocab loaded!")

    train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt = [], [], [], [], [], []
    train_indices, valid_indices, test_indices = [], [], []
    for item, (container_src, container_tgt, container_indices) in zip(["train", "valid", "test"],
                                                    [(train_src, train_tgt, train_indices),
                                                     (valid_src, valid_tgt, valid_indices),
                                                     (test_src, test_tgt, test_indices)]):
        iids = split_indices[item]
        for qt_idx, qb_idx, code_idx in iids:
            container_src.append(idx2tokenized_src[code_idx])
            container_tgt.append(idx2tokenized_tgt[qb_idx])
            container_indices.append((qt_idx, qb_idx, code_idx))

    print("train, valid, test size: %d, %d, %d" % (
        len(split_indices["train"]), len(split_indices["valid"]), len(split_indices["test"])))
    assert len(train_src) == len(train_tgt)

    # average lengths distribution
    src_seq_lengths, tgt_seq_lengths = [], []
    for item, (container_src, container_tgt) in zip(["train", "valid", "test"],
                                                    [(train_src, train_tgt), (valid_src, valid_tgt),
                                                     (test_src, test_tgt)]):
        src_lengths = [len(sent) for sent in container_src]
        tgt_lengths = [len(sent) for sent in container_tgt]
        print("%s data: average length of src %.3f, tgt %.3f" % (
            item, np.average(src_lengths), np.average(tgt_lengths)
        ))
        src_lengths.sort()
        tgt_lengths.sort()
        src_seq_length = src_lengths[int(len(src_lengths) * 0.90)]
        tgt_seq_length = tgt_lengths[int(len(tgt_lengths) * 0.90)]
        print("%s data: length size to cover 90 precent examples, src %d, tgt %d" % (
            item, src_seq_length, tgt_seq_length
        ))
        src_seq_lengths.append(src_seq_length)
        tgt_seq_lengths.append(tgt_seq_length)

    if opt.src_seq_length == 0:
        opt.src_seq_length = max(src_seq_lengths)
    if opt.tgt_seq_length == 0:
        opt.tgt_seq_length = max(tgt_seq_lengths)
    print("src_seq_length %d, tgt_seq_length %d" % (opt.src_seq_length, opt.tgt_seq_length))

    dicts = dict()
    if opt.src_word2id is None:
        dicts['src'] = makeVocabulary('code', train_src, 0, opt.src_min_freq)
    else:
        dicts['src'] = transformVocabulary('code', src_word2id)
    if opt.tgt_word2id is None:
        dicts['tgt'] = makeVocabulary('annotation', train_tgt, 0, opt.tgt_min_freq)
    else:
        dicts['tgt'] = transformVocabulary('annotation', tgt_word2id)

    saveVocabulary("code (src)", dicts['src'], opt.save_data + '.code.dict')
    saveVocabulary("annotation (tgt)", dicts['tgt'], opt.save_data + '.anno.dict')

    save_data = {}
    save_data['dicts'] = dicts
    save_data['train'] = makeDataGeneral('train', train_src, train_tgt, train_indices, dicts, bool_ignore=False)
    save_data['valid'] = makeDataGeneral('valid', valid_src, valid_tgt, valid_indices, dicts, bool_ignore=False)
    save_data['test'] = makeDataGeneral('test', test_src, test_tgt, test_indices, dicts, bool_ignore=False)

    print("Saving data to \"" + opt.save_data + ".train.pt\"...")
    torch.save(save_data, opt.save_data + ".train.pt")

    # word2vec dump
    code_w2v_model = gensim.models.Word2Vec(train_src, size=512, window=5, min_count=2, workers=16)
    code_w2v_model.save(opt.save_data + '.train_xe.src.gz')
    comment_w2v_model = gensim.models.Word2Vec(train_tgt, size=512, window=5, min_count=2, workers=16)
    comment_w2v_model.save(opt.save_data + '.train_xe.tgt.gz')


if __name__ == "__main__":
    global opt
    opt = get_opt()
    main()