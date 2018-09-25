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
    parser.add_argument("-token_code", required=True, help="Path to tokenized source data (C)")
    parser.add_argument("-token_qb", required=True, help="Path to tokenized target data (QB)")
    parser.add_argument("-token_qt", required=True, help="Path to tokenized query (QT)")
    parser.add_argument("-split_indices", required=True, help="Path to the indices of train/valid/test split")
    parser.add_argument('-save_data', required=True, help="Output file for the prepared data")
    # parser.add_argument('-src_vocab_size', type=int, default=50000, help="Size of the source vocabulary")
    # parser.add_argument('-tgt_vocab_size', type=int, default=50000, help="Size of the target vocabulary")
    parser.add_argument('-src_min_freq', type=int, default=2, help="Minimum word frequency for source")
    parser.add_argument('-tgt_min_freq', type=int, default=2, help="Minimum word frequency for target")
    parser.add_argument('-code_word2id', default=None, help="Given source vocabulary")
    parser.add_argument('-qb_word2id', default=None, help="Given target vocabulary")
    parser.add_argument('-qt_word2id', default=None, help="Given query vocabulary")
    parser.add_argument('-src_seq_length', type=int, default=0, help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=0, help="Maximum target sequence length to keep.")
    parser.add_argument('-seed', type=int, default=3435, help="Random seed")

    opt = parser.parse_args()
    return opt


def makeVocabulary(name, tokenized_data, size=0, min_freq=1):
    "Construct the word and feature vocabs."
    print(Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD)
    print("Build vocabulary for %s ..." % name)

    vocab = lib.Dict([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
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


def makeData(token_src, token_tgt, token_qt, srcDicts, tgtDicts, qtDicts):
    src_ids, tgt_ids, qt_ids, trees = [], [], [], []
    ignored, exceps, empty = 0, 0, 0

    for sent_src, sent_tgt, sent_qt in zip(token_src, token_tgt, token_qt):
        if len(sent_src) == 0 or len(sent_tgt) == 0 or len(sent_qt) == 0:
            empty += 1
            continue

        if len(sent_src) <= opt.src_seq_length and len(sent_tgt) <= opt.tgt_seq_length:
            src_ids += [srcDicts.convertToIdx(sent_src, Constants.UNK_WORD)]
            tgt_ids += [tgtDicts.convertToIdx(sent_tgt, Constants.UNK_WORD, eosWord=Constants.EOS_WORD)]
            qt_ids += [qtDicts.convertToIdx(sent_qt, Constants.UNK_WORD)]
            trees += []
        else:
            ignored += 1

    print(('Prepared %d sentences ' +
           '(%d ignored due to src len > %d or tgt len > %d)' +
           '(%d ignored due to empty.)') %
          (len(src_ids), ignored, opt.src_seq_length, opt.tgt_seq_length, empty))
    # print(('Prepared %d sentences ' + '(%d ignored due to Exception)') % (len(src_ids), exceps))
    return src_ids, tgt_ids, qt_ids, trees


def makeDataGeneral(name, token_src, token_tgt, token_qt, dicts):
    print('Preparing ' + name + '...')
    res = {}
    res['src'], res['tgt'], res['qt'], res['trees'] = makeData(token_src, token_tgt, token_qt,
                                                               dicts['src'], dicts['tgt'], dicts['qt'])
    return res


def main():
    torch.manual_seed(opt.seed)

    # load meta data
    idx2tokenized_src = pickle.load(open(opt.token_code)) #src=code
    idx2tokenized_tgt = pickle.load(open(opt.token_qb)) #tgt=qb
    idx2tokenized_qt = pickle.load(open(opt.token_qt))
    split_indices = pickle.load(open(opt.split_indices)) # a dict of {train/valid/test: iids}
    print("Data loaded!")

    if opt.code_word2id is not None:
        code_word2id = pickle.load(open(opt.code_word2id))
        print("Code vocab loaded!")
    if opt.qb_word2id is not None:
        qb_word2id = pickle.load(open(opt.qb_word2id))
        print("QB vocab loaded!")
    if opt.qt_word2id is not None:
        qt_word2id = pickle.load(open(opt.qt_word2id))
        print("QT vocab loaded!")

    train_src, train_tgt, train_qt, \
    valid_src, valid_tgt, valid_qt,\
    test_src, test_tgt, test_qt = [], [], [], [], [], [], [], [], []
    for item, (container_src, container_tgt, container_qt) in zip(["train", "valid", "test"],
                                                    [(train_src, train_tgt, train_qt),
                                                     (valid_src, valid_tgt, valid_qt),
                                                     (test_src, test_tgt, test_qt)]):
        iids = split_indices[item]
        for qt_idx, qb_idx, code_idx in iids:
            container_src.append(idx2tokenized_src[code_idx])
            container_tgt.append(idx2tokenized_tgt[qb_idx])
            container_qt.append(idx2tokenized_qt[qt_idx])

    print("train, valid, test size: %d, %d, %d" % (
        len(split_indices["train"]), len(split_indices["valid"]), len(split_indices["test"])))
    assert len(train_src) == len(train_tgt) == len(train_qt)

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
        src_seq_length = src_lengths[int(len(src_lengths) * 0.95)]
        tgt_seq_length = tgt_lengths[int(len(tgt_lengths) * 0.95)]
        print("%s data: length size to cover 95 precent examples, src %d, tgt %d" % (
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
    if opt.code_word2id is None:
        dicts['src'] = makeVocabulary('code', train_src, 0, opt.src_min_freq)
    else:
        dicts['src'] = transformVocabulary('code', code_word2id)
    if opt.qb_word2id is None:
        dicts['tgt'] = makeVocabulary('annotation', train_tgt, 0, opt.tgt_min_freq)
    else:
        dicts['tgt'] = transformVocabulary('annotation', qb_word2id)
    assert opt.qt_word2id is not None
    dicts['qt'] = transformVocabulary('query', qt_word2id)

    saveVocabulary("code (src)", dicts['src'], opt.save_data + '.code.dict')
    saveVocabulary("annotation (tgt)", dicts['tgt'], opt.save_data + '.anno.dict')

    save_data = {}
    save_data['dicts'] = dicts
    save_data['train_xe'] = makeDataGeneral('train_xe', train_src, train_tgt, train_qt, dicts)
    # save_data['train_pg'] = makeDataGeneral('train_pg', train_src, train_tgt, dicts)
    save_data['train_pg'] = copy.deepcopy(save_data['train_xe'])
    save_data['valid_xe'] = makeDataGeneral('valid_xe', valid_src, valid_tgt, valid_qt, dicts)
    valid_pg = random.sample(zip(valid_src, valid_tgt, valid_qt), 2000)
    valid_src_pg, valid_tgt_pg, valid_qt_pg = zip(*valid_pg)
    save_data['valid_pg'] = makeDataGeneral('valid_pg', list(valid_src_pg), list(valid_tgt_pg), list(valid_qt_pg), dicts)
    save_data['test'] = makeDataGeneral('test', test_src, test_tgt, test_qt, dicts)

    print("Saving data to \"" + opt.save_data + ".train.pt\"...")
    torch.save(save_data, opt.save_data + ".train.pt")

    # toy data for quick test
    toy_data = {}
    toy_data['dicts'] = save_data['dicts']
    for item in ["train_xe", "train_pg", "valid_xe", "valid_pg", "test"]:
        toy_data[item] = {}
        for k,v in save_data[item].items():
            toy_data[item][k] = v[:1000]
    print("Saving toy data to \"" + opt.save_data + ".train_toy.pt\"...")
    torch.save(toy_data, opt.save_data + ".train_toy.pt")

    # # word2vec dump
    # code_sentences = [token for sent in train_src for token in sent]
    # print('code_sentences: ', train_xe_code_sentences[0])
    # print('comment_sentences: ', train_xe_comment_sentences[0])
    # code_w2v_model = gensim.models.Word2Vec(train_xe_code_sentences, size=512, window=5, min_count=5, workers=16)
    # code_w2v_model.save(opt.save_data + '.train_xe.code.gz')
    # comment_w2v_model = gensim.models.Word2Vec(train_xe_comment_sentences, size=512, window=5, min_count=5, workers=16)
    # comment_w2v_model.save(opt.save_data + '.train_xe.comment.gz')


if __name__ == "__main__":
    global opt
    opt = get_opt()
    main()