from __future__ import division

import math
import random
import pdb

import torch
from torch.autograd import Variable

import lib

class Dataset(object):
    def __init__(self, data, batchSize, cuda, data_type, eval=False):
        self.src = data["src"]
        self.tgt = data["tgt"]
        self.qt = data["qt"]
        self.idx = data["indices"]
        self.trees = data["trees"]
        self.leafs = data["leafs"]
        assert(len(self.src) == len(self.tgt))
        self.cuda = cuda

        self.data_type = data_type

        self.batchSize = batchSize
        # self.numBatches = int(math.ceil(len(self.src)/batchSize)-1)
        self.numBatches = int(math.ceil(len(self.src) * 1.0 / batchSize))
        self.eval = eval


    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.qt, self.idx, self.trees, self.leafs))
        random.shuffle(data)
        self.src, self.tgt, self.qt, self.idx, self.trees, self.leafs = zip(*data)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(lib.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, src_lengths = self._batchify(self.src[index*self.batchSize:(index + 1)*self.batchSize], include_lengths=True)
        tgtBatch = self._batchify(self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        qtBatch = self.qt[index * self.batchSize:(index + 1) * self.batchSize]
        idxBatch = self.idx[index * self.batchSize:(index + 1) * self.batchSize]
        indices = range(len(srcBatch))

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            # b = Variable(b, volatile=self.eval)
            return b

        if self.data_type in {"code", "hybrid"}:
            leafBatch, leaf_lengths = self._batchify(self.leafs[index * self.batchSize:(index + 1) * self.batchSize], include_lengths=True)
            srcTrees = self.trees[index * self.batchSize:(index + 1) * self.batchSize]
            src_batch = zip(indices, srcBatch, leafBatch, leaf_lengths, srcTrees, tgtBatch, qtBatch, idxBatch)
            src_batch, src_lengths = zip(*sorted(zip(src_batch, src_lengths), key=lambda x: -x[1]))
            indices, srcBatch, leafBatch, leaf_lengths, srcTrees, tgtBatch, qtBatch, idxBatch = zip(*src_batch)

            tree_lengths = []
            for tree in srcTrees:
                l_c = tree.leaf_count()
                tree_lengths.append(l_c)

            return (wrap(srcBatch), list(src_lengths)), \
                   (srcTrees, tree_lengths, (wrap(leafBatch), list(leaf_lengths))), \
                   wrap(tgtBatch), \
                   indices, \
                   qtBatch, idxBatch

        else:
            src_batch = zip(indices, srcBatch, tgtBatch, qtBatch, idxBatch)
            src_batch, src_lengths = zip(*sorted(zip(src_batch, src_lengths), key=lambda x: -x[1]))
            indices, srcBatch, tgtBatch, qtBatch, idxBatch = zip(*src_batch)

            return (wrap(srcBatch), list(src_lengths)), \
                    None, \
                    wrap(tgtBatch), \
                    indices, \
                    qtBatch, idxBatch

    def __len__(self):
        return self.numBatches


    # def restore_pos(self, sents):
    #     sorted_sents = [None] * len(self.pos)
    #     for sent, idx in zip(sents, self.pos):
    #       sorted_sents[idx] = sent
    #     return sorted_sents
