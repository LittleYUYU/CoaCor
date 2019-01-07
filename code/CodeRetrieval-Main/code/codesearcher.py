import os
import random
import numpy as np
import math
import argparse
import pickle
import pdb
# import logging

import torch
from torch import optim
import torch.nn.functional as F

from utils import *
from configs import get_config
from data import StaQCDataset, CodennDataset
from models import *

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(message)s")

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['workdir']

    #######################
    # Model Loading / saving #####
    #######################
    def save_model(self, model):
        if not os.path.exists(self.conf['model_directory']):
            os.makedirs(self.conf['model_directory'])
        torch.save(model.state_dict(), self.conf['model_directory'] + 'best_model.ckpt')

    def load_model(self, model):
        assert os.path.exists(self.conf['model_directory'] + 'best_model.ckpt'), 'Weights for saved model not found'
        model.load_state_dict(torch.load(self.conf['model_directory'] + 'best_model.ckpt'))


    #######################
    # Training #####
    #######################
    def train(self, model, val_setup="staqc"):
        """
        Trains an initialized model
        :param model: Initialized model
        :return: None
        """
        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        max_patience = self.conf['patience']

        train_set = StaQCDataset(self.path, self.conf, "train")
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                  shuffle=True, drop_last=True, num_workers=1)

        # val set
        if val_setup == "staqc":
            val = StaQCDataset(self.path, self.conf, "val")
        elif val_setup == "codenn":
            val = CodennDataset(self.path, self.conf, "val")
        else:
            raise Exception("Invalid val_setup %s!" % val_setup)

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            if val_setup == "codenn":
                _, max_mrr, _, _ = self.eval_codenn(model, 50, val)
            else:
                _, max_mrr, _, _ = self.eval(model, 50, val)
        else:
            max_mrr = -1

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses = []

            model = model.train()

            for qts, good_cands, bad_cands in data_loader:
                qts, good_cands, bad_cands = gVar(qts), gVar(good_cands), gVar(bad_cands)

                loss, good_scores, bad_scores = model(qts, good_cands, bad_cands)

                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every == 0:
                    print('epo:[%d/%d] itr:%d Loss=%.5f' % (epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []
                itr = itr + 1

            if epoch % valid_every == 0:
                print("validating..")
                if val_setup == "codenn":
                    print("val_setup: codenn")
                    acc1, mrr, map, ndcg = self.eval_codenn(model, 50, val)
                else:
                    acc1, mrr, map, ndcg = self.eval(model, 50, val) 

                if mrr > max_mrr:
                    self.save_model(model)
                    patience = 0
                    print("Model improved. Saved model at %d epoch" % epoch)
                    max_mrr = mrr
                else:
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1

            if patience >= max_patience:
                print("Patience Limit Reached. Stopping Training")
                break

    ########################
    # Evaluation on CodeNN #
    ########################
    def eval_codenn(self, model, poolsize, dataset, bool_collect=False):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize, shuffle=False,
                                                  num_workers=1)
        model = model.eval()

        sims_collection = []
        accs, mrrs, maps, ndcgs = [], [], [], []
        for qts, cands in data_loader:
            cands = gVar(cands)

            cands_repr = model.cand_encoding(cands)

            if isinstance(qts, list):
                assert len(qts) == 3
                qts = [gVar(qts_i) for qts_i in qts]
            else:
                qts = [gVar(qts)]

            sims_per_qts = []
            for qts_i in qts:
                qt_repr = model.qt_encoding(qts_i)

                sims = model.scoring(qt_repr, cands_repr).data.cpu().numpy()
                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                predict = [int(k) for k in predict]
                real = [0]  # index of the positive sample

                # save
                sims_per_qts.append(sims)

                mrrs.append(MRR(real, predict))
                accs.append(ACC(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))

            sims_collection.append(sims_per_qts)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_codenn_%s.pkl" % dataset)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(mrrs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    #######################
    # Evaluation on StaQC #####
    #######################
    def eval(self, model, poolsize, dataset, bool_collect=False):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=poolsize,
                                                  shuffle=False, drop_last=True,
                                                  num_workers=1)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []

        sims_collection = []
        for qts, cands, _ in data_loader:
            qts, cands = gVar(qts), gVar(cands)
            qts_repr = model.qt_encoding(qts)
            cands_repr = model.cand_encoding(cands)

            _poolsize = len(qts) if bool_collect else poolsize # true poolsize
            for i in range(_poolsize):
                _qts_repr = qts_repr[i].expand(_poolsize, -1)

                scores = model.scoring(_qts_repr, cands_repr).data.cpu().numpy()
                neg_scores = np.negative(scores)
                predict = np.argsort(neg_scores)
                predict = [int(k) for k in predict]
                real = [i]  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
                sims_collection.append(scores)

        if bool_collect:
            save_path = os.path.join(self.conf['model_directory'], "collect_sims_staqc_%s.pkl" % dataset)
            print("Save collection to %s" % save_path)
            pickle.dump(sims_collection, open(save_path, "wb"))

        print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
            len(accs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search Model")
    parser.add_argument("-m", "--mode", choices=["train", "eval", "collect"],
                        default='train',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluates models on a test set;"
                             " the `collect` mode collects model scores;",
                        required=True)
    parser.add_argument("--use_anno", type=int, default=0, help="Training QN CR?", required=True)
    parser.add_argument("--reload", type=int, default=-1, help=" Should I reload saved model, yes if reload>0?",
                        required=True)
    # model setup
    parser.add_argument("--dropout", type=float, default=0.0, help="What is the dropout?", required=True)
    parser.add_argument("--emb_size", type=int, default=100, help="What is the embedding size?", required=True)
    parser.add_argument("--lstm_dims", type=int, default=200, help="What is the lstm dimension?", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="What is the batch size?", required=True)
    # dataset setup
    parser.add_argument("--qn_mode", type=str, default="sl",
                        choices=["sl", "rl_bleu", "rl_mrr",
                                 "codenn_gen"], help="Data set to use.")
    parser.add_argument('--val_setup', type=str, choices=["staqc", "codenn"], default="staqc",
                        help="Which val set during training?")
    parser.add_argument('--eval_setup', type=str, choices=["staqc", "codenn"], default="staqc",
                        help="Which dataset to evaluate?")

    # optimization
    parser.add_argument("--lr", type=float, default=0.001, help="What is the learning rate?")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin for pairwise loss.")
    parser.add_argument("--optimizer", type=str,
                        choices=["adam", "adagrad", "sgd", "rmsprop", "asgd", "adadelta"],
                        default="adam", help="Which optimizer to use?")
    return parser.parse_args()


def create_model_name_string(c):
    string1 = 'qtlen_{}_codelen_{}_qtnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
        format(c['qt_len'], c['code_len'], c['qt_n_words'], c['code_n_words'],
               c['batch_size'], c['optimizer'], str(c['lr'])[2:] if c['lr'] < 1.0 else str(c['lr']))
    string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}'. \
        format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:])
    string3 = '_codeenc_{}'.format(c['code_encoder'])
    string = string1 + string2 + string3

    return string


if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args)

    # hyper-params
    conf['bow_dropout'] = args.dropout
    conf['seqenc_dropout'] = args.dropout
    conf['emb_size'] = args.emb_size
    conf['lstm_dims'] = args.lstm_dims
    conf['batch_size'] = args.batch_size
    conf['lr'] = args.lr
    conf['reload'] = args.reload
    conf['optimizer'] = args.optimizer
    print("Modeling QN: ", conf['use_anno'])

    if conf['reload'] <= 0 and args.mode in {'eval', 'collect'}:
        print("For eval/collect mode, please give reload=1. If you looking to train the model, change the mode to train. "
              "\n Note: Train overrides previously saved model, if it had exactly the same parameters")
    else:
        if args.mode == 'train':
            print("Warning: Train overrides previously saved model, if it had exactly the same parameters")
            print("If retraining the model from previous check point, set reload >0 to start training from previous "
                  "checkpoint")

        print(" Code encoder : ", conf['code_encoder'])
        print(" Dropout : ", conf['seqenc_dropout'])
        print(" Embedding size : ", conf['emb_size'])
        print(" LSTM hidden dimension : ", conf['lstm_dims'])
        print(" Margin: ", conf['margin'])
        print(" Optimizer: ", conf['optimizer'])

        # Creating unique model string based on parameters defined. Helps differentiate between different runs of model
        model_string = create_model_name_string(conf)

        if args.use_anno:
            model_dir_str = "QN_%s" % args.qn_mode
        else:
            model_dir_str = "QC"
        model_dir_str += "_val%s" % args.val_setup # val data

        conf['model_directory'] = conf['ckptdir'] + '%s/' % model_dir_str + model_string + '/'

        if not os.path.exists(conf['model_directory']):
            os.makedirs(conf['model_directory'])
        print(" Model Directory : ")
        print(conf['model_directory'])

        searcher = CodeSearcher(conf)

        #####################
        # Define model ######
        #####################
        print('Building Model')
        model = JointEmbeder(conf)
        print("model: ", model)

        if conf['reload'] > 0:
            if args.mode in {'eval', 'collect'}:
                print("Reloading saved model for evaluating/collecting results")
            else:
                print("Reloading saved model for Re-training")
            searcher.load_model(model)

        if torch.cuda.is_available():
            print('using GPU')
            model = model.cuda()
        else:
            print('using CPU')

        print("\nParameter requires_grad state: ")
        for name, param in model.named_parameters():
            print name, param.requires_grad
        print("")

        if conf['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for AdaGrad while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=conf['lr'], momentum=0.9)
            print("Recommend lr 0.1 for SGD (momentum 0.9) while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for RMSprop while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'asgd':
            optimizer = optim.ASGD(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.01 for ASGD while using %.5f." % conf['lr'])
        elif conf['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=conf['lr'])
            print("Recommend lr 1.00 for Adadelta while using %.5f." % conf['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
            print("Recommend lr 0.001 for Adam while using %.5f." % conf['lr'])

        if args.mode == 'train':
            print('Training Model')
            searcher.train(model, val_setup=args.val_setup)

        elif args.mode == 'eval':
            print('Evaluating Model')

            if args.eval_setup == "codenn":
                val = CodennDataset(conf['workdir'], conf, "val")
                searcher.eval_codenn(model, 50, val)
                test = CodennDataset(conf['workdir'], conf, "test")
                searcher.eval_codenn(model, 50, test)
            else:
                val = StaQCDataset(conf['workdir'], conf, "val")
                searcher.eval(model, 50, val)
                test = StaQCDataset(conf['workdir'], conf, "test")
                searcher.eval(model, 50, test)

        elif args.mode == 'collect':
            print('Collecting outputs...')
            for dataset in ['val', 'test']:
                # searcher.eval_codenn(model, 50, dataset, bool_collect=True)
                searcher.eval(model, 50, dataset, bool_collect=True)

        else:
            print("Please provide a Valid argument for mode - train/eval")
