import datetime
import math
import os
import time
import pdb

import torch

import lib
import sys


class Trainer(object):
    def __init__(self, model, train_data, eval_data, metrics, dicts, optim, opt, DEV=None):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        print("\n* eval_data size: %d" % len(eval_data.src))
        self.DEV = DEV
        if DEV is not None:
            print("* DEV size: %d" % len(DEV.src))
        else:
            print("* No DEV")
        self.evaluator = lib.Evaluator(model, metrics, dicts, opt)
        self.loss_func = metrics["xent_loss"]
        self.dicts = dicts
        self.optim = optim
        self.opt = opt

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        for epoch in range(start_epoch, end_epoch + 1):
            print("* XENT epoch *")
            print("Model optim lr: %g" % self.optim.lr)

            if self.optim.lr < lib.Constants.MIN_LR:
                print("Early stop when learning rate is lower than %s" % (str(lib.Constants.MIN_LR)))
                break

            train_loss = self.train_epoch(epoch)
            print('Train perplexity: %.2f' % math.exp(min(train_loss, 100)))

            valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %.2f' % valid_ppl)
            print('Validation sentence reward: %.2f' % (valid_sent_reward * 100))
            print('Validation corpus reward: %.2f' % (valid_corpus_reward * 100))

            if self.DEV is not None:
                dev_loss, dev_sent_reward, dev_corpus_reward = self.evaluator.eval(self.DEV)
                dev_ppl = math.exp(min(dev_loss, 100))
                print('DEV perplexity: %.2f' % dev_ppl)
                print('DEV sentence reward: %.2f' % (dev_sent_reward * 100))
                print('DEV corpus reward: %.2f' % (dev_corpus_reward))

            self.optim.updateLearningRate(valid_loss, epoch)

            checkpoint = {
                'model': self.model,
                'dicts': self.dicts,
                'opt': self.opt,
                'epoch': epoch,
                'optim': self.optim,
            }
            # model_name = os.path.join(self.opt.save_dir, "model_%d.pt" % epoch)
            save_name = "%smodel_xent%s" % (self.opt.data_name, self.opt.show_str)
            save_dir = os.path.join(self.opt.save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_name = os.path.join(save_dir, "%s_%s.pt" % (save_name, epoch))

            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch):
        self.model.train()

        self.train_data.shuffle()

        total_loss, report_loss = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()
        # batch_order = torch.randperm(len(self.train_data))
        for i in range(len(self.train_data)): #
            batch = self.train_data[i] # batch_order[i]
            self.model.zero_grad()
            targets = batch[2]
            attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()

            if self.opt.has_attn:
                self.model.decoder.attn.applyMask(attention_mask)

            outputs = self.model(batch, eval=False)

            weights = targets.ne(lib.Constants.PAD).float()
            num_words = weights.data.sum()
            loss = self.model.backward(outputs, targets, weights, num_words, self.loss_func)
            self.opt.iteration += 1
            print("iteration: %s, loss: %s " % (self.opt.iteration, loss))

            self.optim.step()

            report_loss += loss
            total_loss += loss
            total_words += num_words
            report_words += num_words
            if i % self.opt.log_interval == 0 and i > 0:
                print("""Epoch %3d, %6d/%d batches; perplexity: %8.2f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data), math.exp(report_loss / report_words),
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_loss = report_words = 0
                last_time = time.time()

        return total_loss / total_words