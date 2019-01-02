import datetime
import math
import os
import time

from torch.autograd import Variable
import torch

import lib
import pdb


class ReinforceTrainer(object):
    def __init__(self, actor, critic, train_data, eval_data, metrics, dicts, optim, critic_optim, opt):
        self.actor = actor
        self.critic = critic

        self.train_data = train_data
        self.eval_data = eval_data
        print("\n* eval_data size: %d" % len(eval_data.src))
        self.evaluator = lib.Evaluator(actor, metrics, dicts, opt)

        self.actor_loss_func = metrics["xent_loss"]
        self.critic_loss_func = metrics["critic_loss"]
        self.sent_reward_func = metrics["sent_reward"]["train"]

        self.dicts = dicts

        self.optim = optim
        self.critic_optim = critic_optim

        self.max_length = opt.max_predict_length
        # self.pert_func = opt.pert_func
        self.opt = opt

    def train(self, start_epoch, end_epoch, pretrain_critic, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        self.optim.last_loss = None
        self.optim.set_lr(self.opt.reinforce_lr)

        if self.opt.has_baseline:
            self.critic_optim.last_loss = None
            #  Use large learning rate for critic during pre-training.
            if pretrain_critic:
                self.critic_optim.set_lr(1e-3)
            else:
                self.critic_optim.set_lr(self.opt.reinforce_lr)

        for epoch in range(start_epoch, end_epoch + 1):
            print("* REINFORCE epoch *")
            print("Actor optim lr: %g; Critic optim lr: %g" % (
                self.optim.lr, self.critic_optim.lr if self.opt.has_baseline else 0))
            if self.optim.lr < lib.Constants.MIN_LR:
                print("(Actor) Early stop when learning rate is lower than %s" % (str(lib.Constants.MIN_LR)))
                break

            if pretrain_critic:
                print("Pretrain critic...")

            train_reward, critic_loss = self.train_epoch(epoch, pretrain_critic, False)
            print("Train sentence reward: %.2f" % (train_reward * 100))
            print("Critic loss: %g" % critic_loss)

            # if not pretrain_critic:
            # evaluate the actor model on validation set
            valid_loss, valid_sent_reward, valid_corpus_reward = self.evaluator.eval(self.eval_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print("Validation perplexity: %.2f" % valid_ppl)
            print("Validation sentence reward: %.2f" % (valid_sent_reward * 100))
            print("Validation corpus reward: %.2f" % (valid_corpus_reward * 100))

            # else:
            #     print("Pretraining critic...no eval on actor...")

            if not pretrain_critic:
                self.optim.updateLearningRate(-valid_sent_reward, epoch)
                # Actor and critic use the same lr when jointly trained.
                # if not pretrain_critic:
                if self.opt.has_baseline:
                    self.critic_optim.set_lr(self.optim.lr)

            checkpoint = {
                "model": self.actor,
                "dicts": self.dicts,
                "opt": self.opt,
                "epoch": epoch,
                "optim": self.optim
            }
            if self.opt.has_baseline:
                checkpoint.update({"critic": self.critic, "critic_optim": self.critic_optim})

            save_name = "%smodel_rf_%s%s" % (
                self.opt.data_name, "hasBaseline" if self.opt.has_baseline else "noBaseline", self.opt.show_str)
            if pretrain_critic:
                save_name += "_pretrain"
            else:
                save_name += "_reinforce"

            save_dir = os.path.join(self.opt.save_dir, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_name = os.path.join(save_dir, "%s_%s.pt" % (save_name, epoch))

            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch, pretrain_critic, no_update):
        self.actor.train()
        total_reward, report_reward = 0, 0
        total_critic_loss, report_critic_loss = 0, 0
        total_sents, report_sents = 0, 0
        total_words, report_words = 0, 0
        last_time = time.time()

        self.train_data.shuffle()

        for i in range(len(self.train_data)):
            batch = self.train_data[i]
            targets = batch[2]
            qts = batch[4]
            attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
            batch_size = targets.size(1)

            self.actor.zero_grad()
            if self.opt.has_baseline:
                self.critic.zero_grad()

            # Sample translations
            if self.opt.has_attn:
                self.actor.decoder.attn.applyMask(attention_mask)
            samples, outputs = self.actor.sample(batch, self.max_length)

            # Calculate rewards
            # s0 = time.time()
            rewards, samples = self.sent_reward_func(
                samples.t().tolist(), targets.data.t().tolist(),
                codes=batch[0][0].t().tolist(),
                qts=[item.tolist() for item in qts],
                tgt_dict=self.dicts['tgt'])
            reward = sum(rewards)
            # print("Eval one batch time: %.2f" % (time.time() - s0))

            samples = Variable(torch.LongTensor(samples).t().contiguous())
            rewards = Variable(torch.FloatTensor([rewards] * samples.size(0)).contiguous())
            if self.opt.cuda:
                samples = samples.cuda()
                rewards = rewards.cuda()

            critic_weights = samples.ne(lib.Constants.PAD).float()
            num_words = critic_weights.data.sum()
            if self.opt.has_baseline:
                # Update critic.
                baselines = self.critic((batch[0], batch[1], samples, batch[3]), eval=False, regression=True)

                critic_loss = self.critic.backward(baselines, rewards, critic_weights, num_words, self.critic_loss_func, regression=True)
                self.critic_optim.step()

            else:
                critic_loss = 0

            # Update actor
            if not pretrain_critic and not no_update:
                if self.opt.has_baseline:
                    # Subtract baseline from reward
                    norm_rewards = Variable((rewards - baselines).data)
                else:
                    norm_rewards = Variable(rewards.data)
                actor_weights = norm_rewards * critic_weights
                # TODO: can use PyTorch reinforce() here but that function is a black box.
                # This is an alternative way where you specify an objective that gives the same gradient
                # as the policy gradient's objective, which looks much like weighted log-likelihood.
                actor_loss = self.actor.backward(outputs, samples, actor_weights, 1, self.actor_loss_func)
                self.optim.step()
            else:
                actor_loss = 0

            # Gather stats
            total_reward += reward
            report_reward += reward
            total_sents += batch_size
            report_sents += batch_size
            total_critic_loss += critic_loss
            report_critic_loss += critic_loss
            total_words += num_words
            report_words += num_words
            self.opt.iteration += 1
            print ("iteration: %s, loss: %s " % (self.opt.iteration, actor_loss))
            print ("iteration: %s, reward: %s " % (self.opt.iteration, (report_reward / report_sents) * 100))

            if i % self.opt.log_interval == 0 and i > 0:
                print("""Epoch %3d, %6d/%d batches; actor reward: %.4f; critic loss: %f; %5.0f tokens/s; %s elapsed""" %
                      (epoch, i, len(self.train_data), (report_reward / report_sents) * 100,
                      report_critic_loss / report_words,
                      report_words / (time.time() - last_time),
                      str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                report_reward = report_sents = report_critic_loss = report_words = 0
                last_time = time.time()

        return total_reward / total_sents, total_critic_loss / total_words

