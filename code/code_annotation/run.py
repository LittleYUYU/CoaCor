import os
import subprocess
import os.path
import sys

def preprocess(lang):
    # python run.py preprocess python
    # python run.py preprocess sql

    run = 'python preprocess.py ' \
          '-token_code ../../data/version2/source/%s_index_to_tokenized_code.pkl ' \
          '-token_qb ../../data/version2/source/%s_index_to_tokenized_qb.pkl ' \
          '-token_qt ../../data/version2/source/%s_index_to_tokenized_qt.pkl ' \
          '-split_indices ../../data/version2/split_indices_%s.pkl ' \
          '-code_word2id ../../data/version2/source/Vocab_Files/%s.code.vocab.pkl ' \
          '-qb_word2id ../../data/version2/source/Vocab_Files/%s.qb.vocab.pkl ' \
          '-qt_word2id ../../data/version2/source/Vocab_Files/%s.qt.vocab.pkl ' \
          '-save_data dataset/train/%s.processed_all ' \
          '> log/log.%s.preprocess' % (
          lang, lang, lang, lang, lang, lang, lang, lang, lang
    )
    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def train_a2c(lang, bool_toy, load_from,
              start_reinforce, end_epoch, critic_pretrain_epochs,
              data_type, has_attn, gpus):
    # pretrain: python run.py train_a2c sql 1/0 None None 10 0 text 1 0
    # continue pretraining: python run.py train_a2c sql 1/0 load_from_path None 20 0 text 1 0
    # RL: python run.py train_a2c sql 1/0 11 30 10 text 1 0

    data_name = "_toy" if int(bool_toy) else ""

    arg_str = '-data dataset/train/%s.processed_all.train%s.pt ' \
              '-save_dir dataset/result_%s/ ' \
              '-end_epoch %s ' \
              '-critic_pretrain_epochs %s ' \
              '-data_type %s ' \
              '-has_attn %s ' \
              '-gpus %s ' % (lang, data_name, lang, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus)
    if len(data_name):
        arg_str += '-data_name %s ' % data_name

    if start_reinforce != "None":
        arg_str += '-start_reinforce %s ' % start_reinforce
        log_str = '> log/log.%s.a2c-train%s_RL%s_%s_%s_%s_%s_g%s' % (
            lang, data_name, start_reinforce, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus)
    else:
        log_str = '> log/log.%s.a2c-train%s_noRL_%s_%s_%s_%s_g%s' % (
            lang, data_name, end_epoch, critic_pretrain_epochs, data_type, has_attn, gpus)

    if load_from != "None":
        arg_str += '-load_from %s ' % load_from

    # logging
    arg_str += log_str

    run = 'python a2c-train.py %s' % arg_str

    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def test_a2c(lang, bool_toy, epoch, data_type, has_attn, gpus):
    # python run.py test_a2c sql 1/0 epoch_num text 1 0
    data_name = "_toy" if int(bool_toy) else ""

    run = 'python a2c-train.py ' \
          '-data dataset/train/%s.processed_all.train%s.pt ' \
          '-load_from dataset/result_%s/%smodel_xent_text_1_%s.pt ' \
          '-eval -save_dir . ' \
          '-data_type %s ' \
          '-has_attn %s ' \
          '-gpus %s ' \
          '> log/log.%s.a2c-test_%s_%s_%s_g%s' \
          % (lang, data_name, lang, data_name, epoch, data_type, has_attn, gpus,
             lang, epoch, data_type, has_attn, gpus)
    # % (lang, data_name, lang, data_type, has_attn, gpus)
    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


if sys.argv[1] == 'preprocess':
    preprocess(sys.argv[2])

if sys.argv[1] == 'train_a2c':
    train_a2c(sys.argv[2], sys.argv[3], sys.argv[4],
              sys.argv[5], sys.argv[6], sys.argv[7],
              sys.argv[8], sys.argv[9], sys.argv[10])

if sys.argv[1] == 'test_a2c':
    test_a2c(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
