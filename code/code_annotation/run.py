import os
# import subprocess
import os.path
import sys
import ntpath

def preprocess(lang):
    # python run.py preprocess python
    # python run.py preprocess sql

    run = 'python preprocess.py ' \
          '-token_code ../../data/version2/source/%s_index_to_tokenized_code.pkl ' \
          '-token_qb ../../data/qt_anno/%s_index_to_anno_tokenized_qt.pkl ' \
          '-token_qt ../../data/version2/source/%s_index_to_tokenized_qt.pkl ' \
          '-split_indices ../../data/version2/origin/split_indices_simplified_%s.pkl ' \
          '-code_word2id ../../data/version2/source/Vocab_Files/%s.code.vocab.pkl ' \
          '-qb_word2id ../../data/version2/source/Vocab_Files/%s.qb.vocab.pkl ' \
          '-qt_word2id ../../data/version2/source/Vocab_Files/%s.qt.vocab.pkl ' \
          '-save_data dataset/train_qt_origin/%s.processed_all ' \
          '> log_qt_origin/log.%s.preprocess' % (
          lang, lang, lang, lang, lang, lang, lang, lang, lang
    )
    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def train_a2c(lang, bool_toy, bool_has_baseline, qb_or_qt, load_from,
              start_reinforce, end_epoch, critic_pretrain_epochs,
              attn, brnn, decay_ep, lr, emb_dim, h_dim, dropout, batch_size):
    # pretrain: python run.py train_a2c sql 1/0 1/0 qb_or_qt None None 10 0 attributes...
    # continue pretraining: python run.py train_a2c sql 1/0 1/0 qb_or_qt load_from_path None 20 0 attributes...
    # RL: python run.py train_a2c sql 1/0 1/0 qb_or_qt load_from_path 11 30 10 attributes...

    data_name = "_toy" if int(bool_toy) else ""

    arg_str = '-data dataset/train_%s/%s.processed_all.train%s.pt ' \
              '-save_dir dataset/result_%s_%s/ ' \
              '-end_epoch %s ' \
              '-critic_pretrain_epochs %s ' \
              '-data_type text ' \
              '-has_attn %s -has_baseline %s -start_decay_at %s -word_vec_size %s -rnn_size %s ' \
              '-dropout %s -batch_size %s ' \
              '-gpus 0 ' % (
        qb_or_qt, lang, data_name, lang, qb_or_qt, end_epoch, critic_pretrain_epochs,
        attn, bool_has_baseline, decay_ep, emb_dim, h_dim, dropout, batch_size)

    if brnn == '1':
        arg_str += '-brnn '

    if len(data_name):
        arg_str += '-data_name %s ' % data_name

    if start_reinforce != "None":
        # RL
        if lr != '0.0001':
            arg_str += '-reinforce_lr %s ' % lr
        arg_str += '-start_reinforce %s ' % start_reinforce
        log_str = '> log_%s/log.%s.a2c-train%s_RL%s_%s_%s_%s' % (
            qb_or_qt, lang, data_name, "hasBaseline" if bool_has_baseline == '1' else "noBaseline",
            start_reinforce, end_epoch, critic_pretrain_epochs)
    else:
        # SL
        if lr != '0.001':
            arg_str += '-lr %s ' % lr
        log_str = '> log_%s/log.%s.a2c-train%s_noRL_%s_%s' % (
            qb_or_qt, lang, data_name, end_epoch, critic_pretrain_epochs)

    if load_from != "None":
        arg_str += '-load_from %s ' % load_from

    # show_str as
    show_str = "_attn%s_brnn%s" % (attn, brnn)
    if decay_ep != "5":
        show_str += "_decay%s" % decay_ep
    if start_reinforce == "None" and lr != "0.001":
        show_str += "_lr%s" % lr
    elif start_reinforce != "None" and lr != "0.0001":
        show_str += "_rflr%s" % lr
    if emb_dim != "512":
        show_str += "_emb%s" % emb_dim
    if h_dim != "512":
        show_str += "_rnn%s" % h_dim
    if dropout != "0.3":
        show_str += "_dropout%s" % dropout
    if batch_size != "64":
        show_str += "_bs%s" % batch_size
    arg_str += '-show_str %s ' % show_str

    log_str += show_str
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


def test_a2c(lang, bool_toy, qb_or_qt, sent_reward, load_from_path, eval_set, attn):
    # python run.py test_a2c sql 1/0 qb_or_qt cr_or_bleu load_from_path default|codenn|empty attn
    data_name = "_toy" if int(bool_toy) else ""

    assert sent_reward in {"cr", "bleu"}

    arg_str = '-load_from %s -sent_reward %s ' \
          '-eval -save_dir . ' \
          '-data_type text ' \
          '-has_attn %s ' \
          '-gpus 0 -show_str None ' \
          % (load_from_path, sent_reward, attn)

    if eval_set in {"default", "empty"}:
        arg_str += '-data dataset/train_%s/%s.processed_all.train%s.pt ' % (qb_or_qt, lang, data_name)
        if eval_set == "empty":
            arg_str += '-empty_anno '
    elif eval_set == "codenn":
        arg_str += '-data dataset/%s.processed_all.codenn_test.pt ' % lang
        arg_str += '-eval_codenn '

    checkpoint_name = ntpath.basename(load_from_path)
    log_str = '> log_%s/log.%s.a2c-test%s_Sent%s_%s%s' % (
        qb_or_qt, lang, data_name, sent_reward, checkpoint_name,
        (eval_set if eval_set != "default" else ""))
    arg_str += log_str

    run = 'python a2c-train.py %s' % arg_str

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
              sys.argv[8], sys.argv[9], sys.argv[10],
              sys.argv[11], sys.argv[12], sys.argv[13],
              sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17])

if sys.argv[1] == 'test_a2c':
    test_a2c(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
