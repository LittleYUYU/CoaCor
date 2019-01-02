import os
# import subprocess
import os.path
import sys
import ntpath

def preprocess(lang):
    # python run.py preprocess sql

    run = 'python preprocess.py ' \
          '-token_code ../../data/version2/source/%s_index_to_tokenized_code.pkl ' \
          '-token_qb ../../data/version3/source/%s_index_to_tokenized_qt.pkl ' \
          '-token_qt ../../data/version2/source/%s_index_to_tokenized_qt.pkl ' \
          '-split_indices ../../data/version2/origin/split_indices_%s_cleaned.pkl ' \
          '-code_word2id ../../data/version2/source/Vocab_Files/%s.code.vocab.pkl ' \
          '-src_seq_length 120 -tgt_seq_length 20 '\
          '-qt_word2id ../../data/version2/source/Vocab_Files/%s.qt.vocab.pkl ' \
          '-save_data dataset/train_qt_new_cleaned/%s.processed_all ' \
          '> log_qt_new_cleaned/log.%s.preprocess' % (
          lang, lang, lang, lang, lang, lang, lang, lang)

    # '-qb_word2id ../../data/version2/source/Vocab_Files/%s.qb.vocab.pkl ' \

    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()


def train_a2c(lang, bool_toy, bool_has_baseline, qb_or_qt, max_predict_length, pred_mask,
              sent_reward, cr_setup, cr_replace_all_train, cr_replace_all_eval,
              cr_qt_candidates_train, cr_qt_candidates_eval,
              load_from, start_reinforce, end_epoch, critic_pretrain_epochs,
              attn, brnn, decay_ep, lr, emb_dim, h_dim, dropout, batch_size, pretrain_emb, layers):
    # pretrain: python run.py train_a2c sql 1/0 1/0 {qb|qt|qt_origin} max_len pred_mask {cr|cr_diff|bleu|cr_noqb} {"default"|"slrnd"} cr_replace_all_train, cr_replace_all_eval None None 10 0 attributes...
    # pretrain critic: python run.py train_a2c sql 1/0 1 {qb|qt|qt_origin} max_len pred_mask {cr|cr_diff|bleu|cr_noqb} {"default"|"slrnd"} cr_replace_all_train, cr_replace_all_eval load_from start_ep end_ep pretrain_ep ...
    # RL: python run.py train_a2c sql 1/0 1/0 {qb|qt|qt_origin} max_len pred_mask {cr|cr_diff|bleu} {"default"|"slrnd"} cr_replace_all_train cr_replace_all_eval load_from_path 11 30 10 attributes...

    data_name = "_toy" if int(bool_toy) else ""

    arg_str = '-lang %s ' \
              '-data dataset/train_%s/%s.processed_all.train%s.pt ' \
              '-save_dir dataset/result_%s_%s/ -max_predict_length %s -predict_mask %s ' \
              '-end_epoch %s ' \
              '-critic_pretrain_epochs %s ' \
              '-sent_reward %s -cr_setup %s -cr_replace_all_train %s -cr_replace_all_eval %s ' \
              '-cr_qt_candidates_train %s -cr_qt_candidates_eval %s ' \
              '-has_attn %s -has_baseline %s -start_decay_at %s -word_vec_size %s -rnn_size %s ' \
              '-dropout %s -batch_size %s -layers %s ' \
              '-gpus 0 ' % (
        lang, qb_or_qt, lang, data_name, lang, qb_or_qt, max_predict_length, pred_mask,
        end_epoch, critic_pretrain_epochs,
        sent_reward, cr_setup, cr_replace_all_train, cr_replace_all_eval,
        cr_qt_candidates_train, cr_qt_candidates_eval,
        attn, bool_has_baseline, decay_ep, emb_dim, h_dim, dropout, batch_size, layers)

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

    if pretrain_emb == '1':
        arg_str += '-load_embedding_from dataset/train_%s/ ' % qb_or_qt

    # show_str as
    show_str = "_attn%s_brnn%s" % (attn, brnn)
    if decay_ep != "5":
        if int(decay_ep) == int(end_epoch) + 1:
            show_str += "_nodecay"
        else:
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
    if start_reinforce != "None": #RL signal
        show_str += "_Sent%s" % sent_reward
    if cr_setup != "default":
        show_str += "_cr%s" % cr_setup
    if pretrain_emb == '1':
        show_str += '_embPre'
    if int(layers) > 1:
        show_str += "_layers%s" % layers
    if cr_replace_all_train == '1' and sent_reward != "cr_noqb":
        show_str += "_rAllTrain"
    if cr_replace_all_eval == '1' and sent_reward != "cr_noqb":
        show_str += "_rAllEval"
    if cr_qt_candidates_train == '1' and sent_reward in {"cr", "cr_diff"}:
        show_str += "_rQtCandTrain"
    if cr_qt_candidates_eval == '1' and sent_reward in {"cr", "cr_diff"}:
        show_str += "_rQtCandEval"
    if pred_mask == '1':
        show_str += "_predMask"

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


def test_a2c(lang, bool_toy, qb_or_qt, max_predict_length, pred_mask, sent_reward,
             cr_setup, cr_replace_all_eval, cr_qt_candidates_eval, load_from_path, eval_set, collect,
             attn, layers):
    # python run.py test_a2c sql 1/0 qb_or_qt max_len pred_mask cr_or_bleu {"default"|"slrnd"|"loadPre"|"slrnd_loadPre"}
    # cr_replace_all_eval cr_qt_candidates_eval load_from_path default|codenn|empty collect attn layers
    data_name = "_toy" if int(bool_toy) else ""

    arg_str = '-lang %s -load_from %s -sent_reward %s -cr_setup %s ' \
              '-max_predict_length %s -predict_mask %s ' \
              '-eval -save_dir . -cr_replace_all_eval %s -cr_qt_candidates_eval %s ' \
              '-has_attn %s ' \
              '-gpus 0 -show_str None -layers %s ' \
              % (lang, load_from_path, sent_reward, cr_setup, max_predict_length, pred_mask,
                 cr_replace_all_eval, cr_qt_candidates_eval, attn, layers)

    if eval_set in {"default", "empty"}:
        arg_str += '-data dataset/train_%s/%s.processed_all.train%s.pt ' % (qb_or_qt, lang, data_name)
        if eval_set == "empty":
            arg_str += '-empty_anno '
    elif eval_set == "codenn":
        arg_str += '-data dataset/train_%s/%s.processed_all.train%s.pt ' % (qb_or_qt, lang, data_name)
        arg_str += '-eval_codenn '
    elif eval_set == "codenn_all":
        arg_str += '-data dataset/train_%s/%s.processed_all.codenn_all.pt ' % (qb_or_qt, lang)
        arg_str += '-eval_codenn_all '

    if collect == "1":
        arg_str += '-collect_anno '

    checkpoint_name = ntpath.basename(load_from_path)
    log_str = '> log_%s/log.%s.a2c-test%s_Sent%s_cr%s_%s%s%s' % (
        qb_or_qt, lang, data_name, sent_reward, cr_setup, checkpoint_name,
        (eval_set if eval_set != "default" else ""), ("_collect" if collect == '1' else ""))
    if cr_replace_all_eval == '1' and sent_reward != "cr_noqb":
        log_str += "_rAllEval"
    if cr_qt_candidates_eval == '1' and sent_reward in {"cr", "cr_diff"}:
        log_str += "_rQtCandEval"
    if pred_mask == '1':
        log_str += "_predMask"
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
              sys.argv[14], sys.argv[15], sys.argv[16],
              sys.argv[17], sys.argv[18], sys.argv[19],
              sys.argv[20], sys.argv[21], sys.argv[22],
              sys.argv[23], sys.argv[24], sys.argv[25],
              sys.argv[26], sys.argv[27])

if sys.argv[1] == 'test_a2c':
    test_a2c(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
             sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
             sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13],
             sys.argv[14], sys.argv[15])
