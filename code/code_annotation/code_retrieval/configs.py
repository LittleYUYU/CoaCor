def get_config(cr_setup):
    conf = {
        'cr_setup': cr_setup,

        # Change it to necessary directory
        'workdir': 'dataset/cr_data/',#'./data/',

        'PAD': 0,
        'BOS': 1,
        'EOS': 2,
        'UNK': 3,

        # 'test_qt': 'sql.test.qt.pkl',
        # 'test_code': 'sql.test.code.pkl',
        # 'test_qb': 'sql.test.qb.pkl',
        #
        # 'valid_qt': 'sql.val.qt.pkl',
        # 'valid_code': 'sql.val.code.pkl',
        # 'valid_qb': 'sql.val.qb.pkl',
        #
        # 'train_qt': 'sql.train.qt.pkl',
        # 'train_code': 'sql.train.code.pkl',
        # 'train_qb': 'sql.train.qb.pkl',

        # 'valid_pos2negs': 'pos2negs_valid_sql.pkl',
        # 'test_pos2negs': 'pos2negs_test_sql.pkl',

        # parameters
        'qt_len': 20, #14,
        'qb_len': 20, #83,
        'code_len': 120, #118,

        'qt_n_words': 4947,  # 4 is added for UNK, EOS, SOS, PAD
        'qb_n_words': 38008,
        'code_n_words': 7726,  # 7734,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',
        'vocab_qb': 'sql.qb.vocab.pkl',

        # training_params
        'batch_size': 256,
        'nb_epoch': 500,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 1000,
        'save_every': 10,
        'patience': 20,
        'reload': -1,  # 970,#epoch that the model is reloaded from . If reload=0, then train from scratch

        # model_params
        'emb_size': 200,
        # 'n_hidden': 400,#number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 400,  # * 2
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'init_embed_weights_qt': None,  # word2vec_100_qt.h5,
        'init_embed_weights_code': None,  # 'word2vec_100_code.h5',
        'init_embed_weights_qb': None,  # 'word2vec_100_qb.h5',
        'margin': 0.05,
        'sim_measure': 'cos',  # similarity measure: gesd, cosine, aesd
        'code_encoder': 'bilstm',  # bow,bilstm
        'use_qb': 1,

        'loss': 'pairwise'

    }

    if cr_setup in {"slrnd", "slrnd_loadPre"}:
        for item in ["train_qt", "train_qb", "train_code", "valid_qt", "valid_qb", "valid_code",
                     "test_qt", "test_qb", "test_code"]:
            conf[item] = conf[item].replace(".pkl", ".slrnd.pkl")

        if cr_setup == "slrnd_loadPre":
            conf['batch_size'] = 1024

    elif cr_setup == "loadPre":
        conf['batch_size'] = 1024
        conf['bow_dropout'] = 0.35
        conf['seqenc_dropout'] = 0.35

    elif cr_setup == "qt_new_cleaned_sl_qb_loadPre_fixPre":
        conf['qb_n_words'] = 7775
        conf['vocab_qb'] = 'sql.qb.vocab.qt_new_cleaned_rl_mrr_qb.pkl'
        conf['lr'] = 0.0001
        conf['bow_dropout'] = 0.35
        conf['seqenc_dropout'] = 0.35
        conf['batch_size'] = 128

    elif cr_setup == "qt_new_cleaned_rl_mrr_qb_loadPre_fixPre":
        conf['qb_n_words'] = 7775
        conf['vocab_qb'] = 'sql.qb.vocab.qt_new_cleaned_rl_mrr_qb.pkl'
        conf['lr'] = 0.0001
        conf['bow_dropout'] = 0.35
        conf['seqenc_dropout'] = 0.35
        conf['batch_size'] = 128

    elif cr_setup == "tp_qt_new_cleaned_rl_mrr_qb":
        conf['tp_weight'] = 0.4
        conf['qb_n_words'] = 7775
        conf['vocab_qb'] = 'sql.qb.vocab.qt_new_cleaned_rl_mrr_qb.pkl'
        conf['bow_dropout'] = 0.0
        conf['seqenc_dropout'] = 0.0
        conf['batch_size'] = 128
        conf['use_code'] = 1

    return conf


def get_config_noQB(cr_setup):
    conf = {
        'cr_setup': cr_setup,

        # Change it to necessary directory
        'workdir': 'dataset/cr_data/',#'./data/',

        'PAD': 0,
        'BOS': 1,
        'EOS': 2,
        'UNK': 3,

        # 'test_qt': 'sql.test.qt.pkl',
        # 'test_code': 'sql.test.code.pkl',
        # 'test_qb': 'sql.test.qb.pkl',
        #
        # 'valid_qt': 'sql.val.qt.pkl',
        # 'valid_code': 'sql.val.code.pkl',
        # 'valid_qb': 'sql.val.qb.pkl',
        #
        # 'train_qt': 'sql.train.qt.pkl',
        # 'train_code': 'sql.train.code.pkl',
        # 'train_qb': 'sql.train.qb.pkl',

        # 'valid_pos2negs': 'pos2negs_valid_sql.pkl',
        # 'test_pos2negs': 'pos2negs_test_sql.pkl',

        # parameters
        'qt_len': 20,
        'qb_len': 20,
        'code_len': 120,

        'qt_n_words': 4947,  # 4 is added for UNK, EOS, SOS, PAD
        'qb_n_words': 38008,
        'code_n_words': 7726,  # 7734,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',
        'vocab_qb': 'sql.qb.vocab.pkl',

        # training_params
        'batch_size': 128,
        'nb_epoch': 500,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 1000,
        'save_every': 10,
        'patience': 20,
        'reload': -1,  # 970,#epoch that the model is reloaded from . If reload=0, then train from scratch

        # model_params
        'emb_size': 200,
        # 'n_hidden': 400,#number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 400,  # * 2
        'bow_dropout': 0.35,  # dropout for BOW encoder
        'seqenc_dropout': 0.35,  # dropout for sequence encoder encoder
        'init_embed_weights_qt': None,  # word2vec_100_qt.h5,
        'init_embed_weights_code': None,  # 'word2vec_100_code.h5',
        'init_embed_weights_qb': None,  # 'word2vec_100_qb.h5',
        'margin': 0.05,
        'sim_measure': 'cos',  # similarity measure: gesd, cosine, aesd
        'code_encoder': 'bilstm',  # bow,bilstm
        'use_qb': 0,

        'loss': 'pairwise'
    }
    return conf