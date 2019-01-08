def get_config():
    conf = {
        # Change it to necessary directory
        'workdir': 'dataset/cr_data/',

        'PAD': 0,
        'BOS': 1,
        'EOS': 2,
        'UNK': 3,

        'train_qt': 'sql.train.qt.pkl',
        'train_code': 'sql.train.code.pkl',

        # parameters
        'qt_len': 20,
        'code_len': 120,

        'qt_n_words': 7775,  # 4 is added for UNK, EOS, SOS, PAD
        'code_n_words': 7726,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',

        # model
        'checkpoint': 'dataset/cr_data/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256'
                      '_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35'
                      '_codeenc_bilstm/best_model.ckpt',
        'use_anno': 0,
        'emb_size': 200,
        'lstm_dims': 400,
        'margin': 0.05,
        'code_encoder': 'bilstm',
        'bow_dropout': 1.0,
        'seqenc_dropout': 1.0
    }
    return conf
