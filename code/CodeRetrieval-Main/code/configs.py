def get_config(args):
    conf = {
        # data_params
        'workdir': '../data/',
        'ckptdir': '../checkpoint/',

        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3,

        'train_qt': 'sql.train.qt.pkl',
        'train_code': 'sql.train.code.pkl',

        'val_qt': 'codenn_combine_new.sql.dev.qt.pkl' if args.val_setup == "codenn" else 'sql.val.qt.pkl',
        'val_code': 'codenn_combine_new.sql.dev.code.pkl' if args.val_setup == "codenn" else 'sql.val.code.pkl',

        'test_qt': 'codenn_combine_new.sql.eval.qt.pkl' if args.val_setup == "codenn" else 'sql.test.qt.pkl',
        'test_code': 'codenn_combine_new.sql.eval.code.pkl' if args.val_setup == "codenn" else 'sql.test.code.pkl',

        'qt_len': 20,
        'code_len': 120,
        'anno_len': 20,

        'qt_n_words': 7775,  # 4 is added for UNK, EOS, SOS, PAD
        'code_n_words': 7726,
        'anno_n_words': 7775,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',
        'vocab_anno': 'sql.qt.vocab.pkl',

        # training_params
        'use_anno': args.use_anno,

        'batch_size': 1024,
        'nb_epoch': 100,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 50,
        'save_every': 10,
        'patience': 20,
        'reload': 0,  # reload>0, model is reloaded.

        # model_params
        'emb_size': 200,
        # recurrent
        'lstm_dims': 400,  # * 2
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'margin': 0.05,
        'code_encoder': 'bilstm',  # bow, bilstm
    }
    if conf['use_anno']:
        if args.qn_mode == "codenn_gen":
            conf['vocab_anno'] = 'sql.ga.vocab.pkl'
            conf['anno_n_words'] = 827

            conf['train_anno'] = 'sql.train.ga.pkl'
            conf['val_anno'] = 'codenn.sql.dev.ga.pkl' if args.val_setup == "codenn" else 'sql.valid.ga.pkl'
            conf['test_anno'] = 'codenn.sql.eval.ga.pkl' if args.val_setup == "codenn" else 'sql.test.ga.pkl'

        else:
            conf['train_anno'] = 'sql.train.anno_%s.pkl' % args.qn_mode
            conf['val_anno'] = ('codenn_combine_new.sql.dev.anno_%s.pkl' % args.qn_mode) if args.val_setup == "codenn"\
                else ('sql.val.anno_%s.pkl' % args.qn_mode)
            conf['test_anno'] = ('codenn_combine_new.sql.eval.anno_%s.pkl' % args.qn_mode) if args.val_setup == "codenn"\
                else ('sql.test.anno_%s.pkl' % args.qn_mode)

    return conf
