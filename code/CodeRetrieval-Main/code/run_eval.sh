#!/bin/sh

# test QN-RL-MRR on StaQC
python codesearcher.py --mode eval --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 1 --qn_mode rl_mrr --reload 1 --eval_setup staqc | tee log/qn_rlmrr_test_staqc_drop35_emb200_lstm400_bs256.log

# test QN-RL-MRR on codenn set
python codesearcher.py --mode eval --dropout 0.35 --emb_size 200 --lstm_dims 400 --batch_size 256 --val_setup codenn --use_anno 1 --qn_mode rl_mrr --reload 1 --eval_setup codenn | tee log/qn_rlmrr_test_codenn_drop35_emb200_lstm400_bs256.log

