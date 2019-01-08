#!/bin/sh

python run.py train_a2c sql 0 1 20 0 cr dataset/result_sql_qt_new_cleaned/model_xent_attn1_brnn1_decay15_dropout0.5/model_xent_attn1_brnn1_decay15_dropout0.5_14.pt 15 64 10 1 1 45 0.0001 512 512 0.5 64 0 1  

