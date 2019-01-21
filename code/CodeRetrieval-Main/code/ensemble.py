# ensemble.py
import pickle
from collections import defaultdict
import numpy as np
from utils import *


def get_rank(sims):
    negsims = np.negative(sims)
    predict = np.argsort(negsims)
    predict = [int(k) for k in predict]

    return predict


def weighting_scores_codenn(data_name, pure_size, weight, sims_collection1, sims_collection2,
                     preprocess_collection1=None, preprocess_collection2=None, bool_by_run=False):
    mrrs, accs, maps, ndcgs = [], [], [], []
    mrrs_per_run = defaultdict(list)

    run_idx = 1
    count = 0

    for collect_idx, (sims_list1, sims_list2) in enumerate(zip(sims_collection1, sims_collection2)):
        for idx in range(len(sims_list1)):
            item1 = np.array(sims_list1[idx])
            if preprocess_collection1 == "devide_max":
                denominator = max(abs(item1))
                item1 = item1 / denominator
            elif preprocess_collection1 == "softmax":
                denominator = sum(np.exp(item1))
                item1 = np.exp(item1) / denominator

            item2 = np.array(sims_list2[idx])
            if preprocess_collection2 == "devide_max":
                denominator = max(abs(item2))
                item2 = item2 / denominator
            elif preprocess_collection2 == "softmax":
                denominator = sum(np.exp(item2))
                item2 = np.exp(item2) / denominator

            sims = weight * item1 + (1 - weight) * item2
            predict = get_rank(sims)
            real = [0]  # index of the positive sample

            mrrs.append(MRR(real, predict))
            accs.append(ACC(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))

            mrrs_per_run[run_idx].append(mrrs[-1])

        count += 1
        if count == pure_size:
            count = 0
            run_idx += 1

    print("Data %s, weight %.3f:" % (data_name, weight))
    print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
        len(mrrs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
    
    if bool_by_run:
        average_mrr_by_run = []
        print("#of runs: %d." % len(mrrs_per_run))
        for run_idx, values in mrrs_per_run.items():
            print("Run %d: size %d, average %.4f." % (run_idx, len(values), np.average(values)))
            average_mrr_by_run.append(np.average(values))
        print("Flat mean %.4f" % np.average(average_mrr_by_run))
        print("Flat stdev %.4f" % np.std(average_mrr_by_run))

    return mrrs, accs, maps, ndcgs


def weighting_scores_staqc(data_name, weight, sims_collection1, sims_collection2,
                           preprocess_collection1=None, preprocess_collection2=None):
    mrrs, accs, maps, ndcgs = [], [], [], []

    counter = 0
    for collect_idx, (item1, item2) in enumerate(zip(sims_collection1, sims_collection2)):
        if counter == 50:
            counter = 0 #reset

        item1 = np.array(item1)
        if preprocess_collection1 == "devide_max":
            denominator = max(abs(item1))
            item1 = item1 / denominator
        elif preprocess_collection1 == "softmax":
            denominator = sum(np.exp(item1))
            item1 = np.exp(item1) / denominator

        item2 = np.array(item2)
        if preprocess_collection2 == "devide_max":
            denominator = max(abs(item2))
            item2 = item2 / denominator
        elif preprocess_collection2 == "softmax":
            denominator = sum(np.exp(item2))
            item2 = np.exp(item2) / denominator

        sims = weight * item1 + (1 - weight) * item2
        predict = get_rank(sims)
        real = [counter]  # index of the positive sample

        mrrs.append(MRR(real, predict))
        accs.append(ACC(real, predict))
        maps.append(MAP(real, predict))
        ndcgs.append(NDCG(real, predict))

        counter += 1

    print("Data %s, weight %.3f:" % (data_name, weight))
    print('Size={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(
        len(mrrs), np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))

    return mrrs, accs, maps, ndcgs


if __name__ == "__main__":
    def ensemble_codenn_set():
        # reranking: QC + QN
        for data_name in ["val", "test"]:
            # QC: DCS
            qc_load_dir = "../checkpoint/QC_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"

            # QC: codenn
            # codenn_name = "dev" if data_name == "val" else "eval"
            # code_only = pickle.load(open(
            #     "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/%s_scores_collection.pkl" % codenn_name))

            # QN-RLMRR
            qn_load_dir = "../checkpoint/QN_rl_mrr_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"

            qc = pickle.load(open(qc_load_dir + "collect_sims_codenn_%s.pkl" % data_name))
            qn = pickle.load(open(qn_load_dir + "collect_sims_codenn_%s.pkl" % data_name))

            pure_size = 111 if data_name == "val" else 100
            for weight in [x * 0.1 for x in range(11)]:
                mrrs, _, _, _ = weighting_scores_codenn(data_name, pure_size, weight, qn, qc)  # "devide_max"
            print("-" * 50)

    def ensemble_staqc_set():
        for data_name in ["val", "test"]:
            # QC: DCS
            qc_load_dir = "../checkpoint/QC_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"

            # QC: codenn
            # codenn_name = "dev" if data_name == "val" else "eval"
            # code_only = pickle.load(open(
            #     "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/%s_scores_collection.pkl" % codenn_name))

            # QN-RLMRR
            qn_load_dir = "../checkpoint/QN_rl_mrr_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"

            qc = pickle.load(open(qc_load_dir + "collect_sims_staqc_%s.pkl" % data_name))
            qn = pickle.load(open(qn_load_dir + "collect_sims_staqc_%s.pkl" % data_name))

            for weight in [x * 0.1 for x in range(11)]:
                mrrs, _, _, _ = weighting_scores_staqc(data_name, weight, qn, qc) #"devide_max"
                print("Average mrr: %.4f, stdev: %.4f." % (np.average(mrrs), np.std(mrrs)))

            print("-" * 50)

    # ensemble_codenn_set()
    ensemble_staqc_set()