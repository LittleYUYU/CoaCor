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
    def ensemble(dataset, qc, qn):
        if qc == "dcs":
            qc_load_dir = "../checkpoint/QC_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"
        elif qc == "codenn":  
            qc_load_dir = "../checkpoint/QC_CODENN_saved_models_CSCR_sql_qtNoStem_dropout7/"
        else:
            raise Exception("Invalid QC model %s!" % qc)

        if qn == "rl_mrr":
            # QN-RLMRR
            qn_load_dir = "../checkpoint/QN_rl_mrr_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"
        elif qn == "mle":
            # QN-MLE
            qn_load_dir = "../checkpoint/QN_sl_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_512" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_5_seqencdropout_5_codeenc_bilstm/"
        elif qn == "rl_bleu":
            qn_load_dir = "../checkpoint/QN_rl_bleu_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_128" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_5_seqencdropout_5_codeenc_bilstm/"
        elif qn == "codenn_gen":
            qn_load_dir = "../checkpoint/QN_codenn_gen_valcodenn/qtlen_20_codelen_120_qtnwords_7775_codenwords_7726_batch_256" \
                          "_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_35_seqencdropout_35_codeenc_bilstm/"
        else:
            raise Exception("Invalid QN model %s!" % qn)

        # reranking: QC + QN
        for data_name in ["val", "test"]:
            if qc == "codenn":
                if dataset == "staqc":
                    codenn_tag = "staqc_%s" % data_name
                else:
                    codenn_tag = "dev" if data_name == "val" else "eval"
                qc_results = pickle.load(open(qc_load_dir + "%s_scores_collection.pkl" % codenn_tag))
                preprocess_collection2 = "devide_max"
            else:
                qc_results = pickle.load(open(qc_load_dir + "collect_sims_%s_%s.pkl" % (dataset, data_name)))
                preprocess_collection2 = None
            qn_results = pickle.load(open(qn_load_dir + "collect_sims_%s_%s.pkl" % (dataset, data_name)))
            
            if dataset == "codenn":
                print("Ensemble results on codenn...")
                pure_size = 111 if data_name == "val" else 100
                for weight in [x * 0.1 for x in range(11)]:
                    mrrs, _, _, _ = weighting_scores_codenn(data_name, pure_size, weight, qn_results, qc_results, preprocess_collection2=preprocess_collection2) 
                print("-" * 50)
            elif dataset == "staqc":
                print("Ensemble results on staqc")
                for weight in [x * 0.1 for x in range(11)]:
                    mrrs, _, _, _ = weighting_scores_staqc(data_name, weight, qn_results, qc_results, preprocess_collection2=preprocess_collection2)
                    print("Average mrr: %.4f, stdev: %.4f." % (np.average(mrrs), np.std(mrrs)))
                print("-" * 50)
            else:
                raise Exception("Invalid dataset %s!" % dataset)

    ensemble("codenn", "codenn", "rl_mrr")
    ensemble("staqc", "codenn", "rl_mrr")
