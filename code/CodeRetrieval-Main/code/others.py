# others.py
from utils import *
import pickle
import pdb
from collections import defaultdict


def get_toy_data(data_dir, lang):
    for data_name in ["train", "valid", "test"]:
        print("Working on %s set..." % data_name)
        qt = pickle.load(open(data_dir + "%s.%s.qt.slrnd.pkl" % (lang, data_name)))
        code = pickle.load(open(data_dir + "%s.%s.code.slrnd.pkl" % (lang, data_name)))
        qb = pickle.load(open(data_dir + "%s.%s.qb.slrnd.pkl" % (lang, data_name)))

        toy_qt, toy_code, toy_qb = qt[:1000], code[:1000], qb[:1000]

        pickle.dump(toy_qt, open(data_dir + "%s.%s.qt.slrnd_toy.pkl" % (lang, data_name), "wb"))
        pickle.dump(toy_code, open(data_dir + "%s.%s.code.slrnd_toy.pkl" % (lang, data_name), "wb"))
        pickle.dump(toy_qb, open(data_dir + "%s.%s.qb.slrnd_toy.pkl" % (lang, data_name), "wb"))


def get_CR_eval_from_disordered(data_name, disordered_items, ordered_old_items):
    idx2item = {item[0]:item for item in disordered_items}
    ordered_items = []
    for k,v in ordered_old_items:
        ordered_items.append(idx2item[k])
    print("%s set, size %d" % (data_name, len(ordered_items)))
    return ordered_items


def get_concise_codenn_complete_set(data_name, qts, qbs, codes):
    new_qts, new_qbs, new_codes = [], [], []

    size = len(qts) // (3 * 50)

    for idx in range(size):
        _qts = qts[idx * 150: (idx + 1) * 150]
        _qbs = qbs[idx * 150: idx * 150 + 50]
        _codes = codes[idx * 150: idx * 150 + 50]

        new_qts.extend([[_qts[0], _qts[50], _qts[100]]] * 50)
        new_qbs.extend(_qbs)
        new_codes.extend(_codes)

    print("Data %s set, size %.3f" % (data_name, len(new_qts)))

    return new_qts, new_qbs, new_codes


def get_rank(sims):
    negsims = np.negative(sims)
    predict = np.argsort(negsims)
    predict = [int(k) for k in predict]

    return predict


def weighting_scores(data_name, pure_size, weight, sims_collection1, sims_collection2,
                     preprocess_collection1=None, preprocess_collection2=None):
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

    # print("Size %d, %d" % (len(sims_collection1), len(sims_collection2)))
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


def collect_weighting_examples(weight, sims_collection1, sims_collection2):

    example_indices_1help2 = []
    example_indices_2help1 = []

    for collect_idx, (sims_list1, sims_list2) in enumerate(zip(sims_collection1, sims_collection2)):
        for idx in range(len(sims_list1)):
            pred1 = get_rank(sims_list1[idx])
            rank1 = pred1.index(0)
            pred2 = get_rank(sims_list2[idx])
            rank2 = pred2.index(0)
            pred_w = get_rank(weight * sims_list1[idx] + (1 - weight) * sims_list2[idx])
            rank_w = pred_w.index(0)

            if rank_w < rank2:
                example_indices_1help2.append((collect_idx, idx, (rank1, rank2, rank_w)))

            elif rank_w < rank1:
                example_indices_2help1.append((collect_idx, idx, (rank1, rank2, rank_w)))
    return example_indices_1help2, example_indices_2help1


def show_examples(example_indices, qts, qbs, codes, qt_vocab, qb_vocab, code_vocab, bool_print_winner_anno=False):
    qt_vocab_inv = {v:k for k,v in qt_vocab.items()}
    qb_vocab_inv = {v:k for k,v in qb_vocab.items()}
    code_vocab_inv = {v:k for k,v in code_vocab.items()}

    def _ids2sent(ids, id2word):
        return " ".join([id2word[int(id)] for id in ids])

    for item in example_indices:
        if bool_print_winner_anno:
            collect_idx, qt_idx, (rank1, rank2, rankw), pred = item
        else:
            collect_idx, qt_idx, (rank1, rank2, rankw) = item
        print("Collect %d, %d:" % (collect_idx, qt_idx))
        print("Rank by Anno %d, by Code %d, by both %d." % (rank1, rank2, rankw))
        print("Code: %s\nQT: %s\nAnnotation: %s" % (_ids2sent(codes[collect_idx * 50].split(), code_vocab_inv),
                                                    _ids2sent(qts[collect_idx * 50][qt_idx].split(), qt_vocab_inv),
                                                    _ids2sent(qbs[collect_idx * 50].split(), qb_vocab_inv)))
        if bool_print_winner_anno:
            print("Ranking winner: ")
            for rank_num in range(rank1):
                print("Rank %d annotation: %s" % (
                    rank_num, _ids2sent(qbs[collect_idx * 50 + pred[rank_num]].split(), qb_vocab_inv)))
        print("")

def collect_error_examples(sims_collection):
    example_indices = []

    for collect_idx, sims_list in enumerate(sims_collection):
        for idx in range(len(sims_list)):
            pred = get_rank(sims_list[idx])
            rank = pred.index(0)

            if rank != 0:
                example_indices.append((collect_idx, idx, (rank, -1, -1), pred))

    print("Size: %d" % len(example_indices))
    return example_indices


def organize_codenn_outputs(codenn_dir, num_examples, target_cids_concise=None):
    cid2results_by_run = {repeat_idx: defaultdict(list) for repeat_idx in range(1, 21)}

    print("Number of examples: ", num_examples) # 333 or 300

    for repeat_idx in range(1, 21):
        for example_idx in range(1, num_examples + 1):
            with open(codenn_dir + "repeat%d_ex%d_score.txt" % (repeat_idx, example_idx), "r") as f:
                lines = f.readlines()

            cid = int(lines[0].strip())

            cand_cids = []
            scores = []
            for line in lines[1:]:
                cand_cid, score = line.strip().split(",")
                cand_cids.append(int(cand_cid))
                scores.append(float(score))
            # scores = [float(line.strip().split(",")[1].strip()) for line in lines[1:]]
            assert len(scores) == 50

            cid2results_by_run[repeat_idx][cid].append((cand_cids, np.array(scores)))
    print("Size per run: %d" % len(cid2results_by_run[1])) # should be 111 or 100
    print("Size per cid: %d" % len(cid2results_by_run[1][16561])) # should be 3

    scores_collection = []
    if target_cids_concise is not None:
        print("\nMapping to the target...")
        target_num_examples_per_run = num_examples // 3 * 50
        print("target_num_examples_per_run: ", target_num_examples_per_run)  # 111 * 50 = 5550 or 100 * 50 = 5000
        print("Number of positivie examples per run: ", target_num_examples_per_run // 50)  # 111 or 100
        for repeat_idx in range(1, 21):
            cids_this_run = target_cids_concise[(repeat_idx - 1) * target_num_examples_per_run:
                                                repeat_idx * target_num_examples_per_run]
            for pos_idx in range(target_num_examples_per_run // 50):
                pos_cid = cids_this_run[pos_idx * 50]
                (codenn_cand_cids1, scores1), (codenn_cand_cids2, scores2), (codenn_cand_cids3, scores3) = \
                    cid2results_by_run[repeat_idx][pos_cid]
                assert codenn_cand_cids1 == codenn_cand_cids2 == codenn_cand_cids3 == \
                       cids_this_run[pos_idx * 50: pos_idx * 50 + 50]
                scores_collection.append([scores1, scores2, scores3])

    return cid2results_by_run, scores_collection


def organize_codenn_outputs_staqc(codenn_dir, num_examples):
    scores_collection = []

    for example_idx in range(1, num_examples + 1):
        filename = codenn_dir + "batch%d_score.txt" % example_idx
        with open(filename, 'r') as f:
            lines = f.readlines()

        scores = []
        pos_cid, pos_score = lines[1].split(",")
        for line in lines[2:]:
            cid, score = line.split(",")
            scores.append(float(score))

        assert len(scores) == 49
        scores.insert((example_idx-1) % 50, float(pos_score))
        scores_collection.append(np.array(scores))

    return scores_collection


def main():
    # get_toy_data("../data/", "sql")

    # # fix the val/test order issue
    # item = "code"
    # for data_name in ["val", "test"]:
    #     disordered_items = pickle.load(open("../data/slrnd_disordered/sql.%s.%s.slrnd.pkl" % (data_name, item)))
    #     ordered_old_items = pickle.load(open("../data/sql.%s.%s.pkl" % (data_name, item)))
    #     ordered_items = get_CR_eval_from_disordered(data_name, disordered_items, ordered_old_items)
    #     pickle.dump(ordered_items, open("../data/sql.%s.%s.slrnd.pkl" % (data_name, item), "wb"))


    # load_dir = "../data/"
    # new_data_alias = "qt_new_rl_mrr"
    # for item in ["qt", "qb", "code"]:
    #     data1 = pickle.load(open(load_dir + "sql.train.%s.pkl" % item))
    #     data2 = pickle.load(open(load_dir + "sql.train.%s.%s.pkl" % (item, new_data_alias)))
    #     data = data1 + data2
    #
    #     if item == "qb":
    #         data = [''] * len(data)
    #         print("** Sanity qb ***")
    #
    #     print("Size %d" % len(data))
    #     pickle.dump(data, open(load_dir + "sql.train.%s.%s_default.pkl" % (item, new_data_alias), "wb"))

    # # re-origanize codenn complete set
    # load_dir = "../data/codenn_combine_new/"
    # save_dir = "../data/"
    # for data_name in ["dev", "eval"]:
    #     qts = pickle.load(open(load_dir + "codenn_combine_new.sql.%s.qt.pkl" % data_name))
    #     qbs = pickle.load(open(load_dir + "codenn_combine_new.sql.%s.qb.qt_new_rl_mrr_qb.pkl" % data_name))
    #     codes = pickle.load(open(load_dir + "codenn_combine_new.sql.%s.code.pkl" % data_name))
    #     new_qts, new_qbs, new_codes = get_concise_codenn_complete_set(data_name, qts, qbs, codes)
    #     pdb.set_trace()
    #     pickle.dump(new_qts, open(save_dir + "codenn_combine_new.sql.%s.qt.pkl" % data_name, "wb"))
    #     pickle.dump(new_qbs, open(save_dir + "codenn_combine_new.sql.%s.qb.qt_new_rl_mrr_qb.pkl" % data_name, "wb"))
    #     pickle.dump(new_codes, open(save_dir + "codenn_combine_new.sql.%s.code.pkl" % data_name, "wb"))

    # # reranking from qb_only and code_only
    for data_name in ["val", "test"]:
        load_dir = "../data/with_qb_noCode_tp_qt_new_cleaned_rl_mrr_qb_valcodenn/bilstm/" \
                   "qtlen_20_qblen_20_codelen_120_qtnwords_4947_qbnwords_7775_codenwords_7726_batch_512_" \
                   "optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25_seqencdropout_25_simmeasure_cos_maxpool/"

        # # codenn_gen
        # load_dir = "../data/with_qb_noCode_tp_codenn_gen_valcodenn/bilstm/qtlen_20_qblen_20_codelen_120_" \
        #            "qtnwords_4947_qbnwords_827_codenwords_7726_batch_512_optimizer_adam_lr_001_embsize_200_lstmdims_400" \
        #            "_bowdropout_25_seqencdropout_25_simmeasure_cos_maxpool/"

        qb_only = pickle.load(open(load_dir + "collect_sims_%s.pkl" % data_name))

        # code_only = pickle.load(open("../data/no_qb_valcodenn/bilstm/"
        #                              "qtlen_20_qblen_20_codelen_120_qtnwords_4947_qbnwords_38008_codenwords_7726_"
        #                              "batch_128_optimizer_adam_lr_001_embsize_200_lstmdims_400_"
        #                              "bowdropout_35_seqencdropout_35_simmeasure_cos_maxpool/collect_sims_%s.pkl" % data_name))
        # codenn
        codenn_name = "dev" if data_name == "val" else "eval"
        code_only = pickle.load(open(
            "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/%s_scores_collection.pkl" % codenn_name))

        pure_size = 111 if data_name == "val" else 100
        for weight in [0.4]:#[x * 0.1 for x in range(11)]:
            mrrs, _, _, _ = weighting_scores(data_name, pure_size, weight, qb_only, code_only,
                             preprocess_collection1=None, preprocess_collection2="devide_max") #"devide_max"
            # example_indices_1help2, example_indices_2help1 = collect_weighting_examples(weight, qb_only, code_only)
            # pickle.dump((example_indices_1help2, example_indices_2help1), open(load_dir + "examples_qbhelp_codehelp.pkl", "wb"))
            # print("Size: %d, %d" % (len(example_indices_1help2), len(example_indices_2help1)))
        print("-"*50)

    # Error analysis
    # load_dir = "../data/with_qb_noCode_tp_qt_new_rl_mrr_qb_valcodenn/bilstm/" \
    #            "qtlen_14_qblen_20_codelen_118_qtnwords_4947_qbnwords_7805_codenwords_7726_batch_256_" \
    #            "optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25_seqencdropout_25_" \
    #            "simmeasure_cos_maxpool/"
    # load_dir = "../data/with_qb_noCode_tp_qt_new_rl_mrr_qb_valcodenn/bilstm/" \
    #            "qtlen_14_qblen_20_codelen_118_qtnwords_4947_qbnwords_7805_codenwords_7726_batch_256_" \
    #            "optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25_seqencdropout_25_" \
    #            "simmeasure_cos_weightedsum_attndot/"
    # qb_only = pickle.load(open(load_dir + "collect_sims_val.pkl"))
    # indices = collect_error_examples(qb_only)
    # pickle.dump(indices, open(load_dir + "collect_sims_val_error_indices.pkl", "wb"))

    # print examples
    # load_dir = "../data/with_qb_noCode_tp_qt_new_rl_mrr_qb_valcodenn/bilstm/" \
    #            "qtlen_14_qblen_20_codelen_118_qtnwords_4947_qbnwords_7805_codenwords_7726_batch_256_" \
    #            "optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25_seqencdropout_25_" \
    #            "simmeasure_cos_maxpool/"
    # error_examples = pickle.load(open(load_dir + "collect_sims_val_error_indices.pkl"))
    # qt_vocab = pickle.load(open("../data/sql.qt.vocab.pkl"))
    # qb_vocab = pickle.load(open("../data/sql.qb.vocab.qt_new_rl_mrr_qb.pkl"))
    # code_vocab = pickle.load(open("../data/sql.code.vocab.pkl"))
    # qts = pickle.load(open("../data/codenn_combine_new.sql.dev.qt.pkl"))
    # qbs = pickle.load(open("../data/codenn_combine_new.sql.dev.qb.qt_new_rl_mrr_qb.pkl"))
    # codes = pickle.load(open("../data/codenn_combine_new.sql.dev.code.pkl"))
    # show_examples(error_examples, qts, qbs, codes, qt_vocab, qb_vocab, code_vocab, bool_print_winner_anno=True)

    # # re-organize codenn outputs
    # model_dir = "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/"
    # data_name = "dev"
    # num_examples = 333 # 333 for DEV, 300 for EVAL
    # target_cids_concise = pickle.load(open(
    #     "../../../data/codenn_combine_new/codenn_combine_new.sql.%s.cids_concise.pkl" % data_name))
    # cid2results_by_run, scores_collection = organize_codenn_outputs(model_dir + "output_%s/" % data_name, num_examples,
    #                                                                 target_cids_concise=target_cids_concise)
    # pickle.dump(cid2results_by_run, open(
    #     model_dir + "%s_cid2results_by_run.pkl" % data_name, "wb"))
    # pickle.dump(scores_collection, open(
    #     model_dir + "%s_scores_collection.pkl" % data_name, "wb"))

    # # organize codenn output for staqc data
    # model_dir = "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/"
    # for data_name, num_examples in zip(["staqc_dev", "staqc_eval"], [11900, 17850]):
    #     scores_collection = organize_codenn_outputs_staqc(model_dir + "output_%s/" % data_name, num_examples)
    #     if data_name == "staqc_dev":
    #         data_name = "staqc_val"
    #     else:
    #         data_name = "staqc_test"
    #     pickle.dump(scores_collection, open(model_dir + "%s_scores_collection.pkl" % data_name, "wb"))

    # # StaQC val/test
    # for data_name in ["staqc_test"]:
    #     qb_only = pickle.load(open("../data/with_qb_noCode_tp_qt_new_cleaned_rl_mrr_qb_valcodenn/bilstm/"
    #                                "qtlen_20_qblen_20_codelen_120_qtnwords_4947_qbnwords_7775_codenwords_7726_"
    #                                "batch_512_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25_"
    #                                "seqencdropout_25_simmeasure_cos_maxpool/collect_sims_%s.pkl" % data_name))
    #     # qb_only = pickle.load(open("../data/with_qb_noCode_tp_codenn_gen_valcodenn/bilstm/"
    #     #                            "qtlen_20_qblen_20_codelen_120_qtnwords_4947_qbnwords_827_codenwords_7726"
    #     #                            "_batch_512_optimizer_adam_lr_001_embsize_200_lstmdims_400_bowdropout_25"
    #     #                            "_seqencdropout_25_simmeasure_cos_maxpool/collect_sims_%s.pkl" % data_name))
    #     # code_only = pickle.load(open("../data/no_qb_valcodenn/bilstm/qtlen_20_qblen_20_codelen_120_qtnwords_4947_"
    #     #                              "qbnwords_38008_codenwords_7726_batch_128_optimizer_adam_lr_001_embsize_200_"
    #     #                              "lstmdims_400_bowdropout_35_seqencdropout_35_simmeasure_cos_maxpool/"
    #     #                              "collect_sims_%s.pkl" % data_name))
    #     code_only = pickle.load(open(
    #         "/home/zyao/Projects2/codebases/codenn/src/model/saved_models_CSCR_sql_cleaned_dropout7/%s_scores_collection.pkl" % data_name))
    #
    #     for weight in [0.4]:#[x * 0.1 for x in range(11)]:
    #         mrrs, _, _, _ = weighting_scores_staqc(data_name, weight, qb_only, code_only,
    #                                preprocess_collection1=None, preprocess_collection2="devide_max") #"devide_max"
    #         print("Average mrr: %.4f, stdev: %.4f." % (np.average(mrrs), np.std(mrrs)))
    #
    #     print("-" * 50)


if __name__ == "__main__":
    main()