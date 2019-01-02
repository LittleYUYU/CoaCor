import numpy as np
import lib

# import code_retrieval
# cr = code_retrieval.CrCritic()

# a few configs to set up in a2c-train.py
cr = None
reward_mode = None # {cr, cr_diff, cr_noqb}
cal_mode_train = "sentence" # sample a pool or use the batch as a pool
cal_mode_eval = "batch"
replace_all_train = False # whether to replace Anno for neg examples
replace_all_eval = False
qt_candidates_train = False # whether to consider QTs as candidates
qt_candidates_eval = False


def empty_check(ids):
    return len(ids) == 0


def clean_up_ids(noisy_ids):
    if lib.Constants.EOS in noisy_ids:
        noisy_ids = noisy_ids[:noisy_ids.index(lib.Constants.EOS)]

    # if empty_check(noisy_ids):
    #     return None
    # if noisy_ids[0] == lib.Constants.BOS:
    #     noisy_ids = noisy_ids[1:]

    if empty_check(noisy_ids):
        return None
    # while noisy_ids[-1] == lib.Constants.PAD:
    #     noisy_ids = noisy_ids[:-1]
    #     if empty_check(noisy_ids):
    #         return None

    return np.array(noisy_ids)


def clean_up_ids_list(list_of_noisy_ids):
    fed_ids_list, cleaned_ids_list = [], []
    for ids in list_of_noisy_ids:
        length = len(ids)
        clean_ids = clean_up_ids(ids)
        if clean_ids is None:
            fed_ids_list.append(np.array([lib.Constants.PAD] * length))
            cleaned_ids_list.append([lib.Constants.EOS] + [lib.Constants.PAD] * (length - 1))
        else:
            fed_ids_list.append(clean_ids)
            clean_ids = list(clean_ids)
            if len(clean_ids) < length:
                clean_ids += [lib.Constants.EOS] + [lib.Constants.PAD] * (length - len(clean_ids) - 1)
            cleaned_ids_list.append(clean_ids)
    return fed_ids_list, cleaned_ids_list


def sentence_retrieval_mrr(data_name, code, annotation, qt, qb,
                           number_of_runs=1, bool_empty_qb=False):
    length = len(annotation)
    code, annotation, qt, qb = clean_up_ids(code), clean_up_ids(annotation), clean_up_ids(qt), clean_up_ids(qb)
    if annotation is None:
        annotation = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(annotation)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + [lib.Constants.PAD] * (length - len(annotation) - 1)

    mrr_list = cr.get_reward(data_name, code, annotation, qt, qb, idx=None,
                             reward_mode=reward_mode, number_of_runs=number_of_runs, bool_processed=True,
                             bool_empty_qb=bool_empty_qb)

    mrr = mrr_list[0] if reward_mode in ["cr", "cr_noqb"] else (mrr_list[0] - mrr_list[1])

    return mrr, cleaned_annotation


def batch_retrieval_mrr(codes, annotations, qts, qbs,
                        bool_empty_qb=False, replace_all=False, qt_candidates=False):
    fed_codes, _ = clean_up_ids_list(codes)
    fed_annotations, cleaned_annotations = clean_up_ids_list(annotations)
    fed_qts, _ = clean_up_ids_list(qts)
    fed_qbs, _ = clean_up_ids_list(qbs)

    if qt_candidates:
        rrs_list = cr.get_reward_batch_eval_qt(fed_codes, fed_annotations, fed_qts, fed_qbs,
                                               reward_mode=reward_mode)
    else:
        rrs_list = cr.get_reward_batch_eval(fed_codes, fed_annotations, fed_qts, fed_qbs,
                                            reward_mode=reward_mode, bool_empty_qb=bool_empty_qb,
                                            replace_all=replace_all)

    if reward_mode in ["cr", "cr_noqb"]:
        mrrs = rrs_list[0]
    else:
        mrrs = list(np.array(rrs_list[0]) - np.array(rrs_list[1]))

    return mrrs, cleaned_annotations


def retrieval_mrr_train(annotations, qbs, codes, qts, bool_empty_qb=None, **kwargs):
    if cal_mode_train == "sentence":
        assert not replace_all_train
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt, qb in zip(codes, annotations, qts, qbs):
            mrr, cleaned_annotation = sentence_retrieval_mrr("train", code, annotation, qt, qb,
                                                             number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    elif cal_mode_train == "batch":
        mrrs, cleaned_annotations = batch_retrieval_mrr(codes, annotations, qts, qbs,
                                                        replace_all=replace_all_train,
                                                        qt_candidates=qt_candidates_train)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_mrr_eval(annotations, qbs, codes, qts, bool_empty_qb, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_mrr(codes, annotations, qts, qbs,
                                                    bool_empty_qb=bool_empty_qb,
                                                    replace_all=replace_all_eval,
                                                    qt_candidates=qt_candidates_eval)
    return mrrs, cleaned_annotations






