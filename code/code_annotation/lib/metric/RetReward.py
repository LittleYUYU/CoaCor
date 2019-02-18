import numpy as np
import lib
import pdb

# import code_retrieval
# cr = code_retrieval.CrCritic()

# a few configs to set up in a2c-train.py
cr = None
cal_mode_train = "sentence" # sample a pool or use the batch as a pool
cal_mode_eval = "batch"
reward_lambda = 1.0
reward_gamma = 1.0


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


def sentence_retrieval_mrr(data_name, code, annotation, qt, number_of_runs=1):
    length = len(annotation)
    code, annotation, qt = clean_up_ids(code), clean_up_ids(annotation), clean_up_ids(qt)
    if annotation is None:
        annotation = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(annotation)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + [lib.Constants.PAD] * (length - len(annotation) - 1)

    mrr = cr.get_reward(data_name, code, annotation, qt,
                             number_of_runs=number_of_runs, bool_processed=True)

    return mrr, cleaned_annotation


def batch_retrieval_mrr(codes, annotations, qts):
    fed_codes, _ = clean_up_ids_list(codes)
    fed_annotations, cleaned_annotations = clean_up_ids_list(annotations)
    fed_qts, _ = clean_up_ids_list(qts)

    mrrs = cr.get_reward_batch_eval(fed_codes, fed_annotations, fed_qts)

    return mrrs, cleaned_annotations


def retrieval_mrr_train(annotations, qts, codes, **kwargs):
    if cal_mode_train == "sentence":
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt in zip(codes, annotations, qts):
            mrr, cleaned_annotation = sentence_retrieval_mrr("train", code, annotation, qt,
                                                             number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    elif cal_mode_train == "batch":
        mrrs, cleaned_annotations = batch_retrieval_mrr(codes, annotations, qts)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_mrr_eval(annotations, qts, codes, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_mrr(codes, annotations, qts)

    return mrrs, cleaned_annotations


def mixed_mrr_bleu_train(annotations, qts, codes, tgt_dict, data_name=None, indices=None):
    mrrs, cleaned_annotations = retrieval_mrr_train(annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu*(1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)] 
    
    return rewards, cleaned_annotations


def mixed_mrr_bleu_eval(annotations, qts, codes, tgt_dict, data_name=None, indices=None):
    mrrs, cleaned_annotations = retrieval_mrr_eval(annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu*(1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations

