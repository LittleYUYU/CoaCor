import numpy as np
import lib

import code_retrieval
cr = code_retrieval.CrCritic()


def empty_check(ids):
    return len(ids) == 0


def clean_up_ids(noisy_ids):
    if lib.Constants.EOS in noisy_ids:
        noisy_ids = noisy_ids[:noisy_ids.index(lib.Constants.EOS)]

    if empty_check(noisy_ids):
        return None
    if noisy_ids[0] == lib.Constants.BOS:
        noisy_ids = noisy_ids[1:]

    if empty_check(noisy_ids):
        return None
    while noisy_ids[-1] == lib.Constants.PAD:
        noisy_ids = noisy_ids[:-1]
        if empty_check(noisy_ids):
            return None

    return np.array(noisy_ids)


def sentence_retrieval_mrr(data_name, code, annotation, qt, qb, number_of_runs):
    length = len(annotation)
    code, annotation, qt, qb = clean_up_ids(code), clean_up_ids(annotation), clean_up_ids(qt), clean_up_ids(qb)
    if annotation is None:
        annotation = [lib.Constants.PAD] * length
    else:
        annotation = list(annotation)
        annotation += [lib.Constants.PAD] * (length - len(annotation))
    mrr = cr.get_reward(data_name, code, annotation, qt, qb,
                        number_of_runs=number_of_runs, bool_processed=True)

    return mrr, annotation


def retrieval_mrr_valid(codes, annotations, qts, qbs):
    # cleaned_codes = [clean_up_ids(code) for code in codes]
    # cleaned_annotations = [clean_up_ids(annotation) for annotation in annotations]
    # cleaned_qts = [clean_up_ids(qt) for qt in qts]
    # cleaned_qbs = [clean_up_ids(qb) for qb in qbs]
    #
    # mrrs = cr.get_reward_in_batch(cleaned_codes, cleaned_annotations, cleaned_qts, cleaned_qbs,
    #                              number_of_runs=1)

    cleaned_annotations = []
    mrrs = []

    for code, annotation, qt, qb in zip(codes, annotations, qts, qbs):
        mrr, cleaned_annotation = sentence_retrieval_mrr("valid", code, annotation, qt, qb, 1)
        mrrs.append(mrr)
        cleaned_annotations.append(cleaned_annotation)

    return mrrs, cleaned_annotations


def retrieval_mrr_test(codes, annotations, qts, qbs):
    cleaned_annotations = []
    mrrs = []

    for code, annotation, qt, qb in zip(codes, annotations, qts, qbs):
        mrr, cleaned_annotation = sentence_retrieval_mrr("test", code, annotation, qt, qb, 1)
        mrrs.append(mrr)
        cleaned_annotations.append(cleaned_annotation)

    return mrrs, cleaned_annotations

