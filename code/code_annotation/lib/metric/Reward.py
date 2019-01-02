import lib

# loading codenn truths
codenn_refs = []
for data_name in ["DEV", "EVAL"]:
    with open("from_codenn/%s_ref.txt" % data_name, 'r') as f:
        golds = []
        for line in f.readlines():
            rid, string = line.split('\t')
            golds.append((int(rid), string.strip()))
        codenn_refs.append(golds)

def clean_up_sentence(sent, remove_unk=False, remove_eos=False):
    if lib.Constants.EOS in sent:
        sent = sent[:sent.index(lib.Constants.EOS) + 1]
    if remove_unk:
        sent = filter(lambda x: x != lib.Constants.UNK, sent)
    if remove_eos:
        if len(sent) > 0 and sent[-1] == lib.Constants.EOS:
            sent = sent[:-1]
    return sent

def clean_up_sentence_string(sent, dict):
    sent = clean_up_sentence(sent, remove_unk=False, remove_eos=True)
    sent_string = " ".join([dict.getLabel(w) for w in sent])
    return sent, sent_string

def single_sentence_bleu(pair, tgt_dict):
    length = len(pair[0])
    pred, gold = pair

    pred, pred_string = clean_up_sentence_string(pred, tgt_dict)
    gold, gold_string = clean_up_sentence_string(gold, tgt_dict)

    len_pred = len(pred)
    if len_pred == 0:
        score = 0.
        pred = [lib.Constants.EOS] + [lib.Constants.PAD] * (length - 1)
    else:
        # score = lib.Bleu.score_sentence(pred, gold, 4, smooth=1)[-1]

        # codenn_bleu
        (goldMap, predictionMap) = computeMapsFromPairList([(0, pred_string)], [(0, gold_string)])
        score = bleuFromMaps(goldMap, predictionMap)[0]

        pred.append(lib.Constants.EOS)
        while len(pred) < length:
            pred.append(lib.Constants.PAD)

    return score, pred

def sentence_bleu(preds, golds, tgt_dict):
    results = map(single_sentence_bleu, zip(preds, golds), [tgt_dict] * len(preds))
    scores, preds = zip(*results)
    return scores, preds

def sentence_bleu_codenn(preds, indices, data_name, tgt_dict):
    # clean up preds
    cleaned_preds, cleaned_pred_strings = [], []
    for pred in preds:
        cleaned_pred, cleaned_pred_string = clean_up_sentence_string(pred, tgt_dict)
        cleaned_preds.append(cleaned_pred)
        cleaned_pred_strings.append(cleaned_pred_string)

    (goldMap, predictionMap) = computeMapsFromPairList(
        [(idx[1], pred_string) for idx, pred_string in zip(indices, cleaned_pred_strings)],
        codenn_refs[0] if data_name == "DEV" else codenn_refs[1])
    scores = bleuListFromMaps(goldMap, predictionMap)
    return scores, cleaned_preds

def warpped_sentence_bleu(preds, targets, tgt_dict, data_name=None, indices=None, **kwargs):
    if data_name is not None and data_name in {"DEV", "EVAL"}:
        return sentence_bleu_codenn(preds, indices, data_name, tgt_dict)
    else:
        return sentence_bleu(preds, targets, tgt_dict)