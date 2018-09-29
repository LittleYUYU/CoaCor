# others.py
# Very trivial scripts for processing trivial things...

import pickle
import random
import torch
# import lib


def get_indices(lang, data_name, load_dir):
    assert data_name in {"train", "valid", "test"}

    code = pickle.load(open(load_dir + "%s.%s.code.pkl" % (lang, data_name)))
    qb = pickle.load(open(load_dir + "%s.%s.qb.pkl" % (lang, data_name)))
    qt = pickle.load(open(load_dir + "%s.%s.qt.pkl" % (lang, data_name)))

    code_indices = [item[0] for item in code]
    qb_indices = [item[0] for item in qb]
    qt_indices = [item[0] for item in qt]

    tuple_indices = zip(qt_indices, qb_indices, code_indices) # important order!!!
    print("%s %s: size %d" % (lang, data_name, len(tuple_indices)))

    return tuple_indices


def find_duplicate_indices(name1, indices1, name2, indices2):
    print("-" * 10)
    overlap = set(indices2).intersection(set(indices1))
    print("#of duplicate examples, %s vs %s: %d" % (name1, name2, len(overlap)))
    if len(overlap):
        print overlap
    return overlap


def find_same_code_indices(name1, indices1, name2, indices2):
    code_set = set([item[2] for item in indices1])
    duplicate_tuples = []
    for idx_tuple in indices2:
        if idx_tuple[2] in code_set:
            duplicate_tuples.append(idx_tuple)
    print("#of same_code examples, %s vs %s: %d" % (name1, name2, len(duplicate_tuples)))
    if duplicate_tuples:
        print(random.sample(duplicate_tuples, min(10, len(duplicate_tuples))))
    return duplicate_tuples


def find_all_same_code_indices(train, valid, test):
    print("-" * 10)
    train_and_valid_tuples = find_same_code_indices("train", train, "valid", valid)
    train_valid_and_test_tuples = find_same_code_indices("train_valid", train_and_valid_tuples, "test", test)

    train_and_test_tuples = find_same_code_indices("train", train, "test", test)
    train_test_and_valid_tuples = find_same_code_indices("train_test", train_and_test_tuples, "valid", valid)

    to_remove_tuples = set(train_and_valid_tuples + train_valid_and_test_tuples +
                           train_and_test_tuples + train_test_and_valid_tuples)
    print("Total to remove: %d" % len(to_remove_tuples))
    return to_remove_tuples


def clean_split_indices(indices_list):
    train, valid, test = indices_list

    # remove duplicates
    overlap_train_valid = find_duplicate_indices("train", train, "valid", valid)
    for item in overlap_train_valid:
        valid.remove(item)

    overlap_train_test = find_duplicate_indices("train", train, "test", test)
    for item in overlap_train_test:
        test.remove(item)

    # move examples with the same code idx to train set
    # index order: qt, qb, code
    to_remove_tuples = find_all_same_code_indices(train, valid, test)
    for item in to_remove_tuples:
        if item in valid:
            valid.remove(item)
        if item in test:
            test.remove(item)
        train.append(item)

    return indices_list


def get_split_indices(lang, load_dir):
    data_name_list = ["train", "valid", "test"]
    indices_list = [get_indices(lang, data_name, load_dir) for data_name in data_name_list]

    # indices_list = clean_split_indices(indices_list)
    data_name2indices = {}

    print("After cleaning:")
    for idx in range(3):
        print("%s %s: size %d" % (lang, data_name_list[idx], len(indices_list[idx])))
        data_name2indices[data_name_list[idx]] = indices_list[idx]

    return data_name2indices


def get_pos_negs_pairs(split_indices, data_name):
    print("Working on %s set..." % data_name)
    indices = split_indices[data_name]
    pool_size = 50

    pos_negs_pairs = []

    for pos_idx in range(len(indices) // pool_size):
        pos = indices[pos_idx * pool_size]
        negs = indices[pos_idx * pool_size + 1: (pos_idx + 1) * pool_size]
        assert len(negs) == pool_size - 1
        pos_negs_pairs.append((pos, negs))

    print("#of positive examples: %d" % len(pos_negs_pairs))
    return pos_negs_pairs


def prepare_eval_set(data_name, pos_negs_pairs, available_indices):
    pos_indices = [pos for pos, negs in pos_negs_pairs if pos in available_indices]
    found = len(pos_indices)

    remain_indices = set(available_indices) - set(pos_indices)
    new_indices = random.sample(remain_indices, len(pos_negs_pairs) - found)
    pos_indices += list(new_indices)

    print("%s set: #of positive examples %d, re-sampled %d" % (data_name, len(pos_indices), len(new_indices)))
    return pos_indices


#############################
## for processing CODENN data
#############################
def prepare_codenn_eval(data_name, qid2cid, qid2qt, cid2code, EOS):
    print("Processing %s set" % data_name)
    src, tgt = [], []
    ignored = 0

    for qid, cid in qid2cid:
        qt_str = qid2qt[qid]
        code_str = cid2code[cid]

        qt = [int(i) for i in qt_str.strip().split(" ")] + [EOS]
        code = [int(i) for i in code_str.strip().split(" ")]

        if len(qt) == 0 or len(code) == 0:
            ignored += 1
            continue

        src.append(torch.LongTensor(code))
        tgt.append(torch.LongTensor(qt))

    print("Size %d. #ignored %d." % (len(src), ignored))
    return src, tgt


def main():
    lang = "sql"
    load_dir = "../../data/version2/source/"
    save_dir = "../../data/version2/origin/"
    # split_indices = get_split_indices(lang, load_dir)
    split_indices = pickle.load(open(load_dir + "split_indices_%s.pkl" % lang))

    final_split_indices = {"train": split_indices["train"]}
    for data_name in ["valid", "test"]:
        pos_negs_pairs = pickle.load(open(load_dir + "pos_negs_pairs_%s_%s.pkl" % (data_name, lang)))
        pos_indices = prepare_eval_set(data_name, pos_negs_pairs, split_indices[data_name])
        final_split_indices[data_name] = pos_indices

    pickle.dump(final_split_indices, open(save_dir + "split_indices_simplified_%s.pkl" % lang, "wb"))

    # # preparing codenn data for evaluating CA model
    # load_dir = "../../data/codenn/"
    # data = {"valid": {}, "test": {}}
    #
    # for data_name in ["valid", "test"]:
    #     file_name = "dev" if data_name == "valid" else "eval"
    #     cid2code = pickle.load(open(load_dir + "codenn.%s.ix_to_code.processed.pkl" % file_name))
    #     qid2qt = pickle.load(open(load_dir + "codenn.%s.ix_to_qt.processed.pkl" % file_name))
    #     qid2cid = pickle.load(open(load_dir + "codenn.%s.qid_to_cid.dataset.pkl" % file_name))
    #     src, tgt = prepare_codenn_eval(data_name, qid2cid, qid2qt, cid2code)
    #     data[data_name]["src"] = src
    #     data[data_name]["tgt"] = tgt
    #     data[data_name]["qt"] = [None] * len(src)
    #     data[data_name]["tree"] = []
    #
    # dicts = torch.load("dataset/train_qt/sql.processed_all.train.pt")["dicts"]
    # data["dicts"] = dicts
    # data["train"] = {"src": [], "tgt": [], "qt": [], "tree": []}
    # torch.save(data, "dataset/sql.processed_all.codenn_test.pt")


if __name__ == '__main__':
    main()