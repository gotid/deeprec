import random

import numpy as np
from tqdm import tqdm


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """
    填充序列到等长的 ndarray 数组。
    这是 tf.keras.preprocessing.sequence.pad_sequences 的 Pytorch 等效实现。

    :param sequences: 序列
    :param maxlen: 保留的最大长度
    :param dtype: 数据类型
    :param padding: 填充方位
    :param truncating: 截短方位
    :param value: 填充值或截短值
    :return: 填充或截短后的 ndarray
    """
    assert padding in ['pre', 'post'], f'无效填充方位={padding}，仅支持 pre|post'
    assert truncating in ['pre', 'post'], f'无效截短方位={truncating}，仅支持 pre|post'

    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # 空列表

        if truncating == 'pre':  # 截前
            trunc = x[-maxlen:]
        else:
            trunc = x[:maxlen]  # 截后
        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'pre':  # 填前
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc  # 填后
    return arr


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set,


def gen_data_set_youteube(data, negsample=5):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # 这里的 label = 1 其实相当于是多分类的 1
                train_set.append((reviewerID, hist[::-1], [pos_list[i]] + [neg_list[item_idx] for item_idx in
                                                                           np.random.choice(neg_list, negsample)], 0,
                                  len(hist[::-1]), rating_list[i]))
            else:
                test_set.append((reviewerID, hist[::-1], [pos_list[i]] + [neg_list[item_idx] for item_idx in
                                                                          np.random.choice(neg_list, negsample)], 0,
                                 len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set


def gen_model_input(train_set, user_profile, seq_maxlen):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_maxlen, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label
