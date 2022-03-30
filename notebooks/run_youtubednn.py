import sys

sys.path.append('../')

import pandas as pd
from deepmatch.models import YouTubeDNN
from deepctr.inputs import SparseFeat, SequenceFeat
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

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


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


if __name__ == '__main__':
    data = pd.read_csv('./movielens_sample.txt')
    sparse_features = ['movie_id', 'user_id', 'gender', 'age', 'occupation', 'zip']
    SEQ_LEN = 50

    data.head()

    # 对稀疏特征进行标签编码
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feat in features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) + 1
        feature_max_idx[feat] = data[feat].max() + 1

    # 构建用户画像
    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')

    # 构建物品画像
    item_profile = data[['movie_id']].drop_duplicates('movie_id')

    user_profile.set_index('user_id', inplace=True)

    user_item_list = data.groupby('user_id')['movie_id'].apply(list)

    train_set, test_set = gen_data_set_youteube(data, 5)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)

    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 计算每个稀疏字段的唯一特征，并为序列特征生成特征排至
    embedding_dim = 16

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat('gender', feature_max_idx['gender'], embedding_dim),
                            SparseFeat('age', feature_max_idx['age'], embedding_dim),
                            SparseFeat('occupation', feature_max_idx['occupation'], embedding_dim),
                            SparseFeat('zip', feature_max_idx['zip'], embedding_dim),
                            SequenceFeat(SparseFeat('hist_movie_id',
                                                    vocabulary_size=feature_max_idx['movie_id'],
                                                    embedding_dim=embedding_dim,
                                                    embedding_name='movie_id'), maxlen=10, combiner='mean')]

    item_feature_columns = [
        SequenceFeat(SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim, embedding_name='movie_id'),
                     maxlen=6, combiner='mean')]

    # 定义模型并训练
    model = YouTubeDNN(user_feature_columns,
                       item_feature_columns,
                       num_sampled=5,
                       user_dnn_hidden_units=(64, embedding_dim),
                       criterion=F.cross_entropy,
                       optimizer='Adam',
                       config={})

    # print(model)
    model.fit(train_model_input, train_label, max_epochs=10, batch_size=128)

    # 生成测试用户特征和所有物品特征
    test_user_model_input = test_model_input
    model.mode = 'user_representation'
    user_embedding_model = model

    user_embs = user_embedding_model.full_predict(test_user_model_input, batch_size=2)
    user_embs = user_embs.reshape((user_embs.shape[0], user_embs.shape[-1]))
    user_embs = user_embs.numpy()
    print(user_embs)

    model.mode = "item_representation"
    all_item_model_input = {"movie_id": item_profile['movie_id'].values}
    item_embedding_model = model.rebuild_feature_index(item_feature_columns)
    item_embs = item_embedding_model.full_predict(all_item_model_input, batch_size=2 ** 12)
    item_embs = item_embs.reshape((item_embs.shape[0], item_embs.shape[-1]))
    item_embs = item_embs.numpy()
    print(item_embs)

    # 搜索及评估
    test_true_label = {line[0]: line[2] for line in test_set}
    print(test_true_label)
    # print(item_embs)

    # import numpy as np
    # import faiss
    # from tqdm import tqdm
    # from deepmatch.utils import recall_N
    #
    # index = faiss.IndexFlatL2(embedding_dim)
    # # faiss.normalize_L2(item_embs)
    # index.add(item_embs)
    # print(index.ntotal)
    # faiss.normalize_L2(user_embs)
    # D, I = index.search(np.ascontiguousarray(user_embs), 50)
    # s = []
    # hit = 0
    # for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    #     try:
    #         pred = [item_profile['movie_id'].values[x] for x in I[i]]
    #         filter_item = None
    #         uid = int(uid)
    #         recall_score = recall_N(test_true_label[uid], pred, N=50)
    #         print(uid, "回调得分", recall_score)
    #         s.append(recall_score)
    #         if test_true_label[uid] in pred:
    #             hit += 1
    #     except Exception as e:
    #         print(i, e)
    # print("recall", np.mean(s))
    # print("hr", hit / len(test_user_model_input['user_id']))
