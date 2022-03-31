import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from deepctr.inputs import SparseFeat, SeqFeat
from deepmatch.models import YouTubeDNN
from examples.preprocess import gen_data_set_youteube, gen_model_input

if __name__ == '__main__':
    data = pd.read_csv('./movielens_sample.txt')
    sparse_features = ['movie_id', 'user_id', 'gender', 'age', 'occupation', 'zip']
    SEQ_LEN = 50

    # 对稀疏特征进行标签编码
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feat in features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) + 1
        feature_max_idx[feat] = data[feat].max() + 1

    # 构建用户画像
    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates('user_id')
    user_profile.set_index('user_id', inplace=True)
    user_item_list = data.groupby('user_id')['movie_id'].apply(list)

    # 构建物品画像
    item_profile = data[['movie_id']].drop_duplicates('movie_id')

    # 生成数据集
    train_set, test_set = gen_data_set_youteube(data, 5)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 计算每个稀疏字段的唯一特征，并为序列特征生成特征排至
    emb_dim = 16

    user_features = [SparseFeat('user_id', feature_max_idx['user_id'], emb_dim),
                     SparseFeat('gender', feature_max_idx['gender'], emb_dim),
                     SparseFeat('age', feature_max_idx['age'], emb_dim),
                     SparseFeat('occupation', feature_max_idx['occupation'], emb_dim),
                     SparseFeat('zip', feature_max_idx['zip'], emb_dim),
                     SeqFeat(SparseFeat('hist_movie_id',
                                        vocab_size=feature_max_idx['movie_id'],
                                        emb_dim=emb_dim,
                                        emb_name='movie_id'),
                             maxlen=10, combiner='mean')]

    item_features = [SeqFeat(SparseFeat('movie_id', feature_max_idx['movie_id'], emb_dim, emb_name='movie_id'),
                             maxlen=6, combiner='mean')]

    # 定义模型并训练
    model = YouTubeDNN(user_features,
                       item_features,
                       num_sampled=5,
                       user_dnn_hidden_units=(64, emb_dim),
                       criterion=F.cross_entropy,
                       optimizer='Adam',
                       config={})

    # print(model)
    model.fit(train_model_input, train_label, max_epochs=10, batch_size=128)

    # 生成测试用户特征和所有物品特征
    model.mode = 'user_emb'
    user_emb = model.full_predict(test_model_input, batch_size=64)
    user_emb = user_emb.reshape((user_emb.shape[0], user_emb.shape[-1]))
    user_emb = user_emb.numpy()
    print(user_emb)

    model.mode = "item_emb"
    all_item_model_input = {"movie_id": item_profile['movie_id'].values}
    item_emb_model = model.rebuild_feature_index(item_features)
    item_emb = item_emb_model.full_predict(all_item_model_input, batch_size=2 ** 12)
    item_emb = item_emb.reshape((item_emb.shape[0], item_emb.shape[-1]))
    item_emb = item_emb.numpy()
    print(item_emb)

    # 搜索及评估
    test_true_label = {line[0]: line[2] for line in test_set}
    print(test_true_label)
    # print(item_emb)

    # import numpy as np
    # import faiss
    # from tqdm import tqdm
    # from deepmatch.utils import recall_N
    #
    # index = faiss.IndexFlatL2(embedding_dim)
    # # faiss.normalize_L2(item_emb)
    # index.add(item_emb)
    # print(index.ntotal)
    # faiss.normalize_L2(user_embs)
    # D, I = index.search(np.ascontiguousarray(user_embs), 50)
    # s = []
    # hit = 0
    # for i, uid in tqdm(enumerate(test_model_input['user_id'])):
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
    # print("hr", hit / len(test_model_input['user_id']))
