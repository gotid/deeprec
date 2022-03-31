from abc import ABC

import torch.nn.functional as F
from torch import ModuleDict

from deepctr.inputs import *
from deepctr.layers import DNN
from .basemodel import BaseModel


class YouTubeDNN(BaseModel):
    def __init__(self,
                 user_features,
                 item_features,
                 num_sampled=5,
                 user_dnn_hidden_units=None,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 init_std=0.002,
                 l2_reg_emb=1e-6,
                 l2_reg_linear=1e-5,
                 dnn_dropout=0,
                 seed=1024,
                 device='cpu',
                 **kwargs):
        super(YouTubeDNN, self).__init__(user_features,
                                         item_features,
                                         l2_reg_emb=l2_reg_emb,
                                         l2_reg_linear=l2_reg_linear,
                                         init_std=0.0001,
                                         seed=seed,
                                         task='binary',
                                         device='cpu',
                                         shuffle=True,
                                         num_workers=8,
                                         log_every_n_steps=2,
                                         **kwargs)
        if user_dnn_hidden_units is None:
            user_dnn_hidden_units = [64, 32]
        self.num_sampled = num_sampled

        self.user_dnn = DNN(self.ndim(user_features),
                            user_dnn_hidden_units,
                            activation=dnn_activation,
                            dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn,
                            init_std=init_std,
                            seed=seed,
                            device=device)

    def forward(self, X: torch.Tensor):
        """X：一人一行，包括人和物的输入数据"""
        batch_size = X.size(0)
        user_emb = self.user_tower(X)
        item_emb = self.item_tower(X)

        if self.mode == 'user_emb':
            return user_emb
        if self.mode == 'item_emb':
            return item_emb

        score = F.cosine_similarity(user_emb, item_emb, dim=-1)
        score = score.view(batch_size, -1)
        return score

    def user_tower(self, X):
        if self.mode == 'item_emb':
            return None

        user_sparse_emb_list, user_dense_value_list = self.input_from_features(X,
                                                                               self.user_features,
                                                                               self.dnn_emb_dict)
        user_dnn_input = combined_dnn_input(user_sparse_emb_list, user_dense_value_list)
        user_emb = self.user_dnn(user_dnn_input)  # (batch_size, embedding_dim)
        user_emb = user_emb.unsqueeze(1)  # (batch, 1, embedding_dim)
        return user_emb

    def item_tower(self, X):
        if self.mode == 'user_emb':
            return None

        item_emb_list, _ = self.input_from_item_features(X, self.item_features, self.dnn_emb_dict)
        item_emb = item_emb_list[0]  # (batch, item_list_len, feat_dim)
        return item_emb

    def input_from_item_features(self, X, item_features: FeatList, dnn_emb_dict: ModuleDict, support_dense=True):
        """
            获取给定张量 X 的特征值。

            :param X: 输入张量 TODO 形状是？
            :param item_features: 特征列表
            :param dnn_emb_dict: DNN 特征层嵌入模块字典
            :param support_dense: 是否必须包含稠密特征
            :returns (sparse_emb_list, dense_value_list)

                - 稀疏嵌入值列表（含序列嵌入值） sparse_emb_list
                - 稠密值列表 dense_value_list
            """
        # 提取三类特征列表
        sparse_features = extract_features(item_features, SparseFeat)
        seq_features = extract_features(item_features, SeqFeat)
        dense_features = extract_features(item_features, DenseFeat)

        # 验证稠密特征是否支持
        if not support_dense and len(dense_features) > 0:
            raise ValueError('dnn_features 中不支持 DenseFeat')

        emb_dict = dnn_emb_dict
        idx_dict = self.feature_idx_dict

        # 生成稀疏特征嵌入值
        sparse_emb_list = [emb_dict[feat.emb_name](X[:, idx_dict[feat.name][0]:idx_dict[feat.name][1]].long())
                           for feat in sparse_features]

        # 生成序列稀疏特征嵌入值
        seq_emb_dict = get_seq_emb_dict(X, seq_features, emb_dict, idx_dict)
        feat_name = seq_features[0].name
        item_emb = seq_emb_dict[feat_name]  # (batch_size, item_id_len, feat_dim)
        seq_emb_list = [item_emb]

        # 累加稀疏特征+序列特征嵌入值
        sparse_emb_list += seq_emb_list

        # 提取稠密值列表
        dense_value_list = [X[: idx_dict[feat.name][0]:idx_dict[feat.name][1]] for feat in dense_features]

        return sparse_emb_list, dense_value_list
