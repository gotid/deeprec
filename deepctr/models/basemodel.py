from typing import List

import torch
from torch import Tensor, nn

from deepctr.inputs import SparseFeat, SequenceFeat, DenseFeat, Features


class Linear(torch.nn.Module):
    def __init__(self, feature_columns: List[SparseFeat, SequenceFeat, DenseFeat],
                 init_std=0.0001, device='cpu'):
        """
        特征列线性层。

        :param feature_columns: 特征列列表
        :param init_std: 初始化标准差
        :param device: 设备
       """
        super(Linear, self).__init__()
        self.device = device
        self.weight = None

        # 初始化特征管理器
        self.features = Features(columns=feature_columns, linear=True, init_std=init_std, device=device)

        # 存在稠密特征的线性层，需初始化权重
        if len(self.features.dense_columns) > 0:
            self.weight = nn.Parameter(
                Tensor(sum(fc.embedding_dim for fc in self.features.dense_columns), 1).to(device))
            nn.init.normal_(self.weight, std=init_std)

    def forward(self, inputs: Tensor, sparse_feat_refine_weight=None):
        sparse_emb_list, dense_value_list = self.features.transform(inputs)

        # 初始化线性回归值
        linear_logit = torch.zeros([inputs.shape[0], 1]).to(sparse_emb_list[0].device)

        # 累加稀疏嵌入值
        if len(sparse_emb_list) > 0:
            sparse_emb_cat = torch.cat(sparse_emb_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i} = m_{x,i} * w_i (in IFM and DIFM)
                sparse_emb_cat *= sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_emb_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit

        # 累加稠密值
        if len(dense_value_list) > 0:
            dense_value_cat = torch.cat(dense_value_list, dim=-1)
            dense_value_logit = dense_value_cat.matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit


class BaseModel(torch.nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, seed=1024, init_std=0.0001, device='cpu'):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.feature_index = build_feature_index(linear_feature_columns + dnn_feature_columns)
        self.embedding_dict = create_embedding_dict(dnn_feature_columns, init_std=init_std, device=device)
        self.linear_model = Linear(linear_feature_columns, self.feature_index, device=device)
