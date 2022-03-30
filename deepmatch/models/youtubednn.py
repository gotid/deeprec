import torch
import torch.nn.functional as F

from deepctr.inputs import combined_dnn_input
from deepctr.layers import DNN
from .plbasemodel import PLBaseModel


class YouTubeDNN(PLBaseModel):
    def __init__(self,
                 user_feature_columns,
                 item_feature_columns,
                 num_sampled=5,
                 user_dnn_hidden_units=[64, 32],
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 init_std=0.002,
                 l2_reg_dnn=0,
                 l2_reg_embedding=1e-6,
                 dnn_dropout=0,
                 seed=1024,
                 device='cpu',
                 **kwargs):
        super(YouTubeDNN, self).__init__(user_feature_columns,
                                         item_feature_columns,
                                         # criterion=F.softmax,
                                         l2_reg_linear=1e-5,
                                         l2_reg_embedding=l2_reg_embedding,
                                         init_std=0.0001,
                                         seed=1024,
                                         task='binary',
                                         device='cpu',
                                         **kwargs)
        self.num_sampled = num_sampled

        self.user_dnn = DNN(self.compute_input_dim(user_feature_columns),
                            user_dnn_hidden_units,
                            activation=dnn_activation,
                            l2_reg=l2_reg_dnn,
                            dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn,
                            init_std=init_std,
                            seed=seed,
                            device=device)

    def forward(self, X: torch.Tensor):
        """X：一人一行，包括人和物的输入数据"""
        batch_size = X.size(0)
        user_embedding = self.user_tower(X)
        item_embedding = self.item_tower(X)

        if self.mode == 'user_representation':
            return user_embedding
        if self.mode == 'item_representation':
            return item_embedding

        score = F.cosine_similarity(user_embedding, item_embedding, dim=-1)
        score = score.view(batch_size, -1)
        return score

    def item_tower(self, X):
        if self.mode == 'user_representation':
            return None

        item_emb_list, item_dense_value_list = self.input_from_feature_columns(X,
                                                                               self.item_feature_columns,
                                                                               self.embedding_dict)
        item_embedding = item_emb_list[0]  # (batch, item_list_len, feat_dim)
        return item_embedding

    def user_tower(self, X):
        if self.mode == 'item_representation':
            return None

        user_sparse_emb_list, user_dense_value_list = self.input_from_feature_columns(X,
                                                                                      self.user_feature_columns,
                                                                                      self.embedding_dict)
        user_dnn_input = combined_dnn_input(user_sparse_emb_list, user_dense_value_list)
        user_embedding = self.user_dnn(user_dnn_input)  # (batch_size, embedding_dim)
        user_embedding = user_embedding.unsqueeze(1)  # (batch, 1, embedding_dim)
        return user_embedding
