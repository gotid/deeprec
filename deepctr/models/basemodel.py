from deepctr.inputs import *
from deepctr.layers import PredictionLayer


class Linear(torch.nn.Module):
    def __init__(self, linear_features: FeatList, feature_idx_dict: Dict[str, Tuple[int, int]],
                 init_std=0.0001, device='cpu'):
        """
        特征列线性层。

        :param linear_features: 特征列列表
        :param init_std: 初始化标准差
        :param device: 设备
       """
        super(Linear, self).__init__()
        self.features = linear_features
        self.feature_idx_dict = feature_idx_dict
        self.device = device
        self.weight = None

        # 创建线性特征的嵌入模块字典
        self.feature_emb_dict = create_emb_dict(linear_features, linear=True, sparse=False,
                                                init_std=init_std, device=device)

        # 存在稠密特征的线性层，需初始化权重
        self.dense_features = extract_features(linear_features, DenseFeat)
        if len(self.dense_features) > 0:
            self.weight = nn.Parameter(
                Tensor(sum(feat.emb_dim for feat in self.dense_features), 1).to(device))
            nn.init.normal_(self.weight, std=init_std)

    def forward(self, inputs: Tensor, sparse_feat_refine_weight=None):
        # 获取稀疏特征嵌入值和稠密值列表
        sparse_emb_list, dense_value_list = get_feature_values(inputs,
                                                               self.features,
                                                               self.feature_emb_dict,
                                                               self.feature_idx_dict)

        # 初始化线性回归值
        linear_logit = torch.zeros([inputs.shape[0], 1]).to(self.device)

        # 累加稀疏嵌入值
        if len(sparse_emb_list) > 0:
            linear_logit = linear_logit.to(sparse_emb_list[0].device)
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
    def __init__(self,
                 linear_features: FeatList,
                 dnn_features: FeatList,
                 l2_reg_linear=1e-5,
                 l2_reg_emb=1e-5,
                 seed=1024, init_std=0.0001,
                 task='binary',  # binary | multiclass | regression
                 device='cpu', gpus=None):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError('gpus[0] 应和 device 是同一个 gpu')

        # 索引全部特征位置字典
        self.feature_idx_dict = index_features(linear_features + dnn_features)

        # 初始化 DNN 特征
        self.dnn_features = dnn_features
        self.dnn_emb_dict = create_emb_dict(dnn_features, linear=False, sparse=False, init_std=init_std, device=device)

        # 初始化 Linear 特征
        self.linear_model = Linear(linear_features, self.feature_idx_dict, device=device)

        # 初始化 DNN 特征
        self.all_features = linear_features + dnn_features
        self.dnn_features = dnn_features

        # 初始正则化权重
        self.regularization_weight = []
        self.add_regularization_weight(self.dnn_emb_dict.parameters(), l2=l2_reg_emb)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        # 输出值
        self.out = PredictionLayer(task)
        self.to(device)

    def fit(self):
        pass

    def evaluate(self, x, y, batch_size=256):
        pass

    def predict(self, x, batch_size=256):
        pass

    def add_regularization_weight(self, weights, l1=0.0, l2=0.0):
        """增加正则化权重"""
        if isinstance(weights, nn.parameter.Parameter):
            weights = [weights]
        else:
            weights = list(weights)
        self.regularization_weight.append((weights, l1, l2))

    def get_regularization_loss(self):
        """获取正则化损失值"""
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weights, l1, l2 in self.regularization_weight:
            if isinstance(weights, tuple):
                parameter = weights[1]  # named_parameters
            else:
                parameter = weights
            if l1 > 0:
                total_reg_loss += torch.sum(l1 * torch.abs(parameter))
            if l2 > 0:
                try:
                    total_reg_loss += torch.sum(l2 * torch.square(parameter))
                except AttributeError:
                    total_reg_loss += torch.sum(l2 * parameter * parameter)
        return total_reg_loss

    def input_from_features(self, X, features: FeatList, support_dense=True):
        return get_feature_values(X, features, self.dnn_emb_dict, self.feature_idx_dict, support_dense)

    @classmethod
    def ndim(cls, features: FeatList, sparse=True, dense=True, group=False):
        """返回模型特征的总维度"""
        total_dim = 0

        # 提取特征
        sparse_features = extract_features(features, cls=SparseFeat) + extract_features(features, cls=SeqFeat)
        dense_features = extract_features(features, cls=DenseFeat)

        # 稀疏特征维度
        if sparse:
            if group:
                sparse_input_dim = len(sparse_features)
            else:
                sparse_input_dim = sum(feat.emb_dim for feat in sparse_features)
            total_dim += sparse_input_dim

        # 稠密特征维度
        if dense:
            dense_input_dim = sum(map(lambda x: x.emb_dim, dense_features))
            total_dim += dense_input_dim

        return total_dim

    def rebuild_feature_index(self, feature_columns):
        """为了单独预测 user/item vector，需重算特征列索引位置"""
        self.feature_idx_dict = index_features(feature_columns)
        return self
