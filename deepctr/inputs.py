from collections import OrderedDict, defaultdict
from itertools import chain
from typing import Union, List, Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from deepctr.layers import concat, SequencePoolingLayer

DEFAULT_GROUP_NAME = 'default_group'


class BaseFeature(object):
    """特征基类"""

    def __init__(self, name: str, vocab_size=0, emb_dim=4, emb_name: str = None, dtype='int32'):
        self.__name = name
        self.__vocab_size = vocab_size
        self.__emb_dim = emb_dim
        self.__emb_name = emb_name
        self.__dtype = dtype

        if emb_name is None:
            self.__emb_name = name
        if emb_dim == 'auto':
            self.__emb_dim = 6 * int(pow(vocab_size, 0.25))

    @property
    def name(self):
        """特征名称"""
        return self.__name

    @property
    def vocab_size(self):
        """稀疏特征词汇量"""
        return self.__vocab_size

    @property
    def emb_dim(self):
        """特征向量维度"""
        return self.__emb_dim

    @property
    def emb_name(self):
        """特征向量名称"""
        return self.__emb_name

    @property
    def dtype(self):
        """特征向量数据类型"""
        return self.__dtype

    def __hash__(self):
        return self.__name.__hash__()


class SparseFeat(BaseFeature):
    """稀疏特征"""

    def __init__(self, name, vocab_size, emb_dim: Union[int, str] = 4, emb_name=None, dtype='int32',
                 use_hash=False,
                 group_name=DEFAULT_GROUP_NAME):
        assert name, '特征名称必须提供'
        super(SparseFeat, self).__init__(name=name,
                                         vocab_size=vocab_size,
                                         emb_dim=emb_dim,
                                         emb_name=emb_name,
                                         dtype=dtype)
        self.__use_hash = use_hash
        self.__group_name = group_name

        if use_hash:
            print('注意！torch版本暂不支持动态特征哈希，请使用tf版本')

    @property
    def use_hash(self):
        """是否对词汇进行哈希编码"""
        return self.__use_hash

    @property
    def group_name(self):
        """稀疏特征所属组名"""
        return self.__group_name

    def __repr__(self):
        return f'SparseFeat({self.name})'


class SeqFeat(BaseFeature):
    """序列稀疏特征"""

    def __init__(self, sparsefeat: SparseFeat, maxlen: int, combiner='mean', length_name: str = None):
        super(SeqFeat, self).__init__(name=sparsefeat.name,
                                      vocab_size=sparsefeat.vocab_size,
                                      emb_dim=sparsefeat.emb_dim,
                                      emb_name=sparsefeat.emb_name,
                                      dtype=sparsefeat.dtype)
        self.__sparsefeat = sparsefeat
        self.__maxlen = maxlen
        self.__combiner = combiner
        self.__length_name = length_name

    @property
    def sparsefeat(self):
        return self.__sparsefeat

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def maxlen(self):
        """序列特征最大长度"""
        return self.__maxlen

    @property
    def combiner(self):
        """序列特征的组合方法，默认 mean"""
        return self.__combiner

    @property
    def length_name(self):
        """序列特征的长度"""
        return self.__length_name

    def __repr__(self):
        return f'SeqFeat({self.name})'


class DenseFeat(BaseFeature):
    """稠密特征"""

    def __init__(self, name: str, emb_dim: int = 1, dtype: str = 'float32'):
        super(DenseFeat, self).__init__(name=name, emb_dim=emb_dim, dtype=dtype)

    def __repr__(self):
        return f'DenseFeat({self.name})'


FeatList = List[Union[SparseFeat, SeqFeat, DenseFeat]]
FeatClass = Union[type, SparseFeat, SeqFeat, DenseFeat]


def extract_features(features: FeatList, cls: FeatClass) -> FeatList:
    """从给定特征列中提取给定类别的特征列"""
    return list(filter(lambda x: isinstance(x, cls), features)) if len(features) else []


def index_features(features: FeatList) -> Dict[str, Tuple[int, int]]:
    """
    索引特征嵌入位置字典，返回形如 {特征名称: (起点, 终点)} 的有序特征字典。

    :return: OrderedDict{feature_name: (start, start + 1)}
    """
    index_dict: OrderedDict[str, Tuple[int, int]] = OrderedDict()

    start = 0
    for feat in features:
        name = feat.name
        if name in index_dict:
            continue
        if isinstance(feat, SparseFeat):
            index_dict[name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            index_dict[name] = (start, start + feat.emb_dim)
            start += feat.emb_dim
        elif isinstance(feat, SeqFeat):
            index_dict[name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in index_dict:
                index_dict[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError('无效特征列类型: ', type(feat))

    return index_dict


def get_feature_names(idx_dict: Dict[str, Tuple[int, int]]) -> List[str]:
    """获取特征列名"""
    return list(idx_dict.keys())


def slice_feature_data(X: Tensor, feature_name: str, idx_dict: Dict[str, Tuple[int, int]], to_type='long') -> Tensor:
    """
    切片并返回给定张量X中指定特征列的数据
    :param X: 输入张量
    :param feature_name: 特征名称
    :param idx_dict: 特征位置索引字典
    :param to_type: 转换为什么类型
    :return: (特征索引起点, 特征索引重点)
    """
    start, stop = idx_dict[feature_name][0], idx_dict[feature_name][1]
    out = X[:, start:stop]

    if to_type:
        out = getattr(out, to_type)()
    return out


def create_emb_dict(features: FeatList, linear=False, sparse=False, init_std=0.0001, device='cpu') -> nn.ModuleDict:
    """
    创建特征嵌入模块字典。

    :param features: 特征列
    :param linear: 是否构建线性层嵌入字典
    :param sparse: 是否为稀疏特征
    :param init_std: 初始标准差
    :param device: 设备
    :return: 特征字典 nn.ModuleDict{embedding_name：nn.Embedding}
    """
    embedding_dict = nn.ModuleDict({
        feat.emb_name: nn.Embedding(feat.vocab_size, feat.emb_dim if not linear else 1, sparse=sparse)
        for feat in features
    })

    # 初始化线性层嵌入模块权重的标准差
    for emb in embedding_dict.values():
        nn.init.normal_(emb.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def get_feature_values(X: Tensor,
                       features: FeatList,
                       emb_dict: nn.ModuleDict,
                       idx_dict: Dict[str, Tuple[int, int]],
                       support_dense=True,
                       device='cpu') -> Tuple[List[Tensor], List[Tensor]]:
    """
    获取给定张量 X 的特征值。

    :param X: 输入张量 TODO 形状是？
    :param features: 特征列表
    :param emb_dict: 特征嵌入模块字典
    :param idx_dict: 特征位置索引字典
    :param support_dense: 是否必须包含稠密特征
    :param device: 设备
    :returns (sparse_emb_list, dense_value_list)

        - 稀疏嵌入值列表（含序列嵌入值） sparse_emb_list
        - 稠密值列表 dense_value_list
    """
    # 提取三类特征列表
    sparse_features = extract_features(features, SparseFeat)
    seq_features = extract_features(features, SeqFeat)
    dense_features = extract_features(features, DenseFeat)

    # 验证稠密特征是否支持
    if not support_dense and len(dense_features) > 0:
        raise ValueError('dnn_features 中不支持 DenseFeat')

    # 生成稀疏特征嵌入值
    sparse_emb_list = [emb_dict[feat.emb_name](X[:, idx_dict[feat.name][0]:idx_dict[feat.name][1]].long())
                       for feat in sparse_features]

    # 生成序列稀疏特征嵌入值
    seq_emb_dict = get_seq_emb_dict(X, seq_features, emb_dict, idx_dict)
    seq_emb_list = get_seq_pooling_list(X, seq_features, seq_emb_dict, idx_dict, device=device)

    # 合并稀疏特征嵌入值
    sparse_emb_list += seq_emb_list

    # 提取稠密值列表
    dense_value_list = [X[: idx_dict[feat.name][0]:idx_dict[feat.name][1]] for feat in dense_features]

    return sparse_emb_list, dense_value_list


def get_group_emb_dict(X: Tensor,
                       features: List[Union[SparseFeat, SeqFeat]],
                       idx_dict: Dict[str, Tuple[int, int]],
                       emb_dict: nn.ModuleDict,
                       return_feat_list=(), to_list=False) \
        -> Union[Dict[str, List[Tensor]], List[Tensor]]:
    """
    获取给定张量X和稀疏特征的分组嵌入值，返回分组后的嵌入值字典或列表。

    :param X:输入张量 [batch_size * hidden_dim]
    :param features: 特征列——稀疏特征或序列特征
    :param idx_dict: 特征索引字典
    :param emb_dict: 特征嵌入字典
    :param return_feat_list: 指定返回的特征名称元组，默认返回所有特征
    :param to_list: 是否返回列表，默认False时返回字典
    :return: 稀疏特征字典{特征分组名称: [嵌入值]} 或 [嵌入值]
    """
    group_emb_dict = defaultdict(list)
    for feat in features:
        feature_name = feat.name
        embedding_name = feat.emb_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list:
            # TODO: add hash function
            # if fc.use_hash:
            #     raise NotImplementedError("hash function is not implemented in this version!")
            lookup_idx = np.array(idx_dict[feature_name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            emb = emb_dict[embedding_name](input_tensor)
            group_emb_dict[feat.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_emb_dict.values()))
    return group_emb_dict


def get_seq_emb_dict(X: Tensor,
                     seq_features: List[SeqFeat],
                     emb_dict: nn.ModuleDict,
                     idx_dict: Dict[str, Tuple[int, int]]) -> Dict[str, Tensor]:
    """
    查找给定张量X和序列特征的嵌入值字典或列表。

    :param X: 输入张量 [batch_size * hidden_dim]
    :param seq_features: 序列特征
     :param emb_dict: 特征嵌入模块字典
    :param idx_dict: 特征位置索引字典
    :return: 序列稀疏特征字典{特征名称: 嵌入值}
    """
    seq_emb_dict = {}

    for feat in seq_features:
        feature_name = feat.name
        embedding_name = feat.emb_name
        if feat.use_hash:
            # lookup_idx = Hash(feat.vocabulary_size, mask_zero=True)(seq_input_dict[feature_name])
            # TODO 添加 hash 函数
            lookup_idx = idx_dict[feature_name]
        else:
            lookup_idx = idx_dict[feature_name]

        inputs = X[:, lookup_idx[0]:lookup_idx[1]].long()  # TODO 为什么要转成长整型？
        seq_emb_dict[feature_name] = emb_dict[embedding_name](inputs)

    return seq_emb_dict


def get_seq_pooling_list(X: Tensor,
                         seq_features: List[SeqFeat],
                         seq_emb_dict: Dict[str, Tensor],
                         idx_dict: Dict[str, Tuple[int, int]],
                         device='cpu') -> List[Tensor]:
    """获取给定张量X和序列特征嵌入值的池化列表"""
    seq_emb_list = []
    for feat in seq_features:
        seq_emb = seq_emb_dict[feat.name]
        if feat.length_name is None:
            mask = X[:, idx_dict[feat.name][0]:idx_dict[feat.name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=True, device=device).forward(
                [seq_emb, mask])
        else:
            seq_length = X[:, idx_dict[feat.length_name][0]:idx_dict[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, support_masking=False, device=device).forward(
                [seq_emb, seq_length])
        seq_emb_list.append(emb)
    return seq_emb_list


def combined_dnn_input(sparse_emb_list: List[Tensor], dense_value_list: List[Tensor]) -> Tensor:
    """合并 DNN 层稀疏嵌入值和稠密值"""
    if len(sparse_emb_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(torch.cat(sparse_emb_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_emb_list) > 0:
        return torch.flatten(torch.cat(sparse_emb_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def get_dense_input(X: Tensor, dense_features: List[DenseFeat], idx_dict: Dict[str, Tuple[int, int]]) -> List[Tensor]:
    """获取给定输入 X 中稠密特征的向量列表"""
    dense_input_list = []
    for feat in dense_features:
        lookup_idx = np.array(idx_dict[feat.name])
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(X: Tensor, maxlen_column: str, idx_dict: Dict[str, Tuple[int, int]]):
    """查找给定张量 X 中 length_name 对应部分的张量"""
    if maxlen_column is None or len(maxlen_column) == 0:
        raise ValueError("请给 DIN/DIEN 的序列特征增加 maxlen 列")
    lookup_idx = np.array(idx_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()


class FeatureManager(object):
    """特征管理器"""

    def __init__(self, features: FeatList, linear=False, sparse=False, init_std=1e-4, device='cpu'):
        """
        实例化一个特征管理器。

        :param features: 特征列表
        :param linear: 是否为线性层所需的特征管理器
        :param init_std: 初始标准差
        :param device: 设备
        """
        self.features = features

        self.linear = linear  # 是否为线性模型特征管理器
        self.init_std = init_std
        self.device = device

        # 提取特征
        self.sparse_features = extract_features(self.features, SparseFeat)
        self.seq_features = extract_features(self.features, SeqFeat)
        self.dense_features = extract_features(self.features, DenseFeat)

        # 初始化特征位置索引字典
        self.index_dict = index_features(self.features)

        # 初始化稀疏特征嵌入字典
        self.embedding_dict = create_emb_dict(self.sparse_features + self.seq_features,
                                              linear=linear,
                                              sparse=sparse,
                                              init_std=init_std,
                                              device=device)

    def sequence_embedding_lookup(self, X: Tensor) -> Dict[str, Tensor]:
        """
        返回给定张量 X 的序列特征嵌入值字典或列表。

        :param X: 输入张量 [batch_size * hidden_dim]
        :return: 序列稀疏特征字典{特征名称: 嵌入值}
        """
        seq_emb_dict = get_seq_emb_dict(X, self.seq_features, self.embedding_dict, self.index_dict)
        return seq_emb_dict

    def get_sequence_pooling_list(self, seq_emb_dict: Dict[str, Tensor], X) -> List[Tensor]:
        """获取序列特征的池化嵌入值列表"""
        seq_emb_list = get_seq_pooling_list(X, self.seq_features, seq_emb_dict, self.index_dict, device=self.device)
        return seq_emb_list

    @property
    def names(self) -> List[str]:
        """获取特征列名"""
        return list(self.index_dict.keys())
