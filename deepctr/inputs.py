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

    def __init__(self, name: str, vocabulary_size=0, embedding_dim=4, embedding_name: str = None, dtype='int32'):
        self.__name = name
        self.__vocabulary_size = vocabulary_size
        self.__embedding_dim = embedding_dim
        self.__embedding_name = embedding_name
        self.__dtype = dtype

        if embedding_name is None:
            self.__embedding_name = name
        if embedding_dim == 'auto':
            self.__embedding_dim = 6 * int(pow(vocabulary_size, 0.25))

    @property
    def name(self):
        """特征名称"""
        return self.__name

    @property
    def vocabulary_size(self):
        """稀疏特征词汇量"""
        return self.__vocabulary_size

    @property
    def embedding_dim(self):
        """特征向量维度"""
        return self.__embedding_dim

    @property
    def embedding_name(self):
        """特征向量名称"""
        return self.__embedding_name

    @property
    def dtype(self):
        """特征向量数据类型"""
        return self.__dtype

    def __hash__(self):
        return self.__name.__hash__()


class SparseFeat(BaseFeature):
    """稀疏特征"""

    def __init__(self, name, vocabulary_size, embedding_dim: Union[int, str] = 4, embedding_name=None, dtype='int32',
                 use_hash=False,
                 group_name=DEFAULT_GROUP_NAME):
        assert name, '特征名称必须提供'
        super(SparseFeat, self).__init__(name=name,
                                         vocabulary_size=vocabulary_size,
                                         embedding_dim=embedding_dim,
                                         embedding_name=embedding_name,
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


class SequenceFeat(BaseFeature):
    """序列稀疏特征"""

    def __init__(self, sparsefeat: SparseFeat, maxlen: int, combiner='mean', length_name: str = None):
        super(SequenceFeat, self).__init__(name=sparsefeat.name,
                                           vocabulary_size=sparsefeat.vocabulary_size,
                                           embedding_dim=sparsefeat.embedding_dim,
                                           embedding_name=sparsefeat.embedding_name,
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
        """可变稀疏特征最大长度"""
        return self.__maxlen

    @property
    def combiner(self):
        """可变稀疏特征的组合方法，默认 mean"""
        return self.__combiner

    @property
    def length_name(self):
        """可变稀疏特征的长度"""
        return self.__length_name

    def __repr__(self):
        return f'SequenceFeat({self.name})'


class DenseFeat(BaseFeature):
    """稠密特征"""

    def __init__(self, name: str, embedding_dim: int = 1, dtype: str = 'float32'):
        super(DenseFeat, self).__init__(name=name, embedding_dim=embedding_dim, dtype=dtype)

    def __repr__(self):
        return f'DenseFeat({self.name})'


class Features(object):
    """特征管理器"""

    def __init__(self, columns: List[Union[SparseFeat, SequenceFeat, DenseFeat]],
                 is_linear=False, init_std=1e-4, device='cpu'):
        """
        实例化一个特征管理器。

        :param columns: 特征列数组
        :param is_linear: 是否为线性层所需的特征管理器
        :param init_std: 初始标准差
        :param device: 设备
        """
        self.columns = columns
        self.is_linear = is_linear
        self.init_std = init_std
        self.device = device

        # 提取特征列
        self.sparse_columns = self.extract(SparseFeat)
        self.sequence_columns = self.extract(SequenceFeat)
        self.dense_columns = self.extract(DenseFeat)

        # 初始化特征位置索引字典
        self.index_dict = self.init_index()

        # 初始化稀疏特征嵌入字典
        self.embedding_dict = self.init_embedding()

    def transform(self, X: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        转换输入张量为值元组

        返回元组：
            - 稀疏嵌入值列表（含序列稀疏特征嵌入值） sparse_emb_list
            - 稠密值列表 dense_value_list
        """

        # 生成稀疏特征嵌入值
        sparse_emb_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.index_dict[feat.name][0]:self.index_dict[feat.name][1]].long())
            for feat in self.sparse_columns]

        # 生成序列稀疏特征嵌入值
        seq_emb_dict = self.sequence_embedding_lookup(X)
        seq_emb_list = self.get_sequence_pooling_list(seq_emb_dict, X)

        # # 合并稀疏特征嵌入值
        sparse_emb_list += seq_emb_list

        # 提取稠密值列表
        dense_value_list = [X[: self.index_dict[feat.name][0]:self.index_dict[feat.name][1]]
                            for feat in self.dense_columns]

        return sparse_emb_list, dense_value_list

    def combined_dnn_input(self, sparse_emb_list, dense_value_list) -> Tensor:
        """合并稀疏嵌入和稠密值"""
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

    def extract(self, clazz: Union[type, SparseFeat, SequenceFeat, DenseFeat]) -> \
            List[Union[SparseFeat, SequenceFeat, DenseFeat]]:
        """提取指定类别的特征列"""
        return list(filter(lambda x: isinstance(x, clazz), self.columns)) if len(self.columns) else []

    def init_index(self) -> Dict[str, Tuple[int, ...]]:
        """
        初始化特征索引，返回形如 {特征名称: (起点, 终点)} 的有序特征字典。

        :return: OrderedDict{feature_name: (start, start + 1)}
        """
        index_dict: OrderedDict[str, Tuple[int, ...]] = OrderedDict()

        start = 0
        for feat in self.columns:
            name = feat.name
            if name in index_dict:
                continue
            if isinstance(feat, SparseFeat):
                index_dict[name] = (start, start + 1)
                start += 1
            elif isinstance(feat, DenseFeat):
                index_dict[name] = (start, start + feat.embedding_dim)
                start += feat.embedding_dim
            elif isinstance(feat, SequenceFeat):
                index_dict[name] = (start, start + feat.maxlen)
                start += feat.maxlen
                if feat.length_name is not None and feat.length_name not in index_dict:
                    index_dict[feat.length_name] = (start, start + 1)
                    start += 1
            else:
                raise TypeError('无效特征列类型: ', type(feat))

        return index_dict

    def init_embedding(self) -> nn.ModuleDict:
        """
        初始化嵌入字典（含稀疏特征、序列稀疏特征）。

        :return: 特征字典 nn.ModuleDict{特征嵌入名称：nn.Embedding}
        """
        embedding_dict = nn.ModuleDict({
            feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not self.is_linear else 1)
            for feat in self.sparse_columns + self.sequence_columns
        })

        # 初始化线性层嵌入模块权重的标准差
        for emb in embedding_dict.values():
            nn.init.normal_(emb.weight, mean=0, std=self.init_std)

        return embedding_dict.to(self.device)

    def embedding_lookup(self, X: Tensor, return_feat_list=(), to_list=False) \
            -> Union[Dict[str, List[nn.Embedding]], List[nn.Embedding]]:
        """
        返回给定张量的稀疏特征嵌入值字典或列表

        :param X:输入张量 [batch_size * hidden_dim]
        :param return_feat_list: 指定返回的特征名称元组，默认返回所有特征
        :param to_list: 是否返回列表，默认False时返回字典
        :return: 稀疏特征字典{特征名称: 嵌入值} 或 嵌入值列表
        """
        group_embedding_dict = defaultdict(list)
        for feat in self.sparse_columns:
            feature_name = feat.name
            embedding_name = feat.embedding_name
            if len(return_feat_list) == 0 or feature_name in return_feat_list:
                # TODO: add hash function
                # if fc.use_hash:
                #     raise NotImplementedError("hash function is not implemented in this version!")
                lookup_idx = np.array(self.index_dict[feature_name])
                input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
                emb = self.embedding_dict[embedding_name](input_tensor)
                group_embedding_dict[feat.group_name].append(emb)
        if to_list:
            return list(chain.from_iterable(group_embedding_dict.values()))
        return group_embedding_dict

    def sequence_embedding_lookup(self, X: Tensor) -> Dict[str, Tensor]:
        """
        返回给定张量 X 的序列特征嵌入值字典或列表。

        :param X: 输入张量 [batch_size * hidden_dim]
        :return: 序列稀疏特征字典{特征名称: 嵌入值}
        """
        seq_emb_dict = {}

        for feat in self.sequence_columns:
            feature_name = feat.name
            embedding_name = feat.embedding_name
            if feat.use_hash:
                # lookup_idx = Hash(feat.vocabulary_size, mask_zero=True)(seq_input_dict[feature_name])
                # TODO 添加 hash 函数
                lookup_idx = self.index_dict[feature_name]
            else:
                lookup_idx = self.index_dict[feature_name]

            seq_emb_dict[feature_name] = self.embedding_dict[embedding_name](X[:, lookup_idx[0]:lookup_idx[1]].long())

        return seq_emb_dict

    def get_sequence_pooling_list(self, seq_emb_dict: Dict[str, Tensor], X) -> List[Tensor]:
        """获取序列特征的池化嵌入值列表"""
        seq_emb_list = []
        for feat in self.sequence_columns:
            seq_emb = seq_emb_dict[feat.name]
            if feat.length_name is None:
                mask = X[:, self.index_dict[feat.name][0]:self.index_dict[feat.name][1]].long() != 0
                emb = SequencePoolingLayer(mode=feat.combiner, support_masking=True, device=self.device).forward(
                    [seq_emb, mask])
            else:
                seq_length = X[:, self.index_dict[feat.length_name][0]:self.index_dict[feat.length_name][1]].long()
                emb = SequencePoolingLayer(mode=feat.combiner, support_masking=False, device=self.device).forward(
                    [seq_emb, seq_length])
            seq_emb_list.append(emb)
        return seq_emb_list

    def get_dense_input(self, X: Tensor) -> List[Tensor]:
        """获取给定输入 X 中稠密特征的向量列表"""
        dense_input_list = []
        for feat in self.dense_columns:
            lookup_idx = np.array(self.index_dict[feat.name])
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
            dense_input_list.append(input_tensor)
        return dense_input_list

    def maxlen_lookup(self, X: Tensor, maxlen_column):
        """查找给定张量 X 中 length_name 对应部分的张量"""
        if maxlen_column is None or len(maxlen_column) == 0:
            raise ValueError("请给 DIN/DIEN 的序列特征增加 maxlen 列")
        lookup_idx = np.array(self.index_dict[maxlen_column[0]])
        return X[:, lookup_idx[0]:lookup_idx[1]].long()

    @property
    def names(self) -> List[str]:
        """获取特征列名"""
        return list(self.index_dict.keys())
