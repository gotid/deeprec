from abc import abstractmethod
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import TensorDataset, DataLoader

from deepctr.inputs import build_feature_index, create_embedding_dict, SparseFeat, DenseFeat, SequenceFeat, \
    sequence_embedding_lookup, get_sequence_pooling_list


class PLBaseModel(LightningModule):
    """DeepMatch 所有模型的基类
    模型参考：https://github.com/Rose-STL-Lab/torchTS/blob/main/torchts/nn/model.py
    """

    def __init__(self,
                 user_feature_columns,
                 item_feature_columns,
                 optimizer=None,
                 optimizer_args=None,
                 criterion=F.mse_loss,
                 criterion_args=None,
                 scheduler=None,
                 scheduler_args=None,
                 scaler=None,
                 config=None,
                 **kwargs):
        super().__init__()
        if config is None:
            config = {}
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.config = config
        self.config.update(kwargs)

        # 输出模式为 logits 还是 user/item 的嵌入向量
        self.mode = self.config.get('mode', 'train')

        self.linear_feature_columns = user_feature_columns + item_feature_columns
        self.dnn_feature_columns = self.linear_feature_columns

        # 在 pl 中不需要 to(device)
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))

        # 构建特征索引字典
        self.feature_index = build_feature_index(self.linear_feature_columns)

        # 创建特征嵌入字典
        self.embedding_dict = create_embedding_dict(self.dnn_feature_columns,
                                                    init_std=self.config.get('init_std'))

        # 实例化线性模型
        self.linear_model = Linear(self.linear_feature_columns, self.feature_index, device=self.device)

        # 增加嵌入层和线性层的规范化权重
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=self.config.get('l2_reg_embedding'))
        self.add_regularization_weight(self.linear_model.parameters(), l2=self.config.get('l2_reg_linear'))

        # loss 标准，默认 F.mse_loss
        self.criterion = criterion
        self.criterion_args = criterion_args
        self.scaler = scaler  # 定标器（用来查看翻转后的 y_true/y_pred）
        self.gpus = self.config.get('gpus', 0)

        # 配置优化器
        optimizer = self.init_optimizer(optimizer)
        if optimizer_args is not None:
            self.optimizer = partial(optimizer, **optimizer_args)
        else:
            self.optimizer = optimizer

        # 配置调度器
        if scheduler is not None and scheduler_args is not None:
            self.scheduler = partial(scheduler, **scheduler_args)
        else:
            self.scheduler = scheduler

    def fit(self, x, y, max_epochs=10, batch_size=128):
        """
        使模型适合数据

        :param x: torch.Tensor 输入数据
        :param y: torch.Tensor 输出数据
        :param max_epochs: 训练轮数
        :param batch_size: 训练批量
        """
        x, y = self.construct_input(x, y)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = Trainer(max_epochs=max_epochs, gpus=self.gpus)
        trainer.fit(self, loader)

    def training_step(self, batch, batch_idx):
        """
        训练一批数据

        :param batch: torch.Tensor，torch.utils.data.DataLoader 返回的一批数据
        :param batch_idx: int, 批次索引
        """
        train_loss = self._step(batch, batch_idx, len(self.trainer.train_dataloader))
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        验证一批数据

        :param batch: torch.Tensor，torch.utils.data.DataLoader 返回的一批数据
        :param batch_idx: int, 批次索引
        """
        val_loss = self._step(batch, batch_idx, len(self.trainer.val_dataloader))
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        测试一批数据

        :param batch: torch.Tensor，torch.utils.data.DataLoader 返回的一批数据
        :param batch_idx: int, 批次索引
        """
        test_loss = self._step(batch, batch_idx, len(self.trainer.test_dataloader))
        self.log('test_loss', test_loss)
        return test_loss

    @abstractmethod
    def forward(self, x):
        """
        前向计算，抽象方法由子类实现。

        :param x: torch.Tensor，输入数据
        :return: torch.Tensor，预测数据
        """

    def predict(self, x):
        """
        运行模型推理。
        :param x: torch.Tensor，输入数据
        :return: torch.Tensor，预测数据
        """
        return self(x).detach()

    def full_predict(self, x, batch_size=256):
        # 若输入字典，则取出字典值
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        # 若输入1维，则变为2维
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        x = torch.from_numpy(np.concatenate(x, axis=-1))
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ret = []
        for batch in loader:
            pred = self.predict(batch[0])
            ret.append(pred)
        return torch.cat(ret, axis=0)

    def _step(self, batch, batch_idx, num_batches):
        """
        用于训练、验证、测试的步骤。

        :param batch: torch.Tensor，torch.utils.data.DataLoader 返回的一批数据
        :param batch_idx: int, 批次索引
        :param num_batches: 总批次
        :return: 该步骤 loss 值
        """
        x, y = self.prepare_batch(batch)

        # 看过的批次
        batches_seen = batch_idx + self.current_epoch * num_batches if self.training else batch_idx
        self.log('批次', float(batches_seen))

        # 预测
        pred = self(x)

        # 使用定标器翻转标签值
        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)
            pred = self.scaler.inverse_transform(pred)

        # 计算损失
        if self.criterion == F.cross_entropy:
            y = y.long()
        if self.criterion_args is not None:
            loss = self.criterion(pred, y, **self.criterion_args)
        else:
            loss = self.criterion(pred, y)

        return loss

    def prepare_batch(self, batch):
        """准备一批数据"""
        return batch

    def construct_input(self, x, y):
        """构造模型输入"""

        # 取字典值数组
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        # 扩维：1维转2维
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        # 拼接
        x = torch.from_numpy(np.concatenate(x, axis=-1))
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def configure_optimizers(self):
        """
        配置优化器及学习速率

        :return: 优化器 torch.optim.Optimizer
        """
        optimizer = self.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer

    @classmethod
    def init_optimizer(cls, optimizer_or_name):
        """返回指定的优化器，如果是字符串则兑换成优化器"""
        if isinstance(optimizer_or_name, str):
            if optimizer_or_name.lower() == 'adam':
                return torch.optim.Adam
            elif optimizer_or_name.lower() == 'sgd':
                return torch.optim.SGD
        else:
            return optimizer_or_name

    def get_regularization_loss(self):
        """获取规范化损失值"""
        total_reg_loss = torch.zeros((1,))

        for weights, l1, l2 in self.regularization_weight:
            for w in weights:
                if isinstance(w, tuple):
                    parameter = w[1]  # 命名参数 named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_regularization_weight(self, weights, l1=0.0, l2=0.0):
        """增加规范化权重值"""
        if isinstance(weights, torch.nn.parameter.Parameter):
            weights = [weights]
        else:
            weights = list(weights)
        self.regularization_weight.append((weights, l1, l2))

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        """
        从特征列生成稀疏特征的嵌入值列表，稠密值列表
        :param X: torch.Tensor，输入数据
        :param feature_columns: 特征列
        :param embedding_dict: 特征嵌入层字典
        :param support_dense: 是否支持稠密特征
        :return: 稀疏特征嵌入值列表，稠密值列表
        """
        # 提取稀疏特征列
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
            feature_columns) else []

        # 提取稠密特征列
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        # 提取变长稀疏特征列
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SequenceFeat), feature_columns)) if len(
            feature_columns) else []

        if len(dense_feature_columns) > 0 and not support_dense:
            return ValueError('dense_feature_columns 不支持 DenseFeat ')

        # 生成稀疏特征嵌入值列表
        sparse_emb_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
            for feat in sparse_feature_columns]

        # 生成变长稀疏特征嵌入值字典（这里返回的就是 item 的 embedding）
        seq_emb_dict = sequence_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                 varlen_sparse_feature_columns)
        varlen_sparse_emb_list = get_sequence_pooling_list(seq_emb_dict, X, self.feature_index,
                                                           varlen_sparse_feature_columns, self.device)

        # 生成稠密特征值列表
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                            for feat in dense_feature_columns]

        return sparse_emb_list + varlen_sparse_emb_list, dense_value_list

    @staticmethod
    def compute_input_dim(feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        """
        计算输入总维数

        :param feature_columns: 特征列
        :param include_sparse: 是否包含稀疏特征列
        :param include_dense: 是否包含稠密特征列
        :param feature_group: 特征分组
        :return:
        """
        # 提取稀疏特征列（包括变长稀疏特征列）
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, SequenceFeat)), feature_columns)) if len(
            feature_columns) else []

        # 提取稠密特征列
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        # 计算稠密特征列总维数
        dense_input_dim = sum(map(lambda x: x.dim, dense_feature_columns))

        # 计算稀疏特征列总维数
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)

        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim

        return input_dim

    def rebuild_feature_index(self, feature_columns):
        """为了单独预测 user/item vector，需重算特征列索引位置"""
        self.feature_index = build_feature_index(feature_columns)
        return self


class Linear(nn.Module):
    """DeepMatch 的线性转换模型"""

    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        """
        实例化线性转换层

        :param feature_columns: 特征列表
        :param feature_index: 特征索引，记录了每个特征的起止索引位
        :param init_std: 正态分布标准差
        :param device: 设备
        """
        super(Linear, self).__init__()
        self.device = device
        self.feature_index = feature_index

        # 提取稀疏特征列
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        # 提取稠密特征列
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        # 提取变长稀疏特征列
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SequenceFeat), feature_columns)) if len(feature_columns) else []

        # 创建稀疏特征词典（含变长稀疏特征列，不含稠密特征列）
        self.embedding_dict = create_embedding_dict(feature_columns, init_std, linear=True, device=device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        # 稠密特征维度之和作为线性层权重参数
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dim for fc in self.dense_feature_columns), 1))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        # 生成稀疏特征的嵌入值列表
        sparse_emb_list = [
            self.embedding_dict[feat.embedding_name](
                X[: self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()
            ) for feat in self.sparse_feature_columns
        ]

        # 提取 X 中的稠密特征值
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                            for feat in self.dense_feature_columns]

        # 查找变长稀疏特征的嵌入词典
        seq_emb_dict = sequence_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                 self.varlen_sparse_feature_columns)
        # 获取变长稀疏特征的嵌入值列表
        varlen_emb_list = get_sequence_pooling_list(seq_emb_dict, X, self.feature_index,
                                                    self.varlen_sparse_feature_columns, self.device)

        # 合并稀疏特征值和变长稀疏特征值
        sparse_emb_list += varlen_emb_list

        # 准备线性逻辑回归张量
        linear_logit = torch.zeros([X.shape[0], 1]).to(sparse_emb_list[0].device)

        # 累加稀疏特征逻辑回归值
        if len(sparse_emb_list) > 0:
            sparse_emb_cat = torch.cat(sparse_emb_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                sparse_emb_cat = sparse_emb_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_emb_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit

        # 累加稠密特征逻辑回归值
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit
