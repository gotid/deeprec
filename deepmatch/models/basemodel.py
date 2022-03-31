from abc import abstractmethod
from functools import partial

import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import TensorDataset, DataLoader

from deepctr.inputs import *
from deepctr.models.basemodel import Linear


class BaseModel(LightningModule):
    """DeepMatch 所有模型的基类
    模型参考：https://github.com/Rose-STL-Lab/torchTS/blob/main/torchts/nn/model.py
    """

    def __init__(self,
                 user_features,
                 item_features,
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
        self.user_features = user_features
        self.item_features = item_features
        self.config = config
        self.config.update(kwargs)

        # 输出模式为 logits 还是 user/item 的嵌入向量
        self.mode = self.config.get('mode', 'train')

        self.linear_features = user_features + item_features
        self.dnn_features = self.linear_features

        # 在 pl 中不需要 to(device)
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))

        # 构建特征索引字典
        self.feature_idx_dict = index_features(self.linear_features)

        # 创建特征嵌入字典
        self.dnn_emb_dict = create_emb_dict(self.dnn_features, init_std=self.config.get('init_std'))

        # 实例化线性模型
        self.linear_model = Linear(self.linear_features, self.feature_idx_dict, device=self.device)

        # 增加嵌入层和线性层的规范化权重
        self.regularization_weight = []
        self.add_regularization_weight(self.dnn_emb_dict.parameters(), l2=self.config.get('l2_reg_emb'))
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
        :param max_epochs: 最大训练轮数
        :param batch_size: 每批数量
        """
        x, y = self.construct_input(x, y)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=self.config.get('shuffle', True),
                            num_workers=self.config.get('num_workers', 8))
        trainer = Trainer(max_epochs=max_epochs,
                          gpus=self.gpus,
                          log_every_n_steps=self.config.get('log_every_n_steps', 50))
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
        val_loss = self._step(batch, batch_idx, len(self.trainer.val_dataloaders))
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        测试一批数据

        :param batch: torch.Tensor，torch.utils.data.DataLoader 返回的一批数据
        :param batch_idx: int, 批次索引
        """
        test_loss = self._step(batch, batch_idx, len(self.trainer.test_dataloaders))
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

    def full_predict(self, X, batch_size=256):
        # 若输入字典，则取出字典值
        if isinstance(X, dict):
            X = [X[feature] for feature in self.feature_idx_dict]

        # 若输入1维，则变为2维
        for i in range(len(X)):
            if len(X[i].shape) == 1:
                X[i] = np.expand_dims(X[i], axis=1)

        X = torch.from_numpy(np.concatenate(X, axis=-1))
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ret = []
        for batch in loader:
            pred = self.predict(batch[0])
            ret.append(pred)
        return torch.cat(ret, dim=0)

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
            x = [x[feature] for feature in self.feature_idx_dict]

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

    def add_regularization_weight(self, weights, l1=0.0, l2=0.0):
        """增加规范化权重值"""
        if isinstance(weights, torch.nn.parameter.Parameter):
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

    def input_from_features(self, X, features: FeatList, emb_dict: nn.ModuleDict, support_dense=True):
        """
        从特征列生成稀疏特征的嵌入值列表，稠密值列表
        :param X: torch.Tensor，输入数据
        :param features: 特征列表
        :param emb_dict: 特征嵌入模块字典
        :param support_dense: 是否支持稠密特征
        :return: 稀疏特征嵌入值列表，稠密值列表
        """
        return get_feature_values(X, features, emb_dict, self.feature_idx_dict, support_dense)

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

    def rebuild_feature_index(self, features):
        """为了单独预测 user/item emb，需重算特征列索引位置"""
        self.feature_idx_dict = index_features(features)
        return self
