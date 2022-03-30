import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer


class LocalActivationUnit(nn.Module):
    """深度兴趣网络中使用 LocalActivationUnit 表示给定不同的候选项目，用户兴趣会自适应地变化。"""

    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid',
                 dropout_rate=0, dice_dim=3, l2_reg=0, use_bn=False):
        """
        实例化本地激活单元。

        :param hidden_units: 正整数列表，注意力网络的层数及每层单元
        :param embedding_dim: 嵌入维度
        :param activation: 用于注意力网络的激活函数
        :param dropout_rate: [0., 1.) 随机归零的概率
        :param dice_dim:
        :param l2_reg:
        :param use_bn:
        """
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(input_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        """
        利用 DNN 求 query 和 user_behavior 的注意力得分。

        :param query:           请求物品，尺寸 -> (batch_size, 1, embedding_size)
        :param user_behavior:   用户行为，尺寸 -> (batch_size,  time_seq_len, embedding_size)
        """
        user_behavior_len = user_behavior.size(1)

        queries = query.extend(-1, user_behavior_len, -1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior],
                                    dim=-1)  # 减法模拟了 vectors 之间的差异

        attention_output = self.dnn(attention_input)

        attention_score = self.dense(attention_output)  # [batch_size, time_seq_len, 1]

        return attention_score


class DNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units,
                 activation='relu',
                 l2_reg=0,
                 dropout_rate=0,
                 use_bn=False,
                 init_std=1e-4,
                 dice_dim=3,
                 seed=1024,
                 device='cpu'):
        """实例化多层感知机。

        :param input_dim: 输入特征维度
        :param hidden_units: 正整数列表，层数和单元
        :param activation: 使用的激活函数
        :param l2_reg: [0., 1.]，内核权重矩阵的L2正则化器
        :param dropout_rate: [0., 1.) 随机归零的概率
        :param use_bn: 激活前是否进行批量规格化
        :param init_std: 正态分布的标准差
        :param dice_dim: Dice 维数，2维|3维
        :param seed: 随机种子数
        """
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError('hidden_units 为空!!')
        hidden_units = [input_dim] + list(hidden_units)

        # 线性隐藏层
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        # 批量规范层
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        # 激活层
        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        # 归一值
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc

        return deep_input


class PredictionLayer(nn.Module):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        """
        预测层

        :param task: 预测任务，支持 binary|multiclass|regression
        :param use_bias: 是否添加偏差值
        """
        if task not in ['binary', 'multiclass', 'regression']:
            raise ValueError('task 必须为 binary|multiclass|regression')

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == 'binary':
            output = torch.sigmoid(output)
        return output


class Conv2dSame(nn.Conv2d):
    """类 Tensorflow 的二维卷积包装类"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        ih, iw = X.size()[-2:]
        kh, kw = self.weight.size()[-1:]
        oh = math.ceil(ih / self.stride[0])
        ow = math.ceil(iw / self.stride[1])
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(X, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(X, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out
