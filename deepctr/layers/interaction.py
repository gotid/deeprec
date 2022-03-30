import torch
import torch.nn as nn


class FM(nn.Module):
    """分解机模型(Factorization Machine)，无线性项和偏置的成对特征交互。

    参考：[Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        """
        求交叉项。

        :param inputs: 三维张量 `(batch_size, field_size, emb_size)`
        :return: 二维张量 `(batch_size, 1)`
        """
        fm_inputs = inputs

        square_of_sum = torch.pow(torch.sum(fm_inputs, dim=1, keepdim=True), 2)  # 和的平方
        sum_of_square = torch.sum(fm_inputs * fm_inputs, dim=1, keepdim=True)  # 平方和
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class CIN(nn.Module):
    """压缩后的交互网络，用于 xDeepFM"""
