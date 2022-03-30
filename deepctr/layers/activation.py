import torch
from torch import nn, Tensor


class Dice(nn.Module):
    """
    深度兴趣网络(DIN)中的数据自适应激活函数，可以视为PReLu的泛化，可以根据输入数据的分布自适应调整校正点。

    输出形状：和输入一致。

    参考
        - https://github.com/zhougr1993/DeepInterestNetwork
        - https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, num_features: int, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        # 对二维或三维输入批量规格化
        self.bn = nn.BatchNorm1d(num_features, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # 包装 alpha 以使其可被训练
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((num_features,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((num_features, 1)).to(device))

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(inputs))
            out = self.alpha * (1 - x_p) * inputs + x_p * inputs
        else:
            # 矩阵转置
            inputs = torch.transpose(inputs, 1, 2)
            x_p = self.sigmoid(self.bn(inputs))
            out = self.alpha * (1 - x_p) * inputs + x_p * inputs
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, num_features: int = None, dice_dim=2):
    """
    构建激活层

    :param act_name: 激活函数名称或 nn.Module
    :param num_features: 用于 Dice 激活
    :param dice_dim: 用于 Dice 激活，2维|3维
    :return: 激活层
    """
    act_layer = None

    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(num_features, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


if __name__ == '__main__':
    X = torch.randn(1, 2)
    print('X', X)
    print('sigmoid', activation_layer('sigmoid')(X))
    print('linear', activation_layer('linear')(X))
    print('relu', activation_layer('relu')(X))
    print('dice', activation_layer('dice', 3)(torch.randn(2, 3)))
    print('prelu', activation_layer('prelu')(X))
    print('dropout', activation_layer(nn.Dropout)(X))
