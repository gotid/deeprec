import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from deepctr.inputs import SparseFeat, build_feature_index
from deepctr.layers import DNN, FM
from deepctr.models.basemodel import Linear


def test_dnn():
    model = DNN(10, [64, 32])

    X = torch.arange(10, dtype=torch.float)
    print(X)
    y = model.forward(X)
    print(y)


def test_loader():
    df = pd.DataFrame([[1, 3, 6, 1],
                       [2, 4, 7, 1],
                       [1, 3, 8, 0]],
                      columns=['user_id', 'age', 'media_id', 'click'])
    x, y = df.iloc[:, 0:3].values, df.iloc[:, -1].values
    print(x, y)
    # np.savez('x', x)
    # with h5py.File('x.h5', 'w') as hf:
    #     hf.create_dataset('x', data=x)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    dataset = TensorDataset(x, y)
    for v in dataset:
        print(v)


def test_input():
    x = {'user_id': np.arange(3),
         'age': np.arange(3, 6)}
    y = np.arange(6, 9)
    print(x)
    # 取字典值数组
    if isinstance(x, dict):
        x = list(x.values())
    print(x)
    # 扩维：1维转2维
    for i in range(len(x)):
        if x[i].ndim == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    print(x)
    # 拼接
    x = np.concatenate(x, axis=-1)
    print(x)
    # 转型
    x = torch.from_numpy(x)
    y = torch.tensor(y, dtype=torch.float32)
    print(x, y)


def test_fm():
    fm = FM()
    print(fm.forward(torch.from_numpy(np.random.rand(2, 3, 2))))


def test_cat():
    x = torch.ones(2, 3)
    y = torch.ones(2, 3)
    print()
    print(x)
    print(y)
    print(torch.cat((x, y), dim=0))  # 竖向拼
    print(torch.cat((x, y), dim=1))  # 横向拼


def test_linear():
    feature_columns = [SparseFeat('user_id', vocabulary_size=10, embedding_dim=4, group_name='user')]
    linear = Linear(feature_columns, init_std=1e-4)
    inputs = torch.rand(2, 1)

    print('inputs', inputs.detach().numpy().flatten())
    print('inputs[0][0]', inputs[0][0].long())
    # print(list(linear.parameters()))
    print(linear.forward(inputs))


def test_unsqueeze():
    x = torch.ones(2, 1)
    print()
    print(x)
    print(x.squeeze())


if __name__ == '__main__':
    test_linear()
