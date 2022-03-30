import numpy as np
import torch


def concat(tensors, axis=-1):
    if len(tensors) == 1:
        return tensors[0]
    else:
        return torch.cat(tensors, dim=axis)


def slice_arrays(arrays, start=None, stop: int = None):
    """对数组或列表进行切片。

    如果 `array` 是数组返回：arrays[start:stop]
    如果 `array` 是列表返回：[x[start:stop] for x in arrays]

    start 若为列表也支持如下形式：`slice_arrays(x, indices)`

    :param arrays: 单个数组或数组列表
    :param start: 可以是整数索引，也可以是数组索引
    :param stop: 整数，若`start`为列表，`stop`可为空
    :return: 数组切片
    """
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('若 start 为列表，则 stop 必须为 None')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 数据集只支持列表对象作为索引
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]
