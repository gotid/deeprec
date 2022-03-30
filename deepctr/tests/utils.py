import numpy as np
import torch


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=torch.float32, input_data=None,
               expected_output=None, expected_output_shape=None, expected_output_dtype=None, fixed_batch_size=False):
    """检查指定层是否有效"""

    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    assert expected_output_dtype, f"必须指定期待层输出的形状: expected_output_dtype"

    # 生成默认随机输入数据
    if input_data is None:
        # 生成输入数据
        if not input_shape:
            raise ValueError("输入形状不可为空")

        input_data_shape = list(input_shape)

        # 给空形状赋随机值
        for i, shape in enumerate(input_data_shape):
            if shape is None:
                input_data_shape[i] = np.random.randint(1, 4)

        if all(isinstance(shape, tuple) for shape in input_data_shape):
            input_data = []
            for shape in input_data_shape:
                rand_input = (10 * np.random.random(shape))
                input_data.append(rand_input)
        else:
            rand_input = 10 * np.random.random(input_data_shape)
            input_data = rand_input

    # 初始化层
    layer = layer_cls(**kwargs)

    # 计算层输出
    inputs = torch.tensor(input_data, dtype=input_dtype)
    output = layer(inputs)

    # 验证输出的 dtype 和 shape 是否一致
    assert output.dtype == expected_output_dtype, f"期待层输出的数据类型: {expected_output_dtype}，得到: {output.dtype}"
    got_output_shape = output.shape
    for expected_shape, got_shape in zip(expected_output_shape, got_output_shape):
        if expected_shape is not None:
            assert expected_shape == got_shape, f"期待层输出的形状: {expected_shape}，得到: {got_shape}"

    # 验证输出的结果是否一致
    if expected_output is not None:
        pass  # TODO 验证层输出的结果是否一致

    return output
