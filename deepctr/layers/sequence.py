import torch
import torch.nn as nn
from torch import Tensor


class SequencePoolingLayer(nn.Module):
    """用于变长序列特征或多值特征的池化操作(sum, mean, max)模块。

    输入形状：
        - 一个包含两个张量的列表 [seq_value, seq_list]

        - seq_value 是一个三维张量：`(batch_size, T, embedding_size)`

        - seq_len 是一个二维张量  ：`(batch_size, 1)`，表示每个序列的有效长度

    输出形状：
        - 三维张量：`(batch_size, 1, embedding_size)`
    """

    def __init__(self, mode='mean', support_masking=False, device='cpu'):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('mode 必须为 [sum, mean, max]')
        self.support_masking = support_masking
        self.mode = mode
        self.device = device
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)

    @classmethod
    def _sequence_mask(cls, length, maxlen=None, dtype=torch.bool):
        """返回一个表示每个单元格前N个位置的掩码张量"""
        if maxlen is None:
            maxlen = length.max()
        row_vector = torch.arange(0, maxlen, 1).to(length.device)
        matrix = torch.unsqueeze(length, dim=1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list) -> Tensor:
        if self.support_masking:
            seq_emb_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            use_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            seq_emb_list, use_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            mask = self._sequence_mask(use_behavior_length, maxlen=seq_emb_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        emb_size = seq_emb_list.shape[-1]

        mask = torch.repeat_interleave(mask, emb_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = seq_emb_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist

        hist = seq_emb_list * mask.float()
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            self.eps = self.eps.to(use_behavior_length.device)
            hist = torch.div(hist, use_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist
