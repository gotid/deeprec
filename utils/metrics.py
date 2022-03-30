import logging
from typing import List, Dict, Union

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


class MetricMonitor(object):
    """
    验证集指标监控器。

    用于获取每x轮或每x批训练完成后，返回验证集评估的总指标值。
    """

    def __init__(self, kv: Union[str, Dict]):
        """kv 如 AUC"""
        if isinstance(kv, str):
            kv = {kv: 1}
        self.kv_pair = kv

    def get_value(self, logs):
        """获取验证集评估的总指标值"""
        value = 0
        for k, v in self.kv_pair.items():
            value += logs[k] * v
        return value

    @staticmethod
    def evaluate(y_true, y_pred, metrics: List, **kwargs) -> Dict:
        """
        评估每x轮或每x批训练后，验证集的准确度指标。

        :param y_true: 真实 y 值
        :param y_pred: 预测 y 值
        :param metrics: 评估的指标列表，支持 logloss|binary_crossentropy, AUC, ACC
        :param kwargs: 其他参数
        :return: 指标分值
        """
        result = dict()
        for metric in metrics:
            if metric in ['logloss', 'binary_crossentropy']:
                result[metric] = log_loss(y_true, y_pred, eps=1e-7)
            elif metric == 'AUC':
                result[metric] = roc_auc_score(y_true, y_pred)
            elif metric == 'ACC':
                y_pred = np.argmax(y_pred, axis=1)
                result[metric] = accuracy_score(y_true, y_pred)
            else:
                logging.warning(f"跳过 {metric} - 该评估指标暂不支持")

        logging.info('[Metrics 指标] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in result.items()))
        return result
