from deepmatch.models.basemodel import BaseModel


class DeepFM(BaseModel):
    def __init__(self, linear_features, dnn_features,
                 use_fm=True, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5, l2_reg_emb=1e-6, l2_reg_dnn=0,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 init_std=1e-4, seed=2022, task='binary', device='cpu', gpus=None):
        super(DeepFM, self).__init__(linear_features, dnn_features, l2_reg_linear=l2_reg_linear)
