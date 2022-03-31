from deepctr.inputs import SparseFeat
from deepctr.models.basemodel import BaseModel

if __name__ == '__main__':
    linear_features = [SparseFeat('user_id', 10, emb_dim=4)]
    dnn_features = linear_features
    model = BaseModel(linear_features, dnn_features)
    feature_names = [feature for feature in model.feature_idx_dict]
    print(feature_names)
