from deepctr.inputs import *


def test_features():
    features = [SparseFeat('user_id', vocab_size=100, emb_dim='auto', group_name='user'),
                SparseFeat('age', vocab_size=2, group_name='user'),
                DenseFeat('vision', emb_dim=10),
                SeqFeat(SparseFeat('hist_item_id', vocab_size=50, emb_dim=16),
                        maxlen=100, length_name='seq_length'),
                SeqFeat(SparseFeat('hist_cat_id', vocab_size=50),
                        maxlen=100, length_name='seq_length')]

    fm = FeatureManager(features)
    for feat in fm.features:
        print(feat.name, feat.emb_dim)

    print(fm.names)
    print(fm.sparse_features)
    print(fm.seq_features)
    print(fm.dense_features)

    for name, (start, stop) in fm.index_dict.items():
        print(name, start, stop)
    # X = torch.ones(2, 3, 10, 1, 1)
    # y = features.transform(X)
    # print(y)

    names = get_feature_names(fm.index_dict)
    print(names)
    #
    feature_dict = index_features(features)
    # print(feature_dict)
