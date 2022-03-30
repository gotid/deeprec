from deepctr.inputs import *


def test_features():
    feature_cols = [SparseFeat('user_id', vocabulary_size=100, embedding_dim='auto', group_name='user'),
                    SparseFeat('age', vocabulary_size=2, group_name='user'),
                    DenseFeat('vision', embedding_dim=10),
                    SequenceFeat(SparseFeat('hist_item_id', vocabulary_size=50, embedding_dim=16),
                                 maxlen=100, length_name='seq_length'),
                    SequenceFeat(SparseFeat('hist_cat_id', vocabulary_size=50),
                                 maxlen=100, length_name='seq_length')]

    features = Features(feature_cols)
    for feat in features.columns:
        print(feat.name, feat.embedding_dim)

    print(features.names)
    print(features.sparse_columns)
    print(features.sequence_columns)

    for name, (start, stop) in features.index_dict.items():
        print(name, start, stop)
    # X = torch.ones(2, 3, 10, 1, 1)
    # y = features.transform(X)
    # print(y)

    #
    # names = get_feature_names(feature_cols)
    # print(names)
    #
    # feature_dict = build_feature_index(feature_cols)
    # print(feature_dict)
