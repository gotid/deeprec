from deepctr.inputs import *


def get_xy_fd(use_neg=True):
    feature_columns = [SparseFeat('user', 4, embedding_dim=4),
                       SparseFeat('gender', 2, embedding_dim=4),
                       SparseFeat('item_id', 3 + 1, embedding_dim=8),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4),
                       DenseFeat('pay_score', 1),
                       SequenceFeat(SparseFeat('hist_item_id', 3 + 1, embedding_dim=8, embedding_name='item_id'),
                                    maxlen=4, length_name='seq_length'),
                       SequenceFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                                    maxlen=4, length_name='seq_length')]

    uid = np.array([0, 1, 2, 3])
    gender = np.array([0, 1, 0, 1])
    item_id = np.array([1, 2, 3, 2])  # 0 为掩码
    cate_id = np.array([1, 2, 1, 2])  # 0 为掩码
    pay_score = np.array([0.1, 0.2, 0.3, 0.2])

    hist_item_id = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
    hist_cate_id = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0], [1, 2, 0, 0]])

    behavior_length = np.array([3, 3, 2, 2])

    feature_dict = {'user': uid, 'gender': gender, 'item_id': item_id, 'cate_id': cate_id,
                    'hist_item_id': hist_item_id, 'hist_cate_id': hist_cate_id,
                    'pay_score': pay_score, 'seq_length': behavior_length}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 1, 0], [1, 2, 0, 0]])
        feature_columns += [
            SequenceFeat(SparseFeat('neg_hist_item_id', 3 + 1, embedding_dim=8, embedding_name='item_id'),
                         maxlen=4, length_name='seq_length'),
            SequenceFeat(SparseFeat('neg_hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                         maxlen=4, length_name='seq_length')]

    features = Features(feature_columns)

    x = {name: feature_dict[name] for name in features.names}
    y = np.array([1, 0, 1, 0])

    behavior_feature_list = ['item_id', 'cate_id']

    return x, y, features, behavior_feature_list


if __name__ == '__main__':
    x, y, features, behavior_feature_list = get_xy_fd(use_neg=True)

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    print(device)
