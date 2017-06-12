import lightgbm as lgb
import gc

from utils import split_train_val


def train_boosted_tree(df_train, features_to_use, labels, val_size=None, num_epochs=100):

    print('formating for lgb')
    if val_size:
        print("Using a validation set for early stopping")
        df_train, labels, df_val, val_labels = split_train_val(df=df_train, labels=labels, val_size=val_size)
        d_val = lgb.Dataset(df_val[features_to_use],
                            label=val_labels,
                            categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'

    d_train = lgb.Dataset(df_train[features_to_use],
                          label=labels,
                          categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'

    gc.collect()

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 96,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5
    }

    print('light GBM train :-)')
    if val_size:
        bst = lgb.train(params, d_train, num_epochs, valid_sets=[d_val], early_stopping_rounds=10)
    else:
        bst = lgb.train(params, d_train, num_epochs)

    return bst


def filter_by_confidence(df_test, test_orders, threshold=0.25):
    """
    Filter submissions by a confidence threshold
    :param df_test:
    :param test_orders:
    :param threshold:
    :return:
    """
    d = dict()
    for row in df_test.itertuples():
        if row.pred > threshold:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in test_orders.order_id:
        if order not in d:
            d[order] = 'None'

    return d
