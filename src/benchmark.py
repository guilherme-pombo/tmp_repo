import pandas as pd
from utils import DataPipeline
from model import train_boosted_tree, filter_by_confidence


# Use this to get all the training data

data_pipe = DataPipeline(priors_path='../input/order_products__prior.csv',
                         train_path='../input/order_products__train.csv',
                         orders_path='../input/orders.csv',
                         product_path='../input/products.csv')

# train and test ids
train_orders, test_orders = data_pipe.get_test_and_train()
# get labelled training data
df_train, labels = data_pipe.create_feature_labelled_data(train_orders, labels_given=True)

features_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                   'user_average_days_between_orders', 'user_average_basket',
                   'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                   'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                   'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                   'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                   'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'

# train the model
bst = train_boosted_tree(df_train=df_train,
                         features_to_use=features_to_use,
                         labels=labels,
                         val_size=0.05,
                         num_epochs=150)

# Create testing data
df_test, _ = data_pipe.create_feature_labelled_data(test_orders, labels_given=False)

# Creating predicitions
print('light GBM predict')
preds = bst.predict(df_test[features_to_use], num_iteration=bst.best_iteration)
df_test['pred'] = preds

# Filter by confidence
confidence_filter = filter_by_confidence(df_test=df_test,
                                         test_orders=test_orders,
                                         threshold=0.22)

# Submission
submission = pd.DataFrame.from_dict(confidence_filter, orient='index')

print('Saving submission to filename: submission.csv')
submission.reset_index(inplace=True)
submission.columns = ['order_id', 'products']
submission.to_csv('submission_best_iter_{}.csv'.format(bst.best_iteration), index=False)
