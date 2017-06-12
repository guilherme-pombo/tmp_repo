import pandas as pd
import numpy as np
import gc


def split_train_val(df, labels, val_size=0.05):
    """
    Split and labels into train and validation sets
    :return:
    """
    msk = np.random.rand(len(df)) < 1 - val_size
    # train, train_labels, val, val_labels
    return df[msk], labels[msk], df[~msk], labels[~msk]


class DataPipeline:
    
    def __init__(self, priors_path: str, train_path: str, orders_path: str, product_path: str):
        print('loading prior')
        self.priors = pd.read_csv(priors_path)
        print('loading train')
        self.train = pd.read_csv(train_path)
        print('loading orders')
        self.orders = pd.read_csv(orders_path)
        print('loading products')
        self.products = pd.read_csv(product_path)

        print('priors {}: {}'.format(self.priors.shape, ', '.join(self.priors.columns)))
        print('orders {}: {}'.format(self.orders.shape, ', '.join(self.orders.columns)))
        print('train {}: {}'.format(self.train.shape, ', '.join(self.train.columns)))

        # Reduce memory usage
        self.optimise_memory()

        self.users = None
        self.userXproduct = None

        # get orders, reorders, reorder rate
        self.compute_product_order_info()
        # calculate user info
        self.calculate_user()
        #
        self.compute_user_product()
        # Create dataset
        self.create_dataset()
        
    def optimise_memory(self):
        """
        Use this method to reduce method utilisation by casting dataframes to use only types necessary to
        capture the complexity of the data
        :return:
        """
        print('optimizing memory usages')
        self.orders.order_dow = self.orders.order_dow.astype(np.int8)
        self.orders.order_hour_of_day = self.orders.order_hour_of_day.astype(np.int8)
        self.orders.order_number = self.orders.order_number.astype(np.int16)
        self.orders.order_id = self.orders.order_id.astype(np.int32)
        self.orders.user_id = self.orders.user_id.astype(np.int32)
        self.orders.days_since_prior_order = self.orders.days_since_prior_order.astype(np.float32)

        self.products.drop(['product_name'], axis=1, inplace=True)
        self.products.aisle_id = self.products.aisle_id.astype(np.int8)
        self.products.department_id = self.products.department_id.astype(np.int8)
        self.products.product_id = self.products.product_id.astype(np.int32)

        self.train.reordered = self.train.reordered.astype(np.int8)
        self.train.add_to_cart_order = self.train.add_to_cart_order.astype(np.int16)

        self.priors.order_id = self.priors.order_id.astype(np.int32)
        self.priors.add_to_cart_order = self.priors.add_to_cart_order.astype(np.int16)
        self.priors.reordered = self.priors.reordered.astype(np.int8)
        self.priors.product_id = self.priors.product_id.astype(np.int32)

    def compute_product_order_info(self):
        print('computing product f')
        prods = pd.DataFrame()
        prods['orders'] = self.priors.groupby(self.priors.product_id).size().astype(np.float32)
        prods['reorders'] = self.priors['reordered'].groupby(self.priors.product_id).sum().astype(np.float32)
        prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
        self.products = self.products.join(prods, on='product_id')
        self.products.set_index('product_id', drop=False, inplace=True)
        del prods

        print('add order info to priors')
        self.orders.set_index('order_id', inplace=True, drop=False)
        self.priors = self.priors.join(self.orders, on='order_id', rsuffix='_')
        self.priors.drop('order_id_', inplace=True, axis=1)

    def calculate_user(self):
        print('computing user f')
        usr = pd.DataFrame()
        usr['average_days_between_orders'] = self.orders.groupby('user_id')['days_since_prior_order'].mean().astype(
            np.float32)
        usr['nb_orders'] = self.orders.groupby('user_id').size().astype(np.int16)

        users = pd.DataFrame()
        users['total_items'] = self.priors.groupby('user_id').size().astype(np.int16)
        users['all_products'] = self.priors.groupby('user_id')['product_id'].apply(set)
        users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

        self.users = users.join(usr)
        del usr
        self.users['average_basket'] = (self.users.total_items / self.users.nb_orders).astype(np.float32)
        gc.collect()
        print('user f', users.shape)

    def compute_user_product(self):
        print('compute userXproduct f - this is long...')
        self.priors['user_product'] = self.priors.product_id + self.priors.user_id * 100000

    def create_dataset(self):
        d = dict()
        for row in self.priors.itertuples():
            z = row.user_product
            if z not in d:
                d[z] = (1,
                        (row.order_number, row.order_id),
                        row.add_to_cart_order)
            else:
                d[z] = (d[z][0] + 1,
                        max(d[z][1], (row.order_number, row.order_id)),
                        d[z][2] + row.add_to_cart_order)

        print('to dataframe (less memory)')
        d = pd.DataFrame.from_dict(d, orient='index')
        d.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
        d.nb_orders = d.nb_orders.astype(np.int16)
        d.last_order_id = d.last_order_id.map(lambda x: x[1]).astype(np.int32)
        d.sum_pos_in_cart = d.sum_pos_in_cart.astype(np.int16)

        self.userXproduct = d
        print('user X product f', len(self.userXproduct))

    def get_test_and_train(self):
        print('split orders : train, test')
        test_orders = self.orders[self.orders.eval_set == 'test']
        train_orders = self.orders[self.orders.eval_set == 'train']

        self.train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

        return train_orders, test_orders

    def create_feature_labelled_data(self, selected_orders, labels_given=False):
        print('build candidate list')
        order_list = []
        product_list = []
        labels = []
        i = 0
        for row in selected_orders.itertuples():
            i += 1
            if i % 10000 == 0:
                print('order row', i)
            order_id = row.order_id
            user_id = row.user_id
            user_products = self.users.all_products[user_id]
            product_list += user_products
            order_list += [order_id] * len(user_products)
            if labels_given:
                labels += [(order_id, product) in self.train.index for product in user_products]

        df = pd.DataFrame({'order_id': order_list, 'product_id': product_list})
        # reduce used memory
        df.order_id = df.order_id.astype(np.int32)
        df.product_id = df.product_id.astype(np.int32)
        labels = np.array(labels, dtype=np.int8)

        print('user related features')
        df['user_id'] = df.order_id.map(self.orders.user_id).astype(np.int32)
        df['user_total_orders'] = df.user_id.map(self.users.nb_orders)
        df['user_total_items'] = df.user_id.map(self.users.total_items)
        df['total_distinct_items'] = df.user_id.map(self.users.total_distinct_items)
        df['user_average_days_between_orders'] = df.user_id.map(self.users.average_days_between_orders)
        df['user_average_basket'] = df.user_id.map(self.users.average_basket)

        print('order related features')
        # df['dow'] = df.order_id.map(orders.order_dow)
        df['order_hour_of_day'] = df.order_id.map(self.orders.order_hour_of_day)
        df['days_since_prior_order'] = df.order_id.map(self.orders.days_since_prior_order)
        df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

        print('product related features')
        df['aisle_id'] = df.product_id.map(self.products.aisle_id).astype(np.int8)
        df['department_id'] = df.product_id.map(self.products.department_id).astype(np.int8)
        df['product_orders'] = df.product_id.map(self.products.orders).astype(np.float32)
        df['product_reorders'] = df.product_id.map(self.products.reorders).astype(np.float32)
        df['product_reorder_rate'] = df.product_id.map(self.products.reorder_rate)

        print('user_X_product related features')
        df['z'] = df.user_id * 100000 + df.product_id
        df.drop(['user_id'], axis=1, inplace=True)
        df['UP_orders'] = df.z.map(self.userXproduct.nb_orders)
        df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_last_order_id'] = df.z.map(self.userXproduct.last_order_id)
        df['UP_average_pos_in_cart'] = (df.z.map(self.userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
        df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(self.orders.order_number)
        df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - \
                                          df.UP_last_order_id.map(self.orders.order_hour_of_day)).map(
            lambda x: min(x, 24 - x)).astype(np.int8)
        # df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
        #                                              df.order_id.map(orders.order_dow)

        df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
        print(df.dtypes)
        print(df.memory_usage())
        print(self.train.memory_usage())
        print(self.products.memory_usage())
        gc.collect()

        return df, labels
