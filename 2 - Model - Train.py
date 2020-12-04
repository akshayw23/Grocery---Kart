#--------------Product Feature Creation-------------------------------
products_raw = pd.DataFrame()

#Total Orders Grouped By Product
products_raw['ordersTotal'] = order_products_priorDF.groupby(
    order_products_priorDF.product_id).size().astype(np.int32)

#Total Re-Orders Grouped By Product
products_raw['reordersTotal'] = order_products_priorDF['reordered'].groupby(
    order_products_priorDF.product_id).sum().astype(np.float32)

#Total Re-Orders Rate for each Product
products_raw['reorder_rate'] = (products_raw.reordersTotal / products_raw.ordersTotal).astype(np.float32)

products = productsDF.join(products_raw, on = 'product_id')

del products_raw

products.head()


#-------------- Merging dataframes------------------------------
priors = pd.merge(order_products_priorDF, ordersDF, how = 'inner', on = 'order_id')
ordersDF.set_index('order_id', inplace = True, drop = False)


#--------------Product and Users-------------------------------
users = pd.DataFrame()
users['total_user'] = priors.groupby('product_id').size()
users['all_users'] = priors.groupby('product_id')['user_id'].apply(set)
users['total_distinct_users_perProduct'] = users.all_users.map(len)


users.head()
priors.head()
ordersDF.info()

#--------------Customer Feature Creation-------------------------------
# customers: total_items, all_products, total_unique_items,
# avgDaysBetwOrders, NumberOfOrders, avg_per_cart
customers_raw = pd.DataFrame()
customers_raw['avgDaysBetwOrders'] = ordersDF.groupby(
    'user_id')['days_since_prior_order'].mean().astype(np.float32)

customers_raw['NumberOfOrders'] = ordersDF.groupby('user_id').size().astype(np.int16)

customers = pd.DataFrame()

customers['total_items'] = priors.groupby('user_id').size().astype(np.int16)
customers['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
customers['total_unique_items'] = customers.all_products.map(len).astype(np.float32)

customers = customers.join(customers_raw)
customers['avg_per_cart'] = (customers.total_items / customers.NumberOfOrders).astype(np.float32)

del customers_raw
customers.head()


customerXproduct = priors.copy()
customerXproduct['user_product'] = (customerXproduct.product_id + 
                                    customerXproduct.user_id * 100000).astype(np.int64)

customerXproduct = customerXproduct.sort_values('order_number')

customerXproduct = customerXproduct.groupby('user_product', sort = False).agg(
{'order_id': ['size', 'last', 'first'], 'add_to_cart_order': 'sum'})

customerXproduct.columns = ['numbOfOrders', 'last_order_id', 'first_order_id','sum_add_to_cart_order']
customerXproduct = customerXproduct.astype(
    {'numbOfOrders': np.int16, 'last_order_id': np.int32, 'first_order_id': np.int32, 
     'sum_add_to_cart_order': np.int16})


customerXproduct.head()

priors.head()
priors.info()

def get_features(specified_orders,order_product, given_labels = False):
    print('create initial empty list')
    orders_list = []
    products_list = []
    labels = []
    
    training_index = set(order_product.index)
    
    for row in specified_orders.itertuples():
        user_id = row.user_id
        order_id = row.order_id
        
        user_products = customers['all_products'][user_id]
        products_list += user_products
        orders_list += [order_id] * len(user_products)
        
        if given_labels:
            labels += [(order_id, product) in training_index for product in user_products]
        
    DF = pd.DataFrame({'order_id': orders_list, 'product_id': products_list}, dtype = np.int32)
    labels = np.array(labels, dtype = np.int8)
        
    print('get features for user part')
    DF['user_id'] = DF.order_id.map(ordersDF.user_id)
    DF['user_total_orders'] = DF.user_id.map(customers.NumberOfOrders)
    DF['user_total_items'] = DF.user_id.map(customers.total_items)
    DF['total_unique_items'] = DF.user_id.map(customers.total_unique_items)
    DF['user_avgDaysBetwOrders'] = DF.user_id.map(customers.avgDaysBetwOrders)
    DF['user_avg_per_cart'] = DF.user_id.map(customers.avg_per_cart) 
        
    print('get features for order part')
    DF['order_hour_of_day'] = DF.order_id.map(ordersDF.order_hour_of_day)
    DF['days_since_prior_order'] = DF.order_id.map(ordersDF.days_since_prior_order)
    DF['daysSincePrior_avgDaysBetw_ratio'] = DF.days_since_prior_order / DF.user_avgDaysBetwOrders
        
    print('get features for product part')
    DF['aisle_id'] = DF.product_id.map(products.aisle_id)
    DF['department_id'] = DF.product_id.map(products.department_id)
    DF['product_order'] = DF.product_id.map(products.ordersTotal)
    DF['product_reorder'] = DF.product_id.map(products.reordersTotal)
    DF['product_reorder_rate'] = DF.product_id.map(products.reorder_rate)
    DF['product_distinct_user'] = DF.product_id.map(users.total_distinct_users_perProduct)
    
    print('get features for customerXproduct')
    DF['user_product_id']  = (DF.product_id + DF.user_id * 100000).astype(np.int64)
    DF.drop(['user_id'], axis = 1, inplace = True)
    DF['CP_numOrders'] = DF.user_product_id.map(customerXproduct.numbOfOrders)
    DF['CP_orders_ratio'] = DF.CP_numOrders / DF.user_total_orders
    DF['CP_last_order_id'] = DF.user_product_id.map(customerXproduct.last_order_id)
    DF['CP_avg_pos_inCart'] = DF.user_product_id.map(customerXproduct.sum_add_to_cart_order) / DF.CP_numOrders
    DF['CP_order_since_last'] = DF.user_total_orders - DF.CP_last_order_id.map(ordersDF.order_number)
    DF['CP_hour_vs_last'] = abs(DF.order_hour_of_day - DF.CP_last_order_id.map(
    ordersDF.order_hour_of_day)).map(lambda x: min(x, 24 - x)).astype(np.int8)

    DF.drop(['CP_last_order_id', 'user_product_id'], axis = 1, inplace = True)
    return(DF, labels)
    
ordersDF.info()
ordersDF.shape


import lightgbm as lgb

test = ordersDF[ordersDF.eval_set == 'test']
train = ordersDF[ordersDF.eval_set == 'train']

order_products_trainDF.set_index(['order_id', 'product_id'], inplace = True, drop = False)


# select features to use for training
features_to_use = ['user_total_orders', 'user_total_items', 'total_unique_items', 
                  'user_avgDaysBetwOrders', 'user_avg_per_cart', 'order_hour_of_day',
                  'days_since_prior_order', 'daysSincePrior_avgDaysBetw_ratio',
                  'aisle_id', 'department_id', 'product_order', 'product_reorder',
                  'product_reorder_rate', 'CP_numOrders', 'CP_orders_ratio', 
                  'CP_avg_pos_inCart', 'CP_order_since_last', 'CP_hour_vs_last',
                  'product_distinct_user'] #'dow'

train_train = train.sample(frac = 0.8, random_state=200)
train_test = train.drop(train_train.index)

def eval_fun(labels, preds):
    labels = labels.split(' ')
    preds = preds.split(' ')
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

df_train_train, labels_train_train = get_features(train_train, order_products_trainDF, given_labels=True)

df_train_test, labels_train_test = get_features(train_test, order_products_trainDF, given_labels=True)

df_to_test, _ = get_features(test, order_products_testDF, given_labels=True)


# Threshold
threshold = np.linspace(0.14, 0.30, num=17)

f1_score_values = []
precision_values = []
recall_values = []

    
d_train_lgb = lgb.Dataset(df_train_train[features_to_use],
                          label = labels_train_train,
                          categorical_feature = ['aisle_id', 'department_id'], free_raw_data = False)

ROUNDS = 80

for n in threshold:
    print(n)
    lgb_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 96,
        'max_depth': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5
    }

    print('model training')
    lgb_bst = lgb.train(lgb_params, d_train_lgb, ROUNDS)

    print('predict on the test set')
    lgb_preds = lgb_bst.predict(df_train_test[features_to_use])

    df_train_test_copy = df_train_test
    df_train_test_copy['pred'] = lgb_preds

    d = dict()
    for row in df_train_test_copy.itertuples():
        if row.pred > n:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in train_test.order_id:
        if order not in d:
            d[order] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')

    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']

    df_train_test_copy['true'] = labels_train_test

    e = dict()
    for row in df_train_test_copy.itertuples():
        if row.true == 1:
            try:
                e[row.order_id] += ' ' + str(row.product_id)
            except:
                e[row.order_id] = str(row.product_id)

    for order in train_test.order_id:
        if order not in e:
            d[order] = 'None'

    sub_true = pd.DataFrame.from_dict(e, orient='index')

    sub_true.reset_index(inplace=True)

    sub_true.columns = ['order_id', 'true']
    
    print('merge result')
    sub_merge = pd.merge(sub_true, sub, how = 'inner', on = 'order_id')

    res = list()
    for entry in sub_merge.itertuples():
        res.append(eval_fun(entry[2], entry[3]))

    res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])
    
    print('append f1 score')
    f1_score_values.append(np.mean(res['f1']))
    precision_values.append(np.mean(res['precision']))
    recall_values.append(np.mean(res['recall']))

import matplotlib.pyplot as plt 
# Plot F1 Score vs Thresholds
plt.plot(threshold,f1_score_values)
plt.ylabel('F1 Score')
plt.xlabel('Threshold')
plt.axvline(x = 0.17,color='darkorange', linestyle='dashed', linewidth=1)
plt.title('F1 Score vs Thresholds')
plt.savefig('f1vsThresholds.png', dpi = 800, figsize = (1,1))


#df_index = pd.DataFrame(list(zip(threshold,f1_score_values,precision_values,recall_values)),
#                      columns=['threshold','f1', 'precision', 'recall'])


------------------------------------------------------------
