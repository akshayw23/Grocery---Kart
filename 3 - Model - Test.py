
#del customerXproduct
#del priors

print('training model')
lgb_bst = lgb.train(lgb_params, d_train_lgb, ROUNDS)

# Plot importance of predictors
lgb.plot_importance(lgb_bst)


# Set threshold
# We get the threshold in cross validation
threshold = 0.17


print('predict on test data')
lgb_preds = lgb_bst.predict(df_to_test[features_to_use])

df_to_test['pred'] = lgb_preds

d = dict()
for row in df_to_test.itertuples():
    if row.pred > threshold:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in test.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']

sub.head()
sub.tail()


print('saving csv')
sub.to_csv('lightgbm_0.20(6th)_nodow.csv', index=False)



