import pandas as pd
import numpy as np


# Set data directory
dir = 'F:\\Study\\Imarticus\\Capstone\\Data\\'
dir ="C:\\Users\\Akshay\\Desktop\\Capstone\\Project and Data Sets\\Data\\"
#---------------Loading Datasets -----------------------

aislesDF = pd.read_csv(dir + 'aisles.csv')	
departmentsDF = pd.read_csv(dir + 'departments.csv')
productsDF = pd.read_csv(dir + 'products.csv')

order_products_priorDF = pd.read_csv(dir + 'order_products_prior.csv')
order_products_trainDF = pd.read_csv(dir + 'order_products_train.csv')
order_products_testDF = pd.read_csv(dir + 'order_products_test.csv')
ordersDF = pd.read_csv(dir + 'orders.csv')


#---------------Checking Dataframes structure and missing values-----------------------

# Aisles Dataframe
aislesDF.head()
aislesDF.tail()
aislesDF.info()
aislesDF.shape
aislesDF.isnull().sum()


# Departments Dataframe
departmentsDF.head()
departmentsDF.tail()
departmentsDF.info()
departmentsDF.shape
departmentsDF.isnull().sum()


# Products Dataframe
productsDF.head()
productsDF.tail()
productsDF.info()
productsDF.shape
productsDF.isnull().sum()


# Order Products Prior Dataframe
order_products_priorDF.head()
order_products_priorDF.tail()
order_products_priorDF.info()
order_products_priorDF.shape
order_products_priorDF.isnull().sum()


# Order Products Train Dataframe
order_products_trainDF.head()
order_products_trainDF.tail()
order_products_trainDF.info()
order_products_trainDF.shape
order_products_trainDF.isnull().sum()


# Order Products Test Dataframe
order_products_testDF.head()
order_products_testDF.tail()
order_products_testDF.info()
order_products_testDF.shape
order_products_testDF.isnull().sum()



# Orders Dataframe
ordersDF.head()
ordersDF.tail()
ordersDF.info()
ordersDF.shape
ordersDF.isnull().sum()



#---------------Pre-Processing -----------------------

# Removing 'Unnamed' column
order_products_testDF = order_products_testDF.drop(order_products_testDF.columns[0], axis=1)
order_products_trainDF = order_products_trainDF.drop(order_products_trainDF.columns[0], axis=1)
ordersDF = ordersDF.drop(ordersDF.columns[0], axis=1)

#----------------------------------------------
ordersDF.isnull().sum()
#Replace na with 0 - First Order
ordersDF['days_since_prior_order'].fillna(0, inplace=True)
ordersDF.isnull().sum()

#----------------------------------------------

order_incorrect_dow = ordersDF.query('order_dow>6')
orders_incorrect_dow_inner = ordersDF.merge(order_incorrect_dow,how='inner',on='order_id')
orders_incorrect_dow_inner.info()

orders_incorrect_dow_inner[['order_id','user_id_x','user_id_x','order_number_x','order_number_y']]
orders_incorrect_dow_inner[['order_id','order_dow_x','order_dow_x','order_number_x','order_number_y']]

#Delete rows with incorrect DoW  
ordersDF.shape
ordersDF = ordersDF.drop(ordersDF.query('order_dow>6').index,axis=0)
ordersDF.shape

# Checking values in Day's of Week 
ordersDF['order_dow'].unique()

# Delete Data Frame's
del orders_incorrect_dow_inner
del order_incorrect_dow


#----------------------------------------------

#order_number column have negative values
negativeDF = ordersDF.loc[ordersDF['order_number'] < 0]
negativeDF[['order_number','order_dow','order_id','user_id']]

orders_incorrect_ordernumber_inner = ordersDF.merge(negativeDF,how='inner',on='order_id')
orders_incorrect_ordernumber_inner.info()
orders_incorrect_ordernumber_inner[['order_id','user_id_x','user_id_y','order_number_x','order_number_y']]
orders_incorrect_ordernumber_inner[['order_id','order_dow_x','order_dow_y','order_number_x','order_number_y']]

#order_number column have negative values
ordersDF = ordersDF.drop(ordersDF.query('order_number<0').index,axis=0)
ordersDF.shape

ordersDF.loc[ordersDF['order_number'] < 0]

del negativeDF
del orders_incorrect_ordernumber_inner


#-------------------Customer's Order Day of the week ------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

reorder_dow_freq = ordersDF['order_dow'].value_counts().sort_index()
reorder_dow_freq.index=['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri']
f, ax = plt.subplots(figsize=(10, 10))
sns.barplot(reorder_dow_freq.index, reorder_dow_freq.values)
plt.xticks(rotation=0)
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Days of order in a week', fontsize=13)
plt.title('Order Frequency in a Week', fontsize=13)
plt.show()

del reorder_dow_freq

#-------------------Customer's Order Time of the day ------------------------------------
reorder_hour_of_day_freq = ordersDF['order_hour_of_day'].value_counts().sort_index()
f, ax = plt.subplots(figsize=(10, 10))
sns.barplot(reorder_hour_of_day_freq.index, reorder_hour_of_day_freq.values)
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Hours', fontsize=13)
plt.title('Order Frequency in Hour of Day', fontsize=13)
plt.show()

del reorder_hour_of_day_freq


#-------------------Customer's Order Day and Time of the day ------------------------------------

grouped_df = ordersDF.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
grouped_df.index=['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri']
plt.figure(figsize=(12,6))
sns.heatmap(grouped_df, linewidths=.5, cmap="YlGnBu")
plt.ylabel('Number of orders', fontsize=13)
plt.xlabel('Hours', fontsize=13)
plt.title("Frequency of Day of Week Vs Hour of Day")
plt.show()

del grouped_df

#-------------------Top 5 - Ordered Products------------------------------------

# orders in prior with product names
Order_Product_Name_Prior = pd.merge(order_products_priorDF[['order_id','product_id','reordered']], 
                                    productsDF[['product_name','product_id']],
                                    how='left', on='product_id')

# Prior orders with user_id, product_id, product_name
Prior_User_Order_Product = pd.merge(Order_Product_Name_Prior, 
                                    ordersDF[['order_id']], 
                                    how='left', on='order_id')


#Plotting Graph
Products_Count = Prior_User_Order_Product['product_name'].value_counts().head(n=5)
f, ax = plt.subplots(figsize=(10, 10))

splot=sns.barplot(x=Products_Count.index,y=Products_Count.values)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
plt.xticks(rotation=90)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Top 5 Ordered Products', fontsize=13)
plt.title('Top 5 Popular Products')
plt.show()

del Order_Product_Name_Prior
del Prior_User_Order_Product
del Products_Count

#-------------------Top 5 - Re-Ordered Products------------------------------------

# orders in prior with product names
Order_Product_Name_Prior = pd.merge(order_products_priorDF[['order_id','product_id','reordered']], 
                                    productsDF[['product_name','product_id']],
                                    how='left', on='product_id').query('reordered == 1')

# Prior orders with user_id, product_id, product_name
Prior_User_Order_Product = pd.merge(Order_Product_Name_Prior, 
                                    ordersDF[['order_id']], 
                                    how='left', on='order_id')

#Get Top 5 records
dfMostReOrderedProduct =  Prior_User_Order_Product['product_name'].value_counts().head(n=5)

#Plotting Graph
f, ax = plt.subplots(figsize=(10, 10))
splot=sns.barplot(x=dfMostReOrderedProduct.index,y=dfMostReOrderedProduct.values)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.ylabel('Number of Re-Orders', fontsize=13)
plt.xlabel('Top 5 Re-Ordered Products', fontsize=13)
plt.title('Top 5 Re-Ordered Products')
plt.xticks(rotation='vertical')
plt.show()

del Order_Product_Name_Prior
del Prior_User_Order_Product
del dfMostReOrderedProduct



#-------------------Order Ratio Departmentwise------------------------------------

#Merging Product and Department
dfProductDept = productsDF.merge(departmentsDF[['department_id','department']] ,how='inner', on='department_id')

#Merging Product and Department with Orders
dfOrderProductAll = order_products_priorDF.merge(dfProductDept[['product_id','product_name','department']],
                                             how='inner', on='product_id')
											 
#Grpuping for Summation	
""" 
#This query is used for latest python version
dfOrderSummaryByDept =  pd.DataFrame()
dfOrderSummaryByDept['reordered_total'] = pd.DataFrame(dfOrderProductAll.groupby("department")["reordered"].
                                                       agg(reordered_total = 'sum')).reset_index()['reordered_total']


dfOrderSummaryByDept['ordered_total'] = pd.DataFrame(dfOrderProductAll.groupby("department")["reordered"].
                                                       agg(ordered_total = 'count')).reset_index()['ordered_total']

"""		
#This query is used for old python version								 
#dfOrderSummaryByDept = dfOrderProductAll.groupby("department")["reordered"].aggregate(
#        {'reordered_total': sum,'ordered_total': 'count'}).reset_index()

#Calculating Ratio
dfOrderSummaryByDept['ratio'] = dfOrderSummaryByDept['reordered_total'] / dfOrderSummaryByDept['ordered_total']

#Sorting based on Ratio 
dfOrderSummaryByDept=dfOrderSummaryByDept.sort_values('ratio', ascending=False)

plt.figure(figsize=(12,8))
sns.pointplot(dfOrderSummaryByDept['department'].values, dfOrderSummaryByDept['ratio'].values)
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

del dfProductDept
del dfOrderProductAll
del dfOrderSummaryByDept
