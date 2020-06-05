# Read the data：

pd.set_option('display.max_columns', None)
customers = pd.read_csv('brazilian-ecommerce/olist_customers_dataset.csv')
orders = pd.read_csv('brazilian-ecommerce/olist_orders_dataset.csv')
order_reviews = pd.read_csv('brazilian-ecommerce/olist_order_reviews_dataset.csv')
order_items = pd.read_csv('brazilian-ecommerce/olist_order_items_dataset.csv')
order_payments=pd.read_csv('brazilianecommerce/olist_order_payments_dataset.csv')
products = pd.read_csv('brazilian-ecommerce/olist_products_dataset.csv')
sellers = pd.read_csv('brazilian-ecommerce/olist_sellers_dataset.csv')
geolocation = pd.read_csv('brazilian-ecommerce/olist_geolocation_dataset.csv')


# Clean Orders:

# remove unrelated variables
Orders=orders.drop(['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_status'],axis=1)
# change the format of order date
orders['order_delivered_customer_date']=pd.to_datetime(orders['order_delivered_customer_date'], format='%Y/%m/%d %H:%M:%S')
orders['order_estimated_delivery_date']=pd.to_datetime(orders['order_estimated_delivery_date'], format='%Y/%m/%d %H:%M:%S')
# obtain the time for delivering in seconds
orders['time_difference'] = (orders['order_estimated_delivery_date']-orders['order_delivered_customer_date']).dt.total_seconds()
# remove variables after obtaining delivery time
orders=orders.drop(['order_delivered_customer_date','order_estimated_delivery_date'],axis=1)
# delete the null value
orders = orders[~(orders['time_difference'].isnull())]


# Clean order_reviews:

# remove unrelated  variables
order_reviews=order_reviews.drop(['review_creation_date','review_answer_timestamp‘,'review_comment_title','review_comment_message'],axis=1)

# Clean order_items

# remove unrelated variable
order_items = order_items.drop(['shipping_limit_date'],axis=1)

# Cleaning order_payments

# remove unrelated variables
order_payments=order_payments.drop(['payment_sequential','payment_value'],axis=1)
# change format of ‘payment_type’
order_payments=pd.get_dummies(order_payments,columns=['payment_type'],prefix=['payment_type'])
# change ‘payment_installments’ from number of installments to whether it has payment installments
order_payments['payment_installments']=order_payments['payment_installments'].apply(lambda x: 0 if x<=1  else 1)


# Clean Products：

# drop unrelated variables
products = products.drop(['product_name_lenght','product_weight_g','product_length_cm','product_height_cm','product_width_cm'],axis=1)

# judge whether there are blank in 'product_category_name’ column
products = products[~(products['product_category_name'].isnull())]

# create a new column to get dummies with product_category_name
products = pd.get_dummies(products, columns=['product_category_name'], prefix=['products_category'])



# Merge orders ：

# merge Customers, Order_reviews &Order_payments based on primary key into Orders

orders = pd.merge(orders,customers,on='customer_id')
orders = pd.merge(orders, order_reviews, how='left',on=['order_id'])
orders = pd.merge(orders, order_payments,on=['order_id'])

# merge Products&Sellers into Order_items

order_items = order_items.merge(products,on='product_id')
order_items = order_items.merge(sellers,on='seller_id')

# merge Orders and Order_items
orders = orders.merge(order_items,on='order_id’)


# create new features based on the merged dataset and normalize the data：

# judge whether customer state is the same with customer state, if yes return 1, if no return 0
orders['isSameState'] = orders.apply(lambda x: 1 if x['customer_state'] == x['seller_state'] else 0, axis = 1)

# drop unrelated variables
orders=orders.drop(['order_id','customer_id','customer_unique_id','customer_zip_code_prefix','customer_city','review_id','product_id','seller_id','seller_zip_code_prefix','seller_city','customer_state','seller_state'],axis=1)

# normalize the data
data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)


Model 1(Identify people who generate  reviews)
# Set X,Y and split data:

# Set Y(label1), Y equals to ‘isComment’
orders['isComment']=orders['review_score']
orders['isComment']=np.where(orders['isComment']>=1,1,0)
label1=orders['isComment']

#Set X(X equals to data which drops 'review_score’ and 'isComment‘)
data = orders.drop(['review_score','isComment'],axis=1)

# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(data, label1, test_size = 0.1, random_state=5)


------------
# Tune parameters :

tuned_parameters = [{'criterion': ['gini', 'entropy'],
                     'max_depth': [3, 5, 7],
                     'min_samples_split': [3, 5, 7],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['accuracy', 'f1_macro']
for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DTC(), tuned_parameters, cv=5, scoring='%s' % score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")

---------------------
# Train and evaluate accuracy:

# a decision tree model with tuned parameters
dtc = DTC(criterion='entropy', min_samples_split=2, max_depth=80, max_features='sqrt')

# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))
-----------

# Perform prediction and visualisation:

# predict the test data
predicted = dtc.predict(X_test)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], normalise=True)

------------

Model 2(Identify people who generate 4/5* reviews )
# Set X,Y and split data:
# Set Y(label2),Y equals to modified ‘review_score’
label2=orders['review_score']
data=data[~label2.isin([0])]
label2=label2.apply(lambda x:1 if x>=4 else 0)

#Set X(X equals to data which drops 'review_score’ and 'isComment‘)
data = orders.drop(['review_score','isComment'],axis=1)

#split data
X_train, X_test, Y_train, Y_test = train_test_split(data, label2, test_size = 0.1, random_state=5)

------------


# Train and evaluate accuracy:
# a decision tree model with tuned values
dtc = DTC(criterion='entropy', min_samples_split=2, max_depth=80, max_features='sqrt')

# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))

# predict the test data
predicted = dtc.predict(X_test)

---------------

Model 1(Identify people who generate  reviews)
# Set X,Y and split data:

# Set Y(label1), Y equals to ‘isComment’
orders['isComment']=orders['review_score']
orders['isComment']=np.where(orders['isComment']>=1,1,0)
label1=orders['isComment']

#Set X(X equals to data which drops 'review_score’ and 'isComment‘)
data = orders.drop(['review_score','isComment'],axis=1)

# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(data, label1, test_size = 0.1, random_state=5)

-------------

# Tune parameters :
 tuned_parameters = [{'penalty': ['l2','none'],
                      'solver': ['lbfgs','newton-cg','sag'],
                      'multi_class': ['multinomial']}]

 scores = ['accuracy', 'roc_auc']

 for score in scores:
     print("# Tuning hyperparameters for %s" % score)
     print("\n")
     clf = GridSearchCV(LREG(random_state = 0), tuned_parameters, cv=5, scoring=score)
     clf.fit(X_train, Y_train)
     print("Best parameters set found on the training set:")
     print(clf.best_params_)
     print("\n")


------------------
     -----

# Train and evaluate accuracy:

# a logistic regression model with tuned parameters
lreg = LREG(multi_class='multinomial',penalty='l2',solver='lbfgs',random_state = 0)
# fit the model using some training data
lreg_fit = lreg.fit(X_train, Y_train)
# generate a mean accuracy score for the predicted data
train_score = lreg.score(X_train, Y_train)
# print the R2 of training data
print("Logistic regression R2 (Train) = " + str(round(train_score, 4)))
# predict the test data
predicted = lreg_fit.predict(X_test)
# generate a mean accuracy score for the predicted data
test_score = lreg_fit.score(X_test, Y_test)
# print the R2 of testing predictions
print("Logistic regression R2 (Test) = " + str(round(test_score, 4)))

--------------------
# Perform prediction and visualisation:

# predict the test data
predicted = lreg.predict(X_test)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], normalise=True)
