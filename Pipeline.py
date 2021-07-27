'''''''''''''''''''''''''''Pipeline'''''''''''''''''''''''''''

https://www.altexsoft.com/blog/machine-learning-pipeline
https://www.datanami.com/2018/09/05/how-to-build-a-better-machine-learning-pipeline
https://medium.com/xandr-tech/lessons-learned-from-building-scalable-machine-learning-pipelines-822acb3412ad


# Case 1: https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf
train = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\loan train.csv')
test = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\loan test.csv')
train = train.drop('Loan_ID', axis=1)
train.dtypes

X = train.drop('Loan_Status', axis = 1)
y = train['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns

preprocessor = ColumnTransformer(transformers = [('num', numeric_transformer, numeric_features),
                                                 ('cat', categorical_transformer, categorical_features)])
rf = Pipeline(steps = [('preprocessor', preprocessor), ('classifier', RandomForestClassifier())])
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

classifiers = [KNeighborsClassifier(3), SVC(kernel = 'rbf', C = 0.025, probability = True), NuSVC(probability = True),
               DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]
for classifier in classifiers:
    pipe = Pipeline(steps = [('preprocessor', preprocessor), ('classifier', classifier)])
    pipe.fit(X_train, y_train)
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))

param_grid = {'classifier__n_estimators': [200, 500], 'classifier__max_features': ['auto', 'sqrt', 'log2'],
              'classifier__max_depth' : [4,5,6,7,8], 'classifier__criterion' :['gini', 'entropy']}

CV = GridSearchCV(rf, param_grid, n_jobs= 1)
CV.fit(X_train, y_train)
print(CV.best_params_)
print(CV.best_score_)

### Case 2: https://www.analyticsvidhya.com/blog/2020/01/build-your-first-machine-learning-pipeline-using-scikit-learn
train_data = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Bigmart sales Train.csv')
# impute missing values in item weight by mean
train_data['Item_Weight'].fillna(train_data.Item_Weight.mean(), inplace = True)
# impute outlet size in training data by mode
train_data['Outlet_Size'].fillna(train_data.Outlet_Size.mode()[0], inplace = True)

ohe = ce.OneHotEncoder(cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
                               'Outlet_Type'], use_cat_names = True)
train_data = ohe.fit_transform(train_data)

scaler = StandardScaler()
scaler.fit(np.array(train_data.Item_MRP).reshape(-1, 1))
train_data.Item_MRP = scaler.transform(np.array(train_data.Item_MRP).reshape(-1, 1))

train_X = train_data.drop(columns = ['Item_Identifier', 'Item_Outlet_Sales'])
train_Y = train_data['Item_Outlet_Sales']

train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size = 0.25, random_state = 0)
train_x.shape, test_x.shape, train_y.shape, test_y.shape

model_LR = LinearRegression()
model_LR.fit(train_x, train_y)
predict_train = model_LR.predict(train_x)
predict_test  = model_LR.predict(test_x)

# Root Mean Squared Error on train and test date
print('RMSE on train data: ', mean_squared_error(train_y, predict_train) ** (0.5))
print('RMSE on test data: ',  mean_squared_error(test_y, predict_test) ** (0.5))

model_RFR = RandomForestRegressor(max_depth = 10)
model_RFR.fit(train_x, train_y)

predict_train = model_RFR.predict(train_x)
predict_test = model_RFR.predict(test_x)

# Root Mean Squared Error on train and test data
print('RMSE on train data: ', mean_squared_error(train_y, predict_train) ** (0.5))
print('RMSE on test data: ',  mean_squared_error(test_y, predict_test) ** (0.5))

# plot the 7 most important features
plt.figure(figsize = (10, 7))
feat_importances = pd.Series(model_RFR.feature_importances_, index = train_x.columns)
feat_importances.nlargest(7).plot(kind = 'barh')

# training data with 7 most important features found above
train_x_if = train_x[['Item_MRP', 'Outlet_Type_Grocery Store', 'Item_Visibility', 'Outlet_Type_Supermarket Type3',
                      'Outlet_Identifier_OUT027', 'Outlet_Establishment_Year', 'Item_Weight']]
# test data with 7 most important features
test_x_if = test_x[['Item_MRP', 'Outlet_Type_Grocery Store', 'Item_Visibility', 'Outlet_Type_Supermarket Type3',
                    'Outlet_Identifier_OUT027', 'Outlet_Establishment_Year', 'Item_Weight']]

# create an object of the RandfomForestRegressor Model
model_RFR_with_if = RandomForestRegressor(max_depth = 10, random_state = 2)

# fit the model with the training data
model_RFR_with_if.fit(train_x_if, train_y)

# predict the target on the training and test data
predict_train_with_if = model_RFR_with_if.predict(train_x_if)
predict_test_with_if = model_RFR_with_if.predict(test_x_if)

# Root Mean Squared Error on the train and test data
print('RMSE on train data: ', mean_squared_error(train_y, predict_train_with_if) ** (0.5))
print('RMSE on test data: ',  mean_squared_error(test_y, predict_test_with_if) ** (0.5))

### Building Pipeline
# define class OutletTypeEncoder, this is custom transformer that creates 3 new binary columns
# custom transformer must have methods fit and transform
class OutletTypeEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y = None):
        return self

    def transform(self, x_dataset):
        x_dataset['outlet_grocery_store'] = (x_dataset['Outlet_Type'] == 'Grocery Store') * 1
        x_dataset['outlet_supermarket_3'] = (x_dataset['Outlet_Type'] == 'Supermarket Type3') * 1
        x_dataset['outlet_identifier_OUT027'] = (x_dataset['Outlet_Identifier'] == 'OUT027') * 1

        return x_dataset

# pre-processsing step
# Drop the columns
# Impute the missing values in column Item_Weight by mean
# Scale the data in the column Item_MRP
pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[('drop_columns', 'drop',
                                               ['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content', 'Item_Type',
                                                'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']),
                                              ('impute_item_weight', SimpleImputer(strategy='mean'), ['Item_Weight']),
                                              ('scale_data', StandardScaler(),['Item_MRP'])])

# define pipeline
model_pipeline = Pipeline(steps=[('get_outlet_binary_columns', OutletTypeEncoder()), ('pre_processing',pre_process),
                                 ('random_forest', RandomForestRegressor(max_depth=10,random_state=2))])
model_pipeline.fit(train_x, train_y)
model_pipeline.predict(train_x)

# read test data set and call predict function only on pipeline object to make predictions
test_data = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\Bigmart sales Test.csv')
model_pipeline.predict(test_data)