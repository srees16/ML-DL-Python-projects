#Complete projects
'''
Project should contain
1. EDA (missing values, distribution, encoding, outliers & removal, feature space exploration)
2. Various visualizations
3. SMOTEd imbalanced dataset
4. PCA
5. Various accuracy metrics
6. Probe for best classifier among a list of multiple classifiers
7. Grid Search & k-Fold
8. Call backs
9. ML interpretability using LIME
10. Flask API

Datasets:
Mushrooms
Housing price
Chocolate rating
Adults
Olympics
Telecom churn
Wine
Breast cancer
Back pain

'''

mush = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\mushrooms.csv')
mush.head()
mush.tail()
mush.sample(6, random_state = 3)
mush.info()
mush.shape
mush.columns
len(mush.select_dtypes(['object']).columns)

# printing unique values in each column
for feature in mush.columns:
    print('No of unique features in {} are: {}'.format(feature, mush[feature].unique()))

# find rows which may have same features but different labels
print('Known mushrooms: {} \nUnique mushrooms: {}'.format(len(mush.index), len(mush.drop_duplicates().index)))

# probing for mushrooms with the same features but different classes
print('Known mushrooms: {}\nMushrooms with same features: {}'.format(len(mush.index), 
      len(mush.drop_duplicates(subset = mush.drop(['class'], axis = 1).columns).index)))

# eliminating constant & quasi constant features
mush_encoded = mush.apply(lambda x : pd.factorize(x)[0])
mush_encoded.sample(5)
mush_encoded.shape
mush_encoded.columns.values
mush_encoded.corr().values[:, 0]
thresh_variance = VarianceThreshold(threshold = .01)
thresh_variance.fit(mush_encoded)
sum(thresh_variance.get_support()) # gives the no of features that are not constant
len(mush_encoded.columns[thresh_variance.get_support()])
[x for x in mush_encoded.columns if x not in mush_encoded.columns[thresh_variance.get_support()]]
len([x for x in mush_encoded.columns if x not in mush_encoded.columns[thresh_variance.get_support()]])
# mush_encoded = thresh_variance.transform(mush_encoded)

# dropping irrelevant feature as per above result
mush_encoded.drop('veil-type', axis = 1, inplace = True)
mush_encoded.shape

# categorical correlation
plt.figure(figsize = (len(mush_encoded.columns), len(mush_encoded.columns) + 1))
corrs = mush_encoded.corr(method = 'pearson', min_periods = 1)
sns.heatmap(corrs, annot = True)

'''prefer gill-attachment over veil-color'''
# zooming the heatmap for strongest 12 features
# zoomed heat map
k = 12
cols = corrs.nlargest(k, 'class')['class'].index
print(cols)
cor_map = np.corrcoef(mush_encoded[cols].values.T)
sns.set(font_scale = 1.25)
plt.subplots(figsize = (14, 12))
sns.heatmap(cor_map, cbar = True, vmax = .8, lw = 0.1, square = True, annot = True, cmap = 'viridis', linecolor = 'w', 
            xticklabels = cols.values, yticklabels = cols.values, annot_kws = {'size': 12})

strong_features = corrs[abs(corrs) > .6].index.sort_values(ascending = False)
len(strong_features)

x = mush_encoded.drop(['class'], axis = 1) # input features
y = mush_encoded['class'] # output variable(poisonous or edible)

split_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = split_size, random_state = 22)

# Creation of Train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = split_size, random_state = 5)


models = [LogisticRegression(solver = 'liblinear'), RandomForestClassifier(n_estimators = 100), MLPClassifier(), 
          GaussianNB(), DecisionTreeClassifier(), xgb()]
models
names = ['Logistic Regression', 'Random Forest', 'ANN', 'Gaussian NB', 'DecisionTree Classifier',
         'XGBClassifier']

kfold = model_selection.KFold(n_splits = 5)

for i in range(len(models)):
    cv_result = model_selection.cross_val_score(models[i], X_train, y_train, cv = kfold, scoring = 'accuracy')
    score = models[i].fit(X_train, y_train)
    prediction = models[i].predict(X_val)
    acc_score = accuracy_score(y_val, prediction)
    print ('-' * 40)
    print ('{0}: {1}'.format(names[i], acc_score))

'''Random Forest classifier'''
randomForest = RandomForestClassifier(n_estimators = 100)
randomForest.fit(X_train, y_train)
prediction = randomForest.predict(X_test)

print('-' * 40)
print('Accuracy score:')
print(accuracy_score(y_test, prediction))
print('-' * 40)
print('Confusion Matrix:')
print(confusion_matrix(y_test, prediction))
print('-' * 40)
print('Classification Matrix:')
print(classification_report(y_test, prediction))

'''XGBoost classifier for important features'''
xgbclassifier = xgb()
xgb_yPred = xgbclassifier.fit(X_train, y_train).predict(X_test)
accuracy_xgb = accuracy_score(xgb_yPred, y_test)
confusion_matrix(xgb_yPred, y_test)
print(accuracy_xgb)
# After fitting the model,plot histogram feature importance graph
fig, ax = plt.subplots(figsize = (10, 4))
plot_importance(xgbclassifier, ax = ax)

'''Tuning for optimal hyper parameters using Grid Search '''

# probing optimal batch size
batch_Size = [8, 16, 32, 50, 64, 100, 128]
# probing optimal no of epochs
epochs = [10, 50, 100, 150, 200]
# probing for best optimizer
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# probing for optimizer learning rate
learn_rate = [0.001, 0.01, 0.1, 0.2 ,0.3]
# probing for momentum
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# probing for weight initialization mode
initialization = ['normal', 'zero', 'uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 
                  'lecun_uniform']
# probing for optimal activation
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# dropout is best combined with a weight constraint such as the max norm constraint
weights = [1, 2, 3, 4, 5]
# probing for best dropout rate
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# probing for no of neurons in hidden layers
no_of_neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(batch_size = batch_Size, epochs = epochs, optimizer = optimizers, learn_rate = learn_rate, 
                  momentum = momentum, init = initialization, activation = activation, weight_constraint = weights,
                  dropout_rate = dropout_rate, neurons = no_of_neurons)
grid = GridSearchCV(estimator = MLPClassifier(), param_grid = param_grid, n_jobs = -1)
gSearch = grid.fit(X, y)
best_params = gSearch.best_params_
best_accuracy = gSearch.best_score_

# summarize results
print("Best score: %f using params %s" % (gSearch.best_score_, gSearch.best_params_))
means = gSearch.cv_results_['mean_test_score']
stds = gSearch.cv_results_['std_test_score']
params = gSearch.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))