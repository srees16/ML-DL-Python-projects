# All functions

''' missing data'''
def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/train.isnull().count() * 100)
    missing_values = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    missing_values['Types'] = types
    missing_values.sort_values('Total', ascending = False, inplace = True)
    return(np.transpose(missing_values))
missing_data(train)

''' Find correlation between columns'''
def plot_correlation(data, size = 15):
    corr = data.corr(method = 'pearson')
    fig, ax = plt.subplots(figsize = (0.4 * size, 0.4 * size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
plot_correlation(dataset)

'''renaming n mapping'''
def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass'] == ' Local-gov' or x['workclass'] == ' State-gov' : return 'govt'
    elif x['workclass'] == ' Private' : return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc' : return 'self_employed'
    else: return 'without_pay'

''' probing for classification algorithms with better accuracy'''

from xgboost import XGBClassifier
models = []
names = ['LR', 'Random Forest', 'Neural Network', 'GaussianNB', 'DecisionTreeClassifier', 'XGBoost']
models.append(LogisticRegression(solver = 'liblinear'))
models.append(RandomForestClassifier(n_estimators = 100))
models.append(MLPClassifier())
models.append(GaussianNB())
models.append(DecisionTreeClassifier())
models.append(XGBClassifier())
models

kfold = model_selection.KFold(n_splits = 5, random_state = 7)

for i in range(0, len(models)):
    cv_result = model_selection.cross_val_score(models[i], X_train, y_train, cv = kfold, scoring = 'accuracy')
    score = models[i].fit(X_train, y_train)
    prediction = models[i].predict(X_val)
    acc_score = accuracy_score(y_val, prediction)
    print ('-' * 40)
    print ('{0}: {1}'.format(names[i], acc_score))


'''Detect & remove outliers by function'''

# Function to detect outliers
minimum = 0
maximum = 0
def detect_outlier(feature):
    first_q = np.percentile(feature, 25)
    third_q = np.percentile(feature, 75)
    IQR = third_q-first_q #IQR is the distance between 3rd Quartile and 1st Qartile
    IQR *= 1.5
    minimum = first_q - IQR #acceptable minimum value
    maximum = third_q + IQR #acceptable maximum value
    flag = False
    if(minimum > np.min(feature)):
        flag = True
    if(maximum < np.max(feature)):
        flag = True
    return flag

# Detecting outliers using above function
X = dataset.iloc[:, :-1] #taking all the columns except the output column
for i in range(len(X.columns)):
    if(detect_outlier(X[X.columns[i]])):
        print('"', X.columns[i], '"', 'contains Outliers!')

# Function to remove outliers
def remove_outlier(feature): #use tukey method to remove outliers. whiskers are set at 1.5 times IQR
    first_q = np.percentile(X[feature], 25)
    third_q = np.percentile(X[feature], 75)
    IQR = third_q-first_q
    IQR *= 1.5
    minimum = first_q - IQR #acceptable minimum value
    maximum = third_q + IQR #acceptable maximum value
    median = X[feature].median()
    #values beyond the acceptance range are considered outliers. replace them with median of that feature
    X.loc[X[feature] < minimum, feature] = median
    X.loc[X[feature] > maximum, feature] = median

# Removing outliers
for i in range(len(X.columns)):
    for i in range(len(X.columns)):
        remove_outlier(X.columns[i])
        
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
    
'''Confusion matrix'''
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    '''Function prints and plots the confusion matrix. Normalization can be applied by setting normalize=True'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    
y_pred = model_tf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion matrix, without normalization')
plt.show()