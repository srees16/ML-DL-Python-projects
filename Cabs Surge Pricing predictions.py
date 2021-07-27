# Cabs Surge Pricing predictions

'''
Ref: https://github.com/architsingh15/Sigma-Cabs-Surge-Pricing_Predictions
'''

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Cabs-Surge-Pricing_Predictions_train.csv')
test = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Cabs-Surge-Pricing_Predictions_test.csv')

# XGBoost works on only numeric vectors
train['Gender'] = train['Gender'].replace(to_replace = {'Male': 0, 'Female': 1})
test['Gender'] = test['Gender'].replace(to_replace = {'Male': 0, 'Female': 1})

#XGBoost works on only numeric vectors
type_of_cab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
train['Type_of_Cab'] = train['Type_of_Cab'].replace(to_replace = type_of_cab)
test['Type_of_Cab'] = test['Type_of_Cab'].replace(to_replace = type_of_cab)

confidence_index = {'A': 0, 'B': 1, 'C': 2}
train['Confidence_Life_Style_Index'] = train['Confidence_Life_Style_Index'].replace(to_replace = confidence_index)
test['Confidence_Life_Style_Index'] = test['Confidence_Life_Style_Index'].replace(to_replace = confidence_index)

train['Surge_Pricing_Type'] = train['Surge_Pricing_Type'] - 1

X_train = train.copy()
X_test = test.copy()

# transforming values in Destination type using label encoding
'''

for f in ['Destination_Type']:
    label = LabelEncoder()
    label.fit(list(X_train[f].values) + list(X_test[f].values))
    X_train[f] = label.transform(list(X_train[f].values))
    X_test[f] = label.transform(list(X_test[f].values))

'''
# same as above, but more concise n better
for f in ['Destination_Type']:
    label = LabelEncoder()
    X_train[f] = label.fit_transform(list(X_train[f].values))
    X_test[f] = label.fit_transform(list(X_test[f].values))

'''returns set difference of the two arrays that have been passed as the arguments. Feature extraction, all columns
except first and last'''
features = np.setdiff1d(train.columns, ['Trip_ID', 'Surge_Pricing_Type'])

params = {"objective": "multi:softmax", "booster": "gbtree", "nthread": 4, "silent": 1, "eta": 0.08, 
         "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 1, "num_class": 3,
         "seed": 2016, "tree_method": "exact"}

dtrain = xgb.DMatrix(X_train[features], X_train['Surge_Pricing_Type'], missing = np.nan)
dtest = xgb.DMatrix(X_test[features], missing = np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round = nrounds, evals = watchlist, verbose_eval = 20)
test_preds = bst.predict(dtest)

submit = pd.DataFrame({'Trip_ID': test['Trip_ID'], 'Surge_Pricing_Type': test_preds + 1})
submit.to_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\XGB.csv', index = False)