'''EDA'''

# https://medium.com/swlh/a-complete-guide-to-exploratory-data-analysis-and-data-cleaning-dd282925320f
# https://github.com/nickvega1989/Predicting-Housing-Prices/blob/master/Final%20-%20Model%20Improvement.ipynb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

train = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Ames Housing train.csv')
test = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Ames Housing test.csv')

train.sample(2)

train.rename(columns={'Id': 'id', 'PID': 'pid', 'MS SubClass': 'ms_subclass',}, inplace=True)

train.columns = [i.replace(' ', '_').lower() for i in train.columns] 

train.columns

def ames_eda(df): 
    eda_df = {}
    eda_df['null_sum'] = df.isnull().sum()
    eda_df['null_pct'] = df.isnull().mean()
    eda_df['dtypes'] = df.dtypes
    eda_df['count'] = df.count()
    eda_df['mean'] = df.mean()
    eda_df['median'] = df.median()
    eda_df['min'] = df.min()
    eda_df['max'] = df.max()    
    return pd.DataFrame(eda_df)
ames_eda(train)

train.dtypes.value_counts()

train.select_dtypes(include=['object']).columns

correlations = train.corrwith(train['saleprice']).iloc[:-1].to_frame()
correlations['abs'] = correlations[0].abs()
sorted_correlations = correlations.sort_values('abs', ascending=False)[0]
fig, ax = plt.subplots(figsize=(10,20))
sns.heatmap(sorted_correlations.to_frame(), cmap='coolwarm', annot=True, vmin=-1, vmax=1, ax=ax)

sns.boxplot(train['centralair'], train['saleprice']).set_title('Central Air vs. Sale Price')
sns.boxplot(train['kitchenqual'], train['saleprice']).set_title('Kitchen Quality vs. Sale Price')

train['garagequal'].value_counts()
train['kitchenqual'].value_counts()

def garage_qual_cleaner(cell):
    if cell == 'Ex':
        return 5
    elif cell == 'Gd':
        return 4
    elif cell == 'TA':
        return 3
    elif cell == 'Fa':
        return 2
    elif cell == 'Po':
        return 1
    else:
        return 0
    
train['kitchenqual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})

def data_cleaner(df):
    # map numeric values onto all the quality columns using a quality dictionary
    qual_dict = {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}
    # create a list of ordinal column names. last section ignores "overall quality columns which will be addressed below
    ordinal_col_names = [col for col in df.columns if (col[-4:] in ['qual', 'cond']) and col[:3] != 'ove']
    # creating a new feature called age
    df['age'] = df.apply(lambda row: row['yrsold'] - max(row['yearbuilt'], row['yearremodadd']), axis=1)
    # dummify the date sold column 
    df['date_sold'] = df.apply(lambda row: str(row['mosold'])+ '-' + str(row['yrsold']), axis=1)
    df.loc[:,df.dtypes!= 'object'] = df.loc[:, df.dtypes != 'object'].apply(lambda col: col.fillna(col.mean()))
    
    # transforming columns 
    df[ordinal_col_names] = df[ordinal_col_names].applymap(lambda cell: 2 if pd.isnull(cell) else qual_dict[cell])
    
    return df

train.columns
# applying the function to train data
train = data_cleaner(train)

plt.figure(figsize=(35,10)) # adjust the fig size to see everything
sns.boxplot(train['neighborhood'], train['saleprice']).set_title('Sale Price varies widely by Ames Neighborhood')

pd.get_dummies(train, columns = ['neighborhood'], drop_first = True)

sns.lmplot(x = '1stflrsf', y = 'saleprice', data = train, fit_reg = False)

train.loc[train['1stflrsf'] > 3800]

rows_to_drop = [616, 960, 1885]
for row in rows_to_drop:
    train.drop(row, inplace=True)