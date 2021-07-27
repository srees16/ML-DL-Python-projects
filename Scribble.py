'''''''''''''''''''''''''''Daily Exercises'''''''''''''''''''''''''''

'''Pandas & Series'''
# https://www.machinelearningplus.com/python/101-pandas-exercises-python

pd.__version__
# convert the index of a series into a column of a dataframe
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(len(mylist))
my_dict = dict(zip(mylist, myarr))
pd.Series(index = myarr, data = mylist)
# combine many series to form a dataframe
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))
ser = pd.DataFrame(ser1, ser2)
# assign name to the series’ index
ser.index.name = 'alphabets'
# get the items of series A not present in series B
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[ser1.isin(ser2)] # items of series A present in series B
ser1[~ser1.isin(ser2)] # items of series A not present in series B
# get the items not common to both series A and series B
seru = pd.Series(np.union1d(ser1, ser2))
seri = pd.Series(np.intersect1d(ser1, ser2))
seru[~seru.isin(seri)]
# get the minimum, 25th percentile, median, 75th, and max of a numeric series
ser = pd.Series(np.random.normal(10, 5, 25))
np.percentile(ser, q = [0, 25, 50, 75, 100])
# get frequency counts of unique items of a series
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size = 30)))
pd.value_counts(ser, ascending = False)
# keep only top 2 most frequent values as it is and replace everything else as ‘Other’
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, 12))
ser[~ser.isin(ser.value_counts().index[:2])] = 'Other'
ser
# bin a numeric series to 10 groups of equal size
ser = pd.Series(np.random.random(20))
pd.qcut(ser, q = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], labels = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']).head()
# np.percentile(ser, [])
# convert a numpy array to a dataframe of given shape
ser = pd.Series(np.random.randint(1, 10, 35))
np.array(ser).reshape(7, 5)
ser.values.reshape(7, 5)
# find the positions of numbers that are multiples of 3 from a series
ser = pd.Series(np.random.randint(1, 10, 7))
print(ser)
np.argwhere(ser % 3 == 0)
# extract items at given positions from a series
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]
ser[pos] # or ser.take(pos)
# stack two series vertically and horizontally
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))
ser1.append(ser2)
pd.concat([ser1, ser2], axis = 1)
# get the positions of items of series A in another series B
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])
[pd.Index(ser1).get_loc(i) for i in ser2]
# Ex:
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5])
[pd.Index(ser1).get_loc(i) for i in ser2]
# compute the mean squared error on a truth and predicted series
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)
np.mean((truth-pred) ** 2)
#convert the first character of each element in a series to uppercase
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
[letter for word in ser[word][0].upper()] # Or
ser.map(lambda x: x[0].upper() + x[1:])
# calculate the number of characters in each word in a series
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.map(lambda x: len(x)) # or
pd.Series([len(i) for i in ser])
# compute difference of differences between consequtive numbers of a series
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
print(ser.diff().tolist())
print(ser.diff().diff().tolist())
# creating a dataframe quickly
df = pd.DataFrame(columns=['Item', 'Qty1', 'Qty2'])
for i in range(5):
    df.loc[i] = ['name_' + str(i)] + list(randint(1, 9, 2))
df
# convert a series of date-strings to a timeseries
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
time = pd.to_datetime(ser)
dics = {'dates': [], 'week_of_year': [], 'day_of_year': [], 'weekday_name': []}
for i in time:
    dics['dates'].append(i.day)
    dics['week_of_year'].append(i.weekofyear)
    dics['day_of_year'].append(i.dayofyear)
    dics['weekday_name'].append(i.weekday_name)
print('Date: ', dics['dates'],'\nWeek number:', dics['week_of_year'],'\nDay num of year:', dics['day_of_year'],'\nDay of week:', dics['weekday_name'])

time = emer['time'].iloc[0]
tym = pd.to_datetime(time)
tym.date()
tym.time()
tym.day

Pick up from Q 23 till Q 32

# filter words that contain atleast 2 vowels from a series
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
vowels = ['a', 'e', 'i', 'o', 'u', 'i']
mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
ser[mask]

[x % 6 == 0 for x in [x for x in range(25) if x % 2 == 0]]

# get the mean of a series grouped by another series
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weight = pd.Series(np.linspace(1, 10, 10))
print(weight.tolist())
print(fruit.tolist())
weight.groupby(fruit).mean()

# compute the euclidean distance between two series
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
np.linalg.norm(p-q) # OR
sum((p - q) ** 2) ** .5

# find all the local maxima (or peaks) in a numeric series
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
dd = np.diff(np.sign(np.diff(ser)))
np.where(dd == -2)[0] + 1

# replace missing spaces in a string with the least frequent character
my_str = 'dbc deb abed gade'
ser = pd.Series(list(my_str))
least = ser.value_counts().index[-1]
''.join(ser.replace(' ', least))

# change column values when importing csv to a dataframe



lis = [x for x in range(1, 10) if x % 2 == 0]
emer = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\911.csv')
lis = [x for x in range(len(emer.index)) if x % 50 == 0]
emer2 = emer.loc[[x for x in range(len(emer.index)) if x % 50 == 0]]

house = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', 
                    converters = {'medv' : lambda x: 'High' if float(x) > 25 else 'Low'})

# identify which columns has ? and replace with nan values
def illegible_chars(df):
    ills = []
    cols = df.columns
    for i in cols:
        if '?' in df[i].value_counts().index.tolist():
            ills.append(i)
            df[i].replace('?', np.nan, inplace = True)
    return ills

illegible_chars(df)

df['price'].isna().sum()
df['price'].dropna(inplace = True)
df[df['price'] == df['price'].max()][['make', 'price']]



'''https://github.com/Yorko/mlcourse.ai/blob/master/docker_files/check_docker.ipynb'''

# cancers = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\breast_cancer2.csv')
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
df = pd.DataFrame(X, columns=cancer.feature_names)
df.head()
sns.boxplot(x='mean radius', data=df)

import xgboost
X_train, X_test, y_train, y_test = train_test_split(X, y)
xgb = xgboost.XGBClassifier(n_estimators=200)
xgb.fit(X_train, y_train)
prediction = xgb.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, prediction[:,1])
plt.plot(fpr, tpr)

from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier(max_depth=3, random_state=17).fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=cancer.feature_names, filled=True)

'''Numpy'''

s = np.arange(4, 12).reshape(4, 2).astype('int16')
s.shape
s.ndim
s.itemsize
np.empty([4, 2])
np.arange(100, 200, 10).reshape(5, 2)

sampleArray = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
newArray = sampleArray[..., 2]
newArray

sampleArray = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], [27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
sampleArray[::]
sampleArray[::2]
sampleArray[::2, 1::2] # 2 here is step size

arrayOne = np.array([[5, 6, 9], [21 ,18, 27]])
arrayTwo = np.array([[15 ,33, 24], [4 ,7, 1]])
resultArray  = arrayOne + arrayTwo
for n in np.nditer(resultArray, op_flags = ['readwrite']):
    n[...] = n * n

narray = np.arange(10, 34).reshape(8, 3)
subarray = np.split(narray, 4)

sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[:, sampleArray[1, :].argsort()] # Sorting array by second row
sampleArray[sampleArray[:, 1].argsort()] # Sorting array by second column

sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
min_of_axis1 = np.amin(sampleArray, axis = 1)
max_of_axis0 = np.amax(sampleArray, axis = 0)

sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 
newColumn = np.array([[10,10,10]])
np.delete(sampleArray, 1, axis = 1)
np.insert(sampleArray, 1, newColumn, axis = 1)

np.__version__
np.show_config()
np.zeros(10)
np.zeros(10).size
np.zeros(10).itemsize
memory_size = np.zeros(10).size * np.zeros(10).itemsize
z = np.zeros(10)
z[4] = 1
x = np.arange(10, 50)
x[::-1]
np.arange(9).reshape(3, 3)
x =  [1,2,0,0,4,0]
np.nonzero(x)
np.eye(3)
np.random.random((3,3,3))
x = np.random.random((10, 10))
x.argmax(), x.argmin()
x.max(), x.min()
np.random.random(30).mean()

x = np.ones((10, 10))
x[1: -1, 1: -1] = 0
np.pad(x, constant_values = 0, pad_width = 1, mode = 'constant') # add a border or pad

np.diag(1 + np.arange(4), k = -1)
np.diag(np.arange(0, 6), k = -2)

x = np.arange(10)
y = np.arange(11, 20)
np.savez('temp_arra.npz', x = x, y = y)
with np.load('temp_arra.npz') as data: # load arrays from the temp_arra.npz file
    x1 = data['x']
    y1 = data['y']
    print(x1, y1)


x = np.random.uniform(0, 1, size = 10)
y = np.random.choice(('male', 'female'), size = 10)

x[0]
y[0]

label = LabelEncoder()
yt = label.fit_transform(y)
print(yt)
# inverse transformation
label.inverse_transform(yt)


# https://pynative.com/python-numpy-exercise
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

pd.Series(mylist)
pd.Series(myarr)
pd.Series(mydict).reset_index()


''''''''''''''''''''''Pytorch''''''''''''''''''''''

import torch

t = torch.ones(2, 2)
t.size()
t.view(4)
t.view(4).size()

a = torch.ones(2, 3)
b = torch.ones(2, 3)

c = a + b
c

c = torch.add(a, b)
c

'''_ means an inplace operation'''

c.add_(a)


# https://datascienceplus.com/selecting-categorical-features-in-customer-attrition-prediction-using-python
churn = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\churn_telecom.csv')


cat_vars = cat_vars.astype('category').cat.codes
cat_vars = cars.select_dtypes(include = ['object'])
cat_vars.cat.codes
cat_vars = cat_vars.apply(lambda x: pd.get_dummies(x, drop_first = True))
cat_vars.apply(lambda x: pd.factorize(x)[0])
np.where(cat_vars)
cat_vars[::] = lb.fit_transform(cat_vars[::]) # OR cat_vars['make'] = lb.fit_transform(cat_vars['make'])
cat_vars.isna().sum()
cat_vars.dropna(inplace = True)
hots = one_hot.fit_transform(cat_vars['make'].values.reshape(-1, 1)).toarray()

'''Principle Component Analysis'''

url = r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\cot_pca.csv'
df = pd.read_csv(url)
df.head(5)
pca_comps = PCA().fit(df)
# view eigenvalues. PCs with eigenvalues > 1 contributes greater variance and helps to retain PCs for analysis 
pca_comps.explained_variance_
pca_comps.explained_variance_ratio_ # proportion of Variance (from PC1 to PC6)

# Cumulative proportion of variance (from PC1 to PC6)   
np.cumsum(pca_comps.explained_variance_ratio_)

# get component loadings (correlation coefficient between original variables and the component) 
components = pca_comps.components_
num_pc = pca_comps.n_features_
pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, components)))
loadings_df['variable'] = df.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df

# get correlation matrix plot for loadings
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()

# get scree plot, useful to determine how many components need to consider for visualizing data. 1st 3 PCs explain the most variance
fig = plt.figure(figsize = (8,8))
ax = fig.add_axes([.1,.1,.6,.6])
ax.plot(pc_list, pca_comps.explained_variance_ratio_)


df = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Automobile_data.csv')
df.columns

car = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Cars93_miss.csv')
car.columns

'''https://pynative.com/python-pandas-exercise'''

df = pd.read_csv(r'D:\Programming Tutorials\Machine Learning\Projects\Datasets\car_prices.csv')
df.columns
df.head()

def weird_chars(dframe):
    weird_features = []
    for i in dframe.columns:
        if '?' in dframe[i].value_counts().index.tolist():
            weird_features.append(dframe[i].name)
    return weird_features
weird_chars(df)

def illegible_nan(df):
    for i in df.columns:
        df[i].replace('?', np.nan, inplace = True)
    return df
illegible_nan(df)

df.isna().sum().sum()

df[['make', 'price']][df['price'] == df['price'].max()]
df[df['make'] == 'toyota']

dataset = [[1,5.1,3.5,1.4,0.2,"Iris-setosa"],
            [2,4.9,3.0,1.4,0.2,"Iris-setosa"],
            [3,4.7,3.2,1.3,0.2,"Iris-setosa"],
            [4,4.6,3.1,1.5,0.2,"Iris-setosa"],
            [5,5.0,3.6,1.4,0.2,"Iris-setosa"],
            [6,5.4,3.9,1.7,0.4,"Iris-setosa"],
            [7,4.6,3.4,1.4,0.3,"Iris-setosa"],
            [8,5.0,3.4,1.5,0.2,"Iris-setosa"],
            [9,5.5,2.3,4.0,1.3,"Iris-versicolor"],
            [10,6.5,2.8,4.6,1.5,"Iris-versicolor"],
            [11,5.7,2.8,4.5,1.3,"Iris-versicolor"],
            [12,6.3,3.3,4.7,1.6,"Iris-versicolor"],
            [13,4.9,2.4,3.3,1.0,"Iris-versicolor"],
            [14,6.6,2.9,4.6,1.3,"Iris-versicolor"],
            [15,5.2,2.7,3.9,1.4,"Iris-versicolor"],
            [16,7.7,3.0,6.1,2.3,"Iris-virginica"],
            [17,6.3,3.4,5.6,2.4,"Iris-virginica"],
            [18,6.4,3.1,5.5,1.8,"Iris-virginica"],
            [19,6.0,3.0,4.8,1.8,"Iris-virginica"],
            [20,6.9,3.1,5.4,2.1,"Iris-virginica"],
            [21,6.7,3.1,5.6,2.4,"Iris-virginica"],
            [22,6.9,3.1,5.1,2.3,"Iris-virginica"],
            [23,5.8,2.7,5.1,1.9,"Iris-virginica"]]

def data_split():
    for i in dataset:
        print(i)
        x = dataset[1][:-1]
        y = dataset[0][-1]
        return x, y
x, y = data_split()
x

dataset[0][:-1]
dataset[0][-1]

# https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/


'''Mathematics about topics
PCA
Back propagation
Naive Bayes
Decision Tree
Logistic regression
Loss function
'''

import unittest

def adds(*args):
    return sum(*args)

class Testing(unittest.TestCase):
    
    def test_adds(self):
        self.assertEqual(adds([2,1]), 3)

if __name__ == '__main__':
    unittest.main()

'''Fast API'''
https://www.youtube.com/watch?v=7t2alSnE2-I


class Movie:
    
    
    def __init__(self, name, year):
        self.name = name # self.name is a property of self, not a variable
        self.year = year