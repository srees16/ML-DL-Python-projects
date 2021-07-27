# Recommender Systems

# Ref: https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb?source=post_page---------------------------

'''
Data base of individual genres, movies, ratings
1. Got the genres from the movies dataset.
2. Got the genres and a count of their occurances as dict & list from the movies dataset
3. Vectorizing the genres from movies dataset
4. Got movie titles and their indexes
5. Function to give movie recommendations based on cosine similarity
6. Collaborative filtering: create a sample dataset from ratings dataset
7. Split to train n test datasets
8. convert the train n test datasets as numpy matrix
9. calculate the pairwise distance for user correlation and item correlation
10. function to predict the train dataset based on input of user correlation or item correlation

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv('D:/Machine Learning Data Science/Projects/Datasets/recommender movielens/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('D:/Machine Learning Data Science/Projects/Datasets/recommender movielens/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('D:/Machine Learning Data Science/Projects/Datasets/recommender movielens/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

'''Ratings Dataset'''

# Check the top 5 rows
print(ratings.head())

# Check the file info
print(ratings.info())

'''Users Dataset'''

# Check the top 5 rows
print(users.head())

# Check the file info
print(users.info())

'''Movies Dataset'''

# Check the top 5 rows
print(movies.head())

# Check the file info
print(movies.info())

'''Data Exploration'''

# Import new libraries
from wordcloud import WordCloud, STOPWORDS

# Create a wordcloud of the movie titles
movies['title'].isna().count()
movies['title'] = movies['title'].fillna("").astype('str')
title_corpus = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()

'''Ratings'''

# Get summary statistics of rating
ratings['rating'].describe()

# Import seaborn library
import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.5)

# Display distribution of rating
ratings['rating'].isna().count()
sns.distplot(ratings['rating'].fillna(ratings['rating'].median()))

# Join all 3 files into one dataframe
dataset = pd.merge(pd.merge(movies, ratings),users)
# Display 20 movies with highest ratings
dataset[['title','genres','rating']].sort_values('rating', ascending = False).head(5)

'''Genres'''

# Make a set of the genre keywords present in the movie dataset
genre_labels = set()
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
genre_labels

# Function to count the number of times each of the genres appear
def count_word(dataset, referring_column, census):
    keyword_count = dict()
    for s in census:
        keyword_count[s] = 0
    for census_keywords in dataset[referring_column].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]: 
            if pd.notnull(s): 
                keyword_count[s] += 1    
    ''' convert the dictionary in a list to sort the keywords by frequency'''
    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
keyword_occurences[:5]
dum

# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword_occurences[0:18]
for s in trunc_occurences:
    genres[s[0]] = s[1]

# Create the wordcloud
genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)

# Plot the wordcloud
f, ax = plt.subplots(figsize=(16, 8))
plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


'''Content-Based Recommendation Model: Implementation'''

# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

# Vectorising the movie genres from the movie dataset
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):
    indexx = indices[title]
    sim_scores = list(enumerate(cosine_sim[indexx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

genre_recommendations('Good Will Hunting (1997)').head(10)
genre_recommendations('Toy Story (1995)').head(10)
genre_recommendations('Saving Private Ryan (1998)').head(10)


'''Collaborative Filtering Recommendation Model: Implementation'''

# Fill NaN values in user_id and movie_id column with 0
ratings['user_id'].isna().sum()
ratings['user_id'] = ratings['user_id'].fillna(0)
ratings['movie_id'] = ratings['movie_id'].fillna(0)

# Replace NaN values in rating column with average of all values
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())

# Randomly sample 1% of the ratings dataset
small_data = ratings.sample(frac = 0.02)
type(small_data)
# Check the sample info
print(small_data.info())

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(small_data, test_size = 0.2)

# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])

# Check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)

from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:4, :4])

# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_ratings = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_ratings[:, np.newaxis])
        pred = mean_user_ratings[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

'''Evaluation'''

from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

# RMSE on the train data
print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))

# RMSE on the test data
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))