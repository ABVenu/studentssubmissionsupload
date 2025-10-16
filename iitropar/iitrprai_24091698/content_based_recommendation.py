import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""# EDA - Exploratory Data Analysis"""

df = pd.read_csv("movie_dataset.csv")

df.head()

df.describe()

print(f"Features Available:\n\n{df.columns.values}")

features = ['genres', 'keywords', 'title', 'cast', 'director']

"""# Preprocessing"""

def combine_features(row):
    return row['title']+' '+row['genres']+' '+row['director']+' '+row['keywords']+' '+row['cast']

for feature in features:
    df[feature] = df[feature].fillna('')

df['combined_features'] = df.apply(combine_features, axis = 1)

print(df.loc[0, 'combined_features'])

"""# Vectorization"""

cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(count_matrix)

cosine_sim

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

"""# Similarity check"""

movie_user_likes = "Star Trek Beyond"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

"""# Recommendations"""

print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for i, element in enumerate(sorted_similar_movies):
    print(f"{i+1}) {get_title_from_index(element[0])}")
    if i+2>10:
        break

"""# Interactive Implementation"""

def get_recommendation_for_movie(movie_user_likes):
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    print("\nTop 10 similar movies to "+movie_user_likes+" are:\n")
    for i, element in enumerate(sorted_similar_movies):
        print(f"{i+1}) {get_title_from_index(element[0])}")
        if i+2>10:
            break

movie_user_likes = input("Enter the movie name: ")
get_recommendation_for_movie(movie_user_likes)

"""1. User Watching a video in a content streaming platform
2. Fetch video title (input)
3. Pass data from **step 2.** to above function `get_recommendation_for_movie`
4. Get top 10 recommended content
5. Send the recommended content to the content streaming platform
"""

