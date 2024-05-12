import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_user_likes = "Night at the Museum"

movie_data = pd.read_csv('dataset\\movie_dataset.csv')


# helper functions to run the algorithm

def get_title_from_index(index):
    return movie_data[movie_data.index == index]["title"].values[0]


def get_index_from_title(title):
    return movie_data[movie_data.title == title]["index"].values[0]


# clean data by dropping na rows

features = ['keywords', 'cast', 'genres', 'director']

# clean data by filling na rows with spaces

for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')


# combine keywords, cast, genres, directors into a singular data column

def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]
    except:
        print("Error:", row)


movie_data["combined_features"] = movie_data.apply(combine_features, axis=1)

# vectorizer to fit data

cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data["combined_features"])
similarity = cosine_similarity(count_matrix)

# grab user's likes (input for algorithm) and print out first 50 titles

print("User likes: " + movie_user_likes)
print("Sorted by similarity (high to low). ")

movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(similarity[movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

i = 0
for element in sorted_similar_movies:
    print(str(i) + ". " + get_title_from_index(element[0]))
    i = i + 1
    if i > 50:
        break