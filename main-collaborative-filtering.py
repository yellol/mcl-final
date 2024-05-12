import pandas as pd

# user input
movie_user_likes = [("Amazing Spider-Man, The (2012)", 5), ("Mission: Impossible III (2006)", 4),
                    ("Toy Story 3 (2010)", 2), ("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)", 4)]

movie_data = pd.read_csv('dataset\\movies.csv')
rating_data = pd.read_csv('dataset\\ratings.csv')
rating_data = pd.merge(movie_data, rating_data).drop(['genres', 'timestamp'], axis=1)

rating_weight = rating_data.pivot_table(index=['userId'], columns=['title'], values='rating')
rating_weight = rating_weight.dropna(thresh=10, axis=1).fillna(0, axis=1)
weighted_movie_data = rating_weight.corr(method='pearson')


# combine keywords, cast, genres, directors into a singular data column

def get_similar(movie_name, rating):
    similar_ratings = weighted_movie_data[movie_name] * (rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings


similar_movies = pd.DataFrame()
for movie, rating in movie_user_likes:
    similar_movies = similar_movies._append(get_similar(movie, rating), ignore_index=True)

print(similar_movies.sum().sort_values(ascending=False).to_string())
