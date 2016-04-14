import pandas as pd

class DataObject(object):
  "A class representing ratings data, movie data and user data"

  def __init__(self, user_data, movie_data, rating_data):
    self.user_data = pd.DataFrame(user_data)
    self.movie_data = pd.DataFrame(movie_data)
    self.rating_data = pd.DataFrame(rating_data)

    # Insert user average and movie average on rating data
    self.rating_data.insert(0, 'user_average', None)
    self.rating_data.insert(0, 'movie_average', None)

    user_average_ratings = self.rating_data.groupby('user_id').rating.mean()
    movie_average_ratings = self.rating_data.groupby('movie_id').rating.mean()

    for index, row in self.rating_data.iterrows():
      self.rating_data.loc[index, 'user_average'] = user_average_ratings[row['user_id']]
      self.rating_data.loc[index, 'movie_average'] = movie_average_ratings[row['movie_id']]

  def number_of_ratings(self):
    print "number of ratings", self.rating_data.shape[0]

  def number_of_users(self):
    print "number of users", self.user_data.id.unique().shape[0]

  def number_of_movies(self):
    print "number of movies", self.movie_data.id.unique().shape[0]

  def mean_rating_value(self):
    print "Mean rating", self.rating_data.rating.mean()