import pandas as pd

class DataObject(object):
  def __init__(self, user_data, movie_data, rating_data):
    self.user_data = pd.DataFrame(user_data)
    self.movie_data = pd.DataFrame(movie_data)
    self.rating_data = pd.DataFrame(rating_data)

  def number_of_ratings(self):
    print "number of ratings", self.rating_data.shape[0]

  def number_of_users(self):
    print "number of users", self.user_data.id.unique().shape[0]

  def number_of_movies(self):
    print "number of movies", self.movie_data.id.unique().shape[0]

  def mean_rating_value(self):
    print "Mean rating", self.rating_data.rating.mean()