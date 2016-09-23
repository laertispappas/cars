import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

class DataObject(object):
  "A class representing ratings data, movie data and user data"

  def __init__(self, user_data, movie_data, rating_data, movie_metadata={}):
    # Holds a mappings between a user / item index from Rating matrix to user and item id accordingly
    self.user_ids, self.movie_ids = {}, {}

    self.user_data = pd.DataFrame(user_data)
    self.movie_data = pd.DataFrame(movie_data)
    self.rating_data = pd.DataFrame(rating_data)
    self.movie_metadata = pd.DataFrame(movie_metadata.values())

    # Insert user average and movie average on rating data
    self.rating_data.insert(0, 'user_average', None)
    self.rating_data.insert(0, 'movie_average', None)
    # calculate user and item mean
    user_average_ratings = self.rating_data.groupby('user_id').rating.mean()
    movie_average_ratings = self.rating_data.groupby('movie_id').rating.mean()
    for index, row in self.rating_data.iterrows():
      self.rating_data.loc[index, 'user_average'] = user_average_ratings[row['user_id']]
      self.rating_data.loc[index, 'movie_average'] = movie_average_ratings[row['movie_id']]
      user_id = row['user_id']
      movie_id = row['movie_id']
      if user_id not in self.user_ids.values():
          self.user_ids[self.user_ids.keys().__len__()] = user_id
      if movie_id not in self.movie_ids.values():
          self.movie_ids[self.movie_ids.keys().__len__()] = movie_id

    self.context_dimension_columns = self.get_dimensions()
    self.context_conditions = self.get_conditions()
    self.num_of_conditions = self.number_of_conditions();

    self.rating_matrix = self.get_rating_matrix()
    self.train_matrix, self.test_matrix = train_test_split(self.rating_matrix, test_size=0.2)

  def get_rating_matrix(self):
      return self.rating_data

      # A sparse matrix in dictionary form (can be a SQLite database). Tuples contains doc_id and term_id.
      # doc_term_dict = {('d1', 't1'): 12, ('d2', 't3'): 10, ('d3', 't2'): 5}

      # extract all unique documents and terms ids and intialize a empty dataframe.
      # rows = set([d for (d, t) in doc_term_dict.keys()])
      # cols = set([t for (d, t) in doc_term_dict.keys()])
      # df = DataFrame(index=rows, columns=cols)
      # df = df.fillna(0)

      # assign all nonzero values in dataframe
      # for key, value in doc_term_dict.items():
      #   df[key[1]][key[0]] = value
      # print df

  def get_train_matrix(self):
    return self.train_matrix

  def get_test_matrix(self):
    return self.test_matrix

  def print_specs(self):
    print "number of users", self.number_of_users()
    print "number of movies", self.number_of_movies()
    print "number of ratings", self.number_of_ratings()
    print "number of dimensions", len(self.context_dimension_columns), self.context_dimension_columns
    print "number of conditions", self.num_of_conditions

  def number_of_ratings(self):
    return self.rating_data.shape[0]

  def number_of_users(self):
    return self.user_data.id.unique().shape[0]

  def number_of_movies(self):
    return self.movie_data.id.unique().shape[0]

  def number_of_conditions(self):
    total = 0
    for context in self.context_conditions:
      total += self.context_conditions[context].size
    return total

  def mean_rating_value(self):
    print "Mean rating", self.get_mean_for(self.rating_data)

  def get_mean_for(self, df):
    return df.rating.mean()

  def get_dimensions(self):
    context_columns = [col for col in self.rating_data if col not in ['user_average', 'movie_average', 'movie_id', 'rating', 'user_id']]
    return context_columns

  # N/A values?
  def get_conditions(self):
    conditions = {}
    for dimension in self.context_dimension_columns:
      conditions[dimension] = self.rating_data[dimension].unique()

    return conditions