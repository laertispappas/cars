import pandas as pd
import matplotlib.pyplot as plt

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

  def plot_average_rating_hist(self):
    # Compute the average rating of reviews in the data set and a histogram of all the ratings in the dataset.
    print "Mean ratings over all reviews:", self.rating_data.rating.mean()
    ratings = self.rating_data.rating
    ax = ratings.hist(bins=5)
    plt.xlabel("Rating")
    plt.grid(False)
    plt.grid(axis='y', color='white', linestyle='-')
    plt.title("ratings over all reviews");
    plt.show();

  def plot_user_and_movie_hist(self):
    print "Total Number of Reviews", self.rating_data.shape[0]
    print "Users in this set", self.rating_data.user_id.unique().shape[0], "Movies", self.rating_data.movie_id.unique().shape[0]
    plt.figure()
    ax = self.rating_data.groupby('user_id').rating.count().hist()
    plt.xlabel("Reviews per user")
    plt.grid(False)
    plt.grid(axis='y', color='white', linestyle='-')
    plt.show()
    ax = self.rating_data.groupby('movie_id').rating.count().hist()
    plt.xlabel("Reviews per movies")
    plt.grid(False)
    plt.grid(axis='y', color='white', linestyle='-')
    plt.show()

  def plot_average_user_rating(self):
    ##  Compute histograms of the average user rating and the average movie rating Print the overall mean.
    plt.figure()
    avg_ratings_by_user = self.rating_data.groupby('user_id').rating.mean()
    ax = avg_ratings_by_user.hist()
    plt.xlabel("Average review score")
    plt.grid(False)
    plt.grid(axis='y', color='white', linestyle='-')
    plt.title("Average User Rating")
    plt.show()

  def plot_average_movie_rating(self):
    plt.figure()
    avg_ratings_by_movie = self.rating_data.groupby('movie_id').rating.mean()
    ax = avg_ratings_by_movie.hist()
    plt.xlabel("Average review score")
    plt.grid(False)
    plt.grid(axis='y', color='white', linestyle='-')
    plt.title("Average Restaurant Rating")
    plt.show()