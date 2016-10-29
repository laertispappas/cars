from time import sleep
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')

class DataPlotter(object):
    def __init__(self, data_object):
        self.ratings = data_object.ratings
        self.users = data_object.users
        self.movies = data_object.movies

    def ratings_per(self, type):
        ax1 = self.ratings[type].value_counts(sort=False).plot(kind='bar', xticks=[])
        ax1.set_xlabel(type)
        ax1.set_ylabel('ratings')
        ax1.set_title('Total Ratings per ' + type)

        plt.show()

    # Which movies are most controversial amongst different ages?
    # how movies are viewed across different age groups. look at
    # how age is distributed amongst our users.
    def user_ages_distribution(self):
        self.ratings.age.plot.hist(bins=30)
        # plt.title("Distribution of users' ages")
        # plt.ylabel('count of users')
        # plt.xlabel('age');

    # Which movies do men and women most disagree on?