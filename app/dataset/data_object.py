from app.dataset.data_plotter import DataPlotter
from app.dataset.loader import Loader
import pandas as pd
import numpy as np

class DataObject(object):
    # Non contextual feature atrributes for LDOS dataset
    NON_CONTEXT_ATTRS = ['userID','itemID','rating','age','sex','city','country', 'director', 'movieCountry',
                              'movieLanguage', 'movieYear', 'genre1', 'genre2', 'genre3', 'actor1', 'actor2','actor3',
                              'budget']

    # Feature attributes for LDos dataset
    feature_dict = {
        'rating': 0, 'age': 1, 'sex': 2, 'city': 3,
        'country': 4, 'time': 5, 'daytype': 6, 'season': 7,
        'location': 8, 'weather': 9, 'social': 10,
        'endEmo': 11, 'dominantEmo': 12, 'mood': 13,
        'physical': 14, 'decision': 15, 'interaction': 16
    }

    def __init__(self):
        self.ratings, self.users, self.movies = Loader().load_dapaul_csv()
        self.total_ratings = self.__total_ratings()
        self.total_movies = self.__total_movies()
        self.total_users = self.__total_users()

        self.context_types, self.total_context_types = self.__get_contexts()

    def user_ids(self):
        return self.users.keys()

    def preferences_from_user(self, user_id, orderByID=True):
        userPrefs = self.users.get(user_id, None)

        if userPrefs is None:
            raise ValueError('User not found in dataset!')

        userPrefs = userPrefs.items()

        if not orderByID:
            userPrefs.sort(key=lambda userPref: userPref[1][0], reverse=True)
        else:
            userPrefs.sort(key=lambda userPref: userPref[0])

        return userPrefs

    def print_specs(self):
        print "Num of ratings", self.total_ratings
        print "Num of users", self.total_users
        print "Num of movies", self.total_movies
        print "Average age", int(self.ratings['age'].mean())
        print "Num of countries", len(self.ratings['country'].unique())
        print "Num of Cities", len(self.ratings['city'].unique())
        print "Min ratings of single User", self.ratings.groupby('userID').size().min()
        print "Max ratings of single User", self.ratings.groupby('userID').size().max()
        print "******"
        print 'Context types: ', self.context_types
        print "Total Context Types: ", self.total_context_types

    def plot_stats(self):
        plotter = DataPlotter(self)
        plotter.ratings_per('age')
        plotter.ratings_per('userID')
        plotter.ratings_per('itemID')
        plotter.num_of_movies_per_genre()

    def top_rated(self, N=25):
        pass

    def high_rated_movies(self):
        movie_stats = self.ratings.groupby('itemID').agg({'rating': [np.size, np.mean]})
        return movie_stats.head()

    def __total_ratings(self):
        return len(self.ratings)

    def __total_movies(self):
        return len(self.movies)

    def __total_users(self):
        return len(self.users)

    def __get_contexts(self):
        total =  0
        types = []
        for context in self.ratings.columns:
            if context not in DataObject.NON_CONTEXT_ATTRS:
                total += 1
                types.append(context)
        return types, total

    def __calculate_total_context_conditions(self, some):
        pass