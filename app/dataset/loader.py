# Dataset is alresy imported in a POstgreSQL database
# This class is responsible for loading the models in
# the apropriate data structure for recommender

from app.entities.datastore import Datastore
import pandas as pd
import numpy as np

# A perl like dictionary implementation
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

class Loader(object):
    """Loads all ratings from DB and store to a dic or to pandas dataframe"""

    """
    Function
    --------
    pd_load_ratings

    Returns
    --------
    A pandas data frame conatining all the user ratings in a given context:
        ex =>
           condition      context       movie_title    rating      user_email
           Weekend         Time          Spark           3           foo@bar.com
    """
    @classmethod
    def load_pd_ratings_data(self):
        store = Datastore()
        df_data = []
        store = Datastore()
        for rating in store.ratings():
            movie = rating.movie
            user = rating.user
            for rating_condition in rating.rating_conditions:
                condition = rating_condition.condition
                context = condition.context
                rating_data = {
                    'db_id': rating.id,
                    'movie_title': movie.title,
                    'user_email': user.email,
                    'user_id': user.id,
                    'movie_id': movie.id,
                    'context': context.name,
                    'condition': condition.name,
                    'rating': rating.score
                }
                df_data.append(rating_data)

        return pd.DataFrame(df_data)
    """
    Function
    --------
    pd_load_users

    Returns
    --------
    A pandas data frame conatining all the users data:
        ex =>
           email         gender       birthday      city_id
           f@ba.co         male          Date           3


    """
    @classmethod
    def load_pd_users_data(self):
        store = Datastore()
        df_data = []
        store = Datastore()
        for user in store.users():
            _user_data = {
                'db_id': user.id,
                'email': user.email,
                'gender': user.gender,
                'birthday': user.birthday,
                'city_id': user.city_id
            }
            df_data.append(_user_data)
        return pd.DataFrame(df_data)


    """
    Function
    --------
    pd_load_movies

    Returns
    --------
    A pandas data frame conatining all movie info

    """
    @classmethod
    def load_pd_movies_data(self):
        store = Datastore()
        df_data = []
        store = Datastore()

        for movie in store.movies():
            _movie_data = {
                'db_id': movie.id,
                'title': movie.title,
                'director': movie.director,
                'language': movie.language,
                'country': movie.country,
                'budget': movie.budget,
                'year': movie.year,
            }
            actors = []
            genres = []
            for actor in movie.actors:
                actors.append(actor.name)
            for genre in movie.genres:
                genres.append(genre.name)

            _movie_data['actors'] = actors
            _movie_data['genres'] = genres
            df_data.append(_movie_data)
        return pd.DataFrame(df_data)

    """
    Function
    --------
    load_ratings

    Returns
    --------
    A dictionary of ratings
        key is the movie title.
        ex:
        "Game of thrones": {
                "natia@example": {
                    "Location": [(condition, score), (condition, score), (condition, score)],
                    "Time": [(condition, score), (condition, score), condition, score])
                    "Company": (condition, score), (condition, score), condition, score],
                },
                'l.pappas@example': {
                    "location": [()...],
                    "Time": [()...],
                    "Company": [()...]
                }
        }
    """
    @classmethod
    def load_ratings(self):
        _ratings = AutoVivification()
        store = Datastore()
        for rating in store.ratings():
            movie = rating.movie
            user = rating.user
            for rating_condition in rating.rating_conditions:
                condition = rating_condition.condition
                context = condition.context
                _ratings[movie.title][user.email][context.name] = _ratings[movie.title][user.email][context.name] or []
                _ratings[movie.title][user.email][context.name].append((condition.name, rating.score))
        return _ratings

    """
    Function
    --------
    load_user_data

    Returns
    --------
    A dictionary of user data
        key is the movie title.
        ex:
        "laertis.pappas@gmail.com": {
            'gender': 'male',
            'birthday': DateTime,
            'city': 3
        }
    """
    @classmethod
    def load_user_data(cls):
        _user_data = AutoVivification()
        store = Datastore()
        for user in store.users():
            _user_data[user.email]['gender'] = user.gender
            _user_data[user.email]['birthday'] = user.birthday
            _user_data[user.email]['city'] = user.city_id

        return _user_data

    """
    Function
    --------
    load_movie_data

    Returns
    --------
    A dictionary of movies data
        key is the movie title.
        ex:
        "Movie Title": {
            'genres': ['Romance', 'Drama'],
            'language': 'English',
            'country': 'United States',
            'budget': '25000000',
            'director': 'Kirsten Sheridan',
            'actors': ['Jonathan Rhys Meyers', u'Keri Russell', u'Freddie Highmore'],
            'year': 2007
        }
    """
    @classmethod
    def load_movie_data(cls):
        _movie_data = AutoVivification()
        store = Datastore()

        for movie in store.movies():
            _movie_data[movie.title]['director'] = movie.director
            _movie_data[movie.title]['language'] = movie.language
            _movie_data[movie.title]['country'] = movie.country
            _movie_data[movie.title]['budget'] = movie.budget
            _movie_data[movie.title]['year'] = movie.year
            _movie_data[movie.title]['actors'] = _movie_data[movie.title]['actors'] or []
            _movie_data[movie.title]['genres'] = _movie_data[movie.title]['genres'] or []
            for actor in movie.actors:
                _movie_data[movie.title]['actors'].append(actor.name)
            for genre in movie.genres:
                _movie_data[movie.title]['genres'].append(genre.name)
        return _movie_data