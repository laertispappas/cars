# Dataset is alresy imported in a POstgreSQL database
# This class is responsible for loading the models in
# the apropriate data structure for recommender

from app.entities.datastore import Datastore

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

class Loader(object):
    """Loads all ratings from DB and store to a dic"""
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
                "natia": {
                    "Location": [(condition, score), (condition, score), (condition, score)],
                    "Time": [(condition, score), (condition, score), condition, score])
                    "Company": (condition, score), (condition, score), condition, score],
                },
                'l.pappas': {
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

            # _ratings[movie.title]['ratings'][user.email] = _ratings[movie.title]['ratings'][user.email] or []
            # _ratings[movie.title]['ratings'][user.email].append((rating.score, context.name, condition.name))
            # _ratings[movie_title]['user_email'] = rating.user.email
            # _ratings[movie_title]['meta']['actors'] = _ratings[movie_title]['meta']['actors'] or ()
            # for actor in rating.movie.actors:
            #     if actor.name not in _ratings[movie_title]['meta']['actors']:
            #         _ratings[movie_title]['meta']['actors'] += (actor.name,)
            # _ratings[movie_title]['user']['email'] = rating.user.email
            # _ratings[movie_title]['user']['gender'] = rating.user.gender
            # _ratings[movie_title]['user']['birthday'] = rating.user.birthday
            #
            #
            #
        return _ratings

    @classmethod
    def load_user_data(cls):
        _user_data = AutoVivification()
        store = Datastore()
        for user in store.users():
            _user_data[user.email]['gender'] = user.gender
            _user_data[user.email]['birthday'] = user.birthday
            _user_data[user.email]['city'] = user.city_id

        return _user_data

    @classmethod
    def load_movie_data(cls):
        _movie_data = AutoVivification()
        store = Datastore()

        for movie in store.movies():
            _movie_data[movie.title]['director'] = movie.director
            _movie_data[movie.title]['language'] = movie.language
            _movie_data[movie.title]['country'] = movie.country
            _movie_data[movie.title]['year'] = movie.year
            _movie_data[movie.title]['actors'] = _movie_data[movie.title]['actors'] or []
            _movie_data[movie.title]['genres'] = _movie_data[movie.title]['genres'] or []
            for actor in movie.actors:
                _movie_data[movie.title]['actors'].append(actor.name)
            for genre in movie.genres:
                _movie_data[movie.title]['genres'].append(genre.name)
        return _movie_data