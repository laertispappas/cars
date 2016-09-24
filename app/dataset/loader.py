# Dataset is alresy imported in a POstgreSQL database
# This class is responsible for loading the models in
# the apropriate data structure for recommender

from app.entities.datastore import Datastore
import pandas as pd

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