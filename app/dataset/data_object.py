from app.dataset.loader import Loader

class DataObject(object):
    NON_CONTEXT_ATTRS = ['userID','itemID','rating','age','sex','city','country', 'director', 'movieCountry',
                              'movieLanguage', 'movieYear', 'genre1', 'genre2', 'genre3', 'actor1', 'actor2','actor3',
                              'budget']
    def __init__(self):
        self.ratings, self.users, self.movies = Loader().load_ldos_csv()
        self.total_ratings = self.__total_ratings()
        self.total_movies = self.__total_movies()
        self.total_users = self.__total_users()

        self.context_types, self.total_context_types = self.__get_contexts()

    def print_specs(self):
        print "Num of ratings", self.total_ratings
        print "Num of users", self.total_users
        print "Num of movies", self.total_movies
        print "******"
        print 'Context types: ', self.context_types
        print "Total Context Types: ", self.total_context_types

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