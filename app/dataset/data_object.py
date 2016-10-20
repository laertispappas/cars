from app.dataset.loader import Loader

class DataObject(object):
    def __init__(self):
        self.ratings = Loader().load_pd_ratings_data()
        self.users = Loader().load_pd_users_data()
        self.movies = Loader().load_pd_movies_data()

    def print_specs(self):
        print "Num of ratings", self.total_ratings()
        print "Num of users", self.total_users()
        print "Num of movies", self.total_movies()
        print "******"
        print "Total context conditions: ", self.total_context_conditions(), self.context_conditions()
        print "Total context dimensions: ", self.total_context_dimensions(), self.context_dimensions()

    def total_ratings(self):
        return len(self.ratings)

    def total_movies(self):
        return len(self.movies)

    def total_users(self):
        return len(self.users)

    """
    Function
    --------
    total_conditions

    Returns
    -------
    Total number of context conditions
    """
    def total_context_conditions(self):
        return len(self.ratings.condition.unique())
    def context_conditions(self):
        return self.ratings.condition.unique()

    """
    Function
    --------
    total_dimensions

    Returns
    -------
    Total number of context dimensions
    """
    def total_context_dimensions(self):
        return len(self.ratings.context.unique())
    def context_dimensions(self):
        return self.ratings.context.unique()