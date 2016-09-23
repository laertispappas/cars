from data.data_reader import DataReader

class Recommender(object):
    def __init__(self):
        self.data_object = DataReader().load()
        self.rating_matrix = self.data_object.get_rating_matrix()
        self.test_matrix = self.data_object.get_test_matrix()
        self.train_matrix = self.data_object.get_train_matrix()
        self.num_conditions = self.data_object.number_of_conditions()
        self.num_users = self.data_object.number_of_users()
        self.num_movies = self.data_object.number_of_movies()

        # TODO Remove hardcoded values
        self.global_mean = 3.5
        self.rating_scale = [0, 1, 2, 3, 4, 5] #self.data_object.getRatingScale();
        self.min_rate = 1 #self.data_object.ratingScale.get(0);
        self.max_rate = 5 #self.data_object.ratingScale.get(ratingScale.size() - 1);
        self.num_levels = 5 #self.data_object.ratingScale.size();

        self.init_mean = 0.0
        self.init_stf = 0.1

        self.is_ranking_pred = False
        self.num_recs = 5
        self.knn = 20
        self.similarity_measure = 'COS'