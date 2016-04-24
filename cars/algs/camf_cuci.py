from algs.recommender import Recommender
import numpy as np
import pandas as pd

class CAMF_CUCI(Recommender):
    NUM_FACTORS = 10
    NUM_ITERATIONS = 100

    def __init__(self):
        super(self.__class__, self).__init__()
        self.algo_name = "CAMF_CUCI"
        
        # user, item and bias regularizations
        self.reg_u = 0.001
        self.reg_i = 0.001
        self.reg_b = 0.001
        self.reg_c = 0.001

        # TODO: Init with Randoms.gaussian
        # Init P and Q vectors with random values
        #self.P = np.random.rand(self.num_users, self.NUM_FACTORS);
        #self.Q = np.random.rand(self.num_movies, self.NUM_FACTORS);
        self.P=pd.DataFrame(np.random.randn(self.num_users, self.NUM_FACTORS), 
                        index=self.rating_matrix.user_id.unique())
        self.Q=pd.DataFrame(np.random.randn(self.num_movies, self.NUM_FACTORS), 
                        index=self.rating_matrix.movie_id.unique())

        # user, item - context biases
        self.uc_bias = {}
        self.ic_bias = {}

        """
        Init user and item context biases
            uc_bias = {
                "0": {
                    "0": 0.11,
                    "1": 0.21
                    ....
                }
            }
        """
        for u in self.rating_matrix.user_id.unique():
            for c in range(self.num_conditions):
                # mu, sigma = 0, 0.1
                # s = np.random.normal(mu, sigma, 1000)
                rand_value = np.random.randn()
                self.uc_bias.setdefault(u, {c: rand_value})[c] = rand_value
        for j in self.rating_matrix.movie_id.unique():
            for c in range(self.num_conditions):
                rand_value = np.random.randn()
                self.ic_bias.setdefault(j, {c: rand_value})[c] = rand_value
        
        print self.uc_bias

    def build_model(self):
        print "*** Building model ***", self.algo_name

        for iter in range(1, self.NUM_ITERATIONS + 1):
            loss = 0
            for index, row in self.train_matrix.iterrows():
                user_id = row['user_id']
                movie_id = row['movie_id']
                social_context_condition = row['social']
                contextual_rating = row['rating']

                prediction = self.predict(user_id, movie_id, social_context_condition)
                error = contextual_rating - prediction
                loss += error * error

                # TODO: Update user and item context biase factors
                # --- ----#
            loss *= 0.5

    def predict(self, user_id, movie_id, context):
        prediction = self.global_mean + self.P.ix[user_id].dot(self.Q.ix[movie_id])
        # TODO: Add item and user context biases
        return prediction


