from collections import Counter

from app.algorithm.cars.context_recommender import ContextRecommender
from app.algorithm.cars.info_gain.plotter import Plotter
from app.dataset.data_object import DataObject
from app.dataset.loader import AutoVivification
from app.algorithm.cars.info_gain.evaluation import precision, KFold
from app.algorithm.cars.info_gain.evaluation import recall
from app.algorithm.cars.info_gain.evaluation import f1score
import numpy

import os

from app.utils.similarities import sim_euclidean, sim_pearson

RATING = 0 # Holds the index of rating in user preference array
OPTIMUM = 3.5 # Optimum rating for recommendation > 3.5
VERBOSE = True

class InfoGainRecommender(ContextRecommender):
    def __init__(self, data_object):
        super(self.__class__, self).__init__(data_object)
        self.filters = [(8, 3), (5, 2), (9, 1), (10, 1), (6, 2), (16, 2), (13, 2), (7, 1), (14, 2), (12, 1), (11, 2)]

    def init_model(self):
        self.userprofile = self.__build_user_profile()

    def build_model(self):
        # TODO: Remove after code cleanup
        self.set_training_set()

    def set_training_set(self, trainingSet = None):
        if trainingSet == None:
            self.training_data = self.dao.users
        else:
            self.training_data = trainingSet

    def generate_graphs(self):
        gain_user_ids = self.userprofile.keys()
        precision_metrics = {}
        for user in gain_user_ids:
            metric = self.evaluate(user)
            precision_metrics[user] = metric

        plotter = Plotter(precision_metrics)
        plotter.plot_precision_bar(type='precision')
        plotter.plot_precision_bar(type='recall')
        plotter.plot_num_of_recommendations()
        plotter.plot_precision_recall_curves()

    # TODO implement on concrete class
    def TraditionalRecommendation(self, user, simMeasure=sim_pearson, nNeighbors=None, topN=10):
        return self.__user_cf_recs(user, simMeasure, nNeighbors, topN)

    def PostFilteringRecommendation(self, user, simMeasure=sim_pearson, nNeighbors=None, topN=10):
        # TODO: remove all 999999999
        recs = self.__user_cf_recs(user, simMeasure, nNeighbors, topN=99999999999)
        return self.__contextual_filter(user, recs, topN)[0:topN]


    def evaluate(self):
        KFold(self.training_data, self)

    # Returns top N context aware recommendations for the given user.
    #
    def top_recommendations(self, user, N = 10):
        recs = self.__user_cf_recs(user)

        # Context - Value tuple
        filter_recs = self.__contextual_filter(user, recs, self.filters, topN=N)
        return filter_recs[1:N]

    def getPredictedRating(self, user, item, nearestNeighbors):
        if item in self.training_data[user].keys():
            return self.training_data[user][item]

        movie_ratings = []
        for movie_details in self.training_data[user].values():
            movie_ratings.append(movie_details[RATING])
        meanRating = numpy.mean(movie_ratings)

        weightedSum = 0
        normalizingFactor = 0
        for neighbor, similarity in nearestNeighbors.items():
            if item not in self.training_data[neighbor]:
                continue
            neighborsRatings = []
            for movie_details in self.training_data[neighbor].values():
                neighborsRatings.append(movie_details[RATING])
            meanRatingOfNeighbor = numpy.mean(neighborsRatings)

            weightedSum += similarity * (self.training_data[neighbor][item][RATING] - meanRatingOfNeighbor)
            normalizingFactor += numpy.abs(similarity)
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)

    def __user_cf_recs(self, user, similarity=sim_pearson, nNeighbors=50, topN=10):
        predictedScores = []
        similarities = self.getNearestNeighbors(user, similarity, nNeighbors)
        for item in self.dao.movies.keys():
            if item in self.training_data[user]:
                continue
            itemRaters = {}  # Nearest neighbors who rated on the item
            for similarity, neighbor in similarities:
                if similarity <= 0 or len(itemRaters) == nNeighbors:
                    break
                if item in self.training_data[neighbor].keys():
                    itemRaters[neighbor] = similarity
            predicted_rating = self.getPredictedRating(user, item, itemRaters)
            predictedScores.append((predicted_rating, item))
        predictedScores.sort(reverse=True)

        return predictedScores[0:topN]

    def getNearestNeighbors(self, target, simMeasure, nNeighbors=None):
        prefs = self.training_data
        # sim = similarity(prefs, person, other)
        similarities = [(simMeasure(prefs, target, other), other) for other in prefs if target != other]
        similarities.sort(reverse=True)
        if nNeighbors != None:
            similarities = similarities[0:nNeighbors]
        return similarities  # similarities = [(similarity, neighbor), ...]


    # Private Methods
    """
    Function
    ---------------------
    __build_user_profile
    ---------------------
    Returns
        Builds a dictionary of User Profiles by parsing through the info gain file
        => {user1: [c1, c2, c5, c6]}

    filename: WEKA output from InfoGainAttributeEval:
        Evaluates the worth of an attribute by measuring the information gain with respect to the class.
    """
    def __build_user_profile(self, filename='InfoGainResults.txt'):
        dir = os.path.dirname(__file__)
        src_path = os.path.join(dir, filename)

        userprofile = {}
        with open(src_path) as f:
            for line in f:
                line = line.strip()
                row = line.split(',')
                key, val = eval(row[0]), [DataObject.feature_dict[x] for x in row[1:]]
                userprofile[key] = val
        return userprofile

    # Given udb datastore return 1/3 as a test dataset and 2/3 as train dataset
    def __remove_for_testing(self, udb, user):
        limit = int(len(udb[user]) / 3.0)
        test_udb = AutoVivification()
        import copy;
        train_udb, i = copy.deepcopy(udb), 0
        for movie in udb[user]:
            test_udb[user][movie] = train_udb[user][movie]
            del (train_udb[user][movie])
            i += 1
            if i > limit: break
        # print len(train_udb[user]), len(test_udb[user]), len(udb[user])
        return train_udb, test_udb
    """
    Function
    __contextual_filter
    Post filter recommendations generated by a traditional 2D recommender, by finding
    the maximum repeating context for high rating movie over all users.
    -----------------
        udb: user datastructure
            => { user_id: { item_id: [rating,age,sex,city,country, c1, c2, ................, c12] } }
        profiles: User Profile built based on Information gain of Contextual attributes
            => {user1: [c1, c2, c5, c6]}
        user: Target User
            => 15
        recommendations: Recommendations generated by a traditions recommender
            => [(rating, item_id)]
        filters: User entered Contextual Attributes
            =>  [(8, 2), (5, 1), (9, 2), (10, 2), (6, 1), (16, 1), (13, 1), (7, 1), (14, 1), (12, 1), (11, 1)]
    Returns
        User post filtered recommendations
            => [(rating, movie), ...]
    """
    def __contextual_filter(self, user, recommendations, topN=10):
        filtered_recs = []
        for rating, movie in recommendations:
            if rating >= OPTIMUM:
                for context, filter_ctx_value in self.filters:
                    if context in self.userprofile[user]:
                        max_ctx_value = float(self.__find_max_context(movie, context, self.training_data))
                        # Filter recommendations based on the maximum context condition provided for this movie
                        # from all other users. If the condition is not the same as the filtered one
                        # we will reject the movie.
                        if max_ctx_value == filter_ctx_value and (rating, movie) not in filtered_recs:
                            filtered_recs.append((rating, movie))
        filtered_recs.sort(reverse=True)
        return filtered_recs[0:topN]

    def __contextual_filterNeighborCtxPropability(self, user, recommendations, topN=10):
        # Relevance of item i for target user u in a particular context c
        # is approximated by the propability Pc(u,i,c) = |Uu,i,c| / k where k is the number
        # of neighbors used by kNNand Uu,i,c = { v in N(u)|Rv,i,c != 0} that is the user's neighbor v
        # in the neighborhood of u, N(u) who have rated / consumed item i in context c. The item relevance
        # is determined by the threshold value tpc (0.1) that is used to contextualize the ratings as follows:
        #
        # F(u,i,c) = F(u,i)       if Pc(u,i,c) >= tpc
        #          = F(u,i) -0.5  if Pc(u,i,c) < tpc
        #
        # where F(u,i) denotes the context-unaware rating prediction by RS and F(u,i,c) denotes
        # the context-aware rating prediction.
        #
        filtered_recs = []
        nNeighbors = 50
        similarities = self.getNearestNeighbors(user, sim_pearson, nNeighbors) #[(similarity, neighbor)]
        tpc = 0.08

        for rating, movie in recommendations:
            nNeighbors_rated_item_in_same_context = 0
            for sim, neighbor in similarities:
                for context, filter_ctx_value in self.filters:
                    if movie in self.training_data[neighbor] and context in self.training_data[neighbor][movie]:
                        nNeighbors_rated_item_in_same_context += 1
            puic = float(nNeighbors_rated_item_in_same_context) / float(nNeighbors)
            if puic >= tpc:
                filtered_recs.append((rating, movie))
            else:
                filtered_recs.append((rating - 0.25, movie))

        filtered_recs.sort(reverse=True)
        return filtered_recs[0:topN]

    def __find_max_context(self, movie, context, udb):
        """
        finds the maximum repeating context for a high rating movie, over all users
        :param movie:
        :param context:
        :param udb:
        :return: Maximum repearing context if found else -1
        """
        list_context = []
        # For each user
        for user in udb:
            # If the current user has rated the movie and the rating is greater that OPTIMUM
            if movie in udb[user] and udb[user][movie][RATING] >= OPTIMUM:
                # Get the context value where the current user has rated this movie
                # and append it to list_context.
                filter_ctx_value = udb[user][movie][context]
                if type(filter_ctx_value) == numpy.float64:
                    list_context.append(udb[user][movie][context])
        # Get the maximum context value found
        if len(list_context) > 1:
            m = max(k for k, v in Counter(list_context).items())
        else:
            m = -1
        return m