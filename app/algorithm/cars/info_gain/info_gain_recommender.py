from app.algorithm.cars.context_recommender import ContextRecommender
from app.algorithm.cars.info_gain.plotter import Plotter
from app.dataset.data_object import DataObject
from app.dataset.loader import AutoVivification
from app.algorithm.cars.info_gain.evaluation import precision
from app.algorithm.cars.info_gain.evaluation import recall
from app.algorithm.cars.info_gain.evaluation import f1score
import numpy

import os

from app.utils.similarities import sim_euclidean

RATING = 0 # Holds the index of rating in user preference array
OPTIMUM = 3.5 # Optimum rating for recommendation > 3.5
VERBOSE = False

class InfoGainRecommender(ContextRecommender):
    def __init__(self, data_object):
        super(self.__class__, self).__init__(data_object)
        self.filters = [(8, 2), (5, 1), (9, 2), (10, 2), (6, 1), (16, 1), (13, 1), (7, 1), (14, 1), (12, 1), (11, 1)]

    def init_model(self):
        self.userprofile = self.__build_user_profile()

    def build_model(self):
        pass

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

    def evaluate(self, user):
        metrics = {}

        train_udb, test_udb = self.__remove_for_testing(self.dao.users, user)
        recs = self.__user_cf_recs(train_udb, user)

        precision_train = precision(user, recs, test_udb)
        recall_train = recall(user, recs, test_udb)
        fscore_train = f1score(precision_train, recall_train)

        if VERBOSE:
            print "****************************************************************************************************"
            print "CF precision: ", precision_train
            print "CF recall: ", recall_train
            print "CF f1Score: ", f1score(precision_train, recall_train)
            print "CF total: ", len(recs)

        filter_recs = self.__contextual_filter(self.dao.users, self.userprofile, user, recs, self.filters)

        precision_test = precision(user, filter_recs, test_udb)
        recall_test = recall(user, filter_recs, test_udb)
        fscore_test = f1score(precision_test, recall_test)
        if VERBOSE:
            print precision_test, recall_test, f1score(precision_test, recall_test), len(filter_recs)
            print "Filtered precision: ", precision_test
            print "Filtered recall: ", recall_test
            print "Filtered f1Score: ", f1score(precision_test, recall_test)
            print "Filtered total: ", len(filter_recs)
            print "****************************************************************************************************"
        metrics['precision'] = (precision_train, precision_test)
        metrics['recall'] = (recall_train, recall_test)
        metrics['f1score'] = (fscore_train, fscore_test)
        metrics['total_recs'] = len(recs)
        metrics['total_ctx_recs'] = len(filter_recs)
        return metrics

    # Returns top N context aware recommendations for the given user.
    #
    def top_recommendations(self, user, N = 10):
        recs = self.__user_cf_recs(self.dao.users, user)

        # Context - Value tuple
        filter_recs = self.__contextual_filter(self.dao.users, self.userprofile, user, recs, self.filters)
        return filter_recs[1:N]

    # Gets recommendations for a person by using a weighted average
    # of every other user's rankings User based CF
    #
    def __user_cf_recs(self, prefs, person, similarity=sim_euclidean):
        totals = {}
        simSum = {}
        for other in prefs:
            if other == person: continue
            sim = similarity(prefs, person, other)
            if sim <= 0: continue
            for item in prefs[other]:
                if item not in prefs[person] or prefs[person][item][RATING] == 0:
                    totals.setdefault(item, 0)
                    totals[item] += prefs[other][item][RATING] * sim
                    # Similarity sums
                    simSum.setdefault(item, 0)
                    simSum[item] += sim
        rankings = [(total / simSum[item], item) for item, total in totals.items()]
        # Testing: Checking if ratings match with that in dataset
        rankings.sort()
        rankings.reverse()
        return rankings

    # TODO Add Item Based CF

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
    def __contextual_filter(self, udb, profiles, user, recommendations, filters):
        filtered_recs = []
        for rating, movie in recommendations:
            if rating >= OPTIMUM:
                for context, filter_ctx_value in filters:
                    if context in profiles[user]:
                        max_ctx_value = float(self.__find_max_context(movie, context, udb))
                        if max_ctx_value == filter_ctx_value and (rating, movie) not in filtered_recs:
                            filtered_recs.append((rating, movie))
        return filtered_recs

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
                # Get the context value where the user has rated this movie
                # and append it to list_context.
                filter_ctx_value = udb[user][movie][context]
                if type(filter_ctx_value) == numpy.float64:
                    list_context.append(udb[user][movie][context])
        # Get the maximum context value found
        if len(list_context) > 1:
            m = max(list_context)
        else:
            m = -1
        return m