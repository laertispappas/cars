import os, sys

sys.path.insert(0, os.path.abspath(".."))

from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

import argparse

def print_usage():
    print """
        Usage:
            (Get a list of user ids):           python app.py list_users
            (Get a list of context ids)         python app.py list_context
            (postFiltering recommendations)     python app.py recommend userID contextID dimensionID
            (Evaluate recommenders)             python evaluate.py
    """

def main():
    if len(sys.argv) == 1:
        print_usage()
        exit()
    if sys.argv[1] == "list_users":
        data_object = DataObject()
        print data_object.users.keys()
        exit()
    if sys.argv[1] == "list_context":
        data_object = DataObject()
        print data_object.feature_dict
        exit()
    if len(sys.argv) < 5:
        print_usage()
        exit()

    user = int(sys.argv [2])
    context = sys.argv [3]
    context_condition = sys.argv [4]

    data_object = DataObject()
    # data_object.print_specs()
    # data_object.plot_stats()

    recommender = InfoGainRecommender(data_object)
    recommender.run()
    recommender.filters = [(context, context_condition)]
    recommendations = recommender.top_recommendations(user)

    print "Predicted movies for user: ", user
    print "Movie\t\tPrediction"
    print "******************************"
    for pred_rating, movie in recommendations:
        print movie, "\t\t" ,pred_rating
if __name__ == "__main__": main()