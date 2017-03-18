import os, sys

sys.path.insert(0, os.path.abspath(".."))

from app.algorithm.cars.info_gain.evaluation import evaluate
from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

import argparse

def main():
    if len(sys.argv) < 4:
        print "Usage: python app.py userID ContextID ContextCondtionID"
        exit()

    user = int(sys.argv [1])
    context = int(sys.argv [2])
    context_condition = int(sys.argv [3])

    data_object = DataObject()
    # data_object.print_specs()
    # data_object.plot_stats()

    recommender = InfoGainRecommender(data_object)
    recommender.run()
    recommender.filters = [(int(context), int(context_condition))]
    recommendations = recommender.top_recommendations(int(user))

    print "Predicted movies for user: ", 193
    print "Movie\t\tPrediction"
    print "******************************"
    for pred_rating, movie in recommendations:
        print movie, "\t\t" ,pred_rating
if __name__ == "__main__": main()