import os, sys

sys.path.insert(0, os.path.abspath(".."))

from app.algorithm.cars.info_gain.evaluation import evaluate
from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

def main():
    data_object = DataObject()
    # data_object.print_specs()
    # data_object.plot_stats()

    recommender = InfoGainRecommender(data_object)
    recommender.run()
    # print recommender.top_recommendations(100)
    evaluate()

    # print recommender.top_recommendations(35)
if __name__ == "__main__": main()