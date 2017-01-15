import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

def main():
    data_object = DataObject()
    # data_object.print_specs()
    # data_object.plot_stats()

    recommender = InfoGainRecommender(data_object)
    recommender.run()
    # recommender.test_user_cf2(61)
    recommender.generate_graphs()
    # recommender.evaluate(33)
    # print "next\n"
    # recommender.evaluate(55)
    # print "next\n"
    # recommender.evaluate(33)


    # print recommender.top_recommendations(35)
if __name__ == "__main__": main()