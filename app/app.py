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
    recommender.generate_graphs()
    # recommender.evaluate(193)
    # print recommender.top_recommendations(193)
if __name__ == "__main__": main()