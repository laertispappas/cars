import os, sys
sys.path.insert(0, os.path.abspath(".."))

from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

def main():
    data_object = DataObject()
    recommender = InfoGainRecommender(data_object)
    recommender.run()
if __name__ == "__main__": main()