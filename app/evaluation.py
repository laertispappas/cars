import os, sys

sys.path.insert(0, os.path.abspath(".."))

from app.algorithm.cars.info_gain.evaluation import evaluate
from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender

import argparse

def main():
    evaluate()
if __name__ == "__main__": main()