import os, sys

sys.path.insert(0, os.path.abspath(".."))

from app.algorithm.cars.info_gain.evaluation import PrecisionRecommenderEvaluator
from app.algorithm.cars.info_gain.evaluation import __evaluate
from app.algorithm.cars.info_gain.evaluation import plot_results
from app.algorithm.cars.info_gain.evaluation import results_to_json
from app.algorithm.cars.info_gain.plotter import Plotter

from app.dataset.data_object import DataObject
from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender
from app.dataset.loader import AutoVivification
import argparse

context_conditions = {
    'Time': ['Weekday', 'Weekend'],
    'Location': ['Home', 'Cinema'],
    'Companion': ['Alone', 'Family', 'Partner'],
}
result = AutoVivification()

def plot_results():
    import json
    with open('precisionTopNEvaluation.json', 'r') as fp:
        data = json.load(fp)
    plotter = Plotter(data)
    plotter.plot()
    plotter.plotf1Score()

def main():
    plot_results()
    exit()
    # __evaluate()
    irStats = {}
    for context in context_conditions.keys():
        for condition in context_conditions[context]:
            # for context2 in context_conditions.keys():
            #     if context2 == context:
            #         continue
            #     for condition2 in context_conditions[context2]:
            #         for context3 in context_conditions.keys():
            #             if context3 == context2 or context3 == context:
            #                 continue
            #             for condition3 in context_conditions[context3]:
            #                 print condition, condition2, condition3
            #
            #                 for topN in range(1, 2):
            #                     data_object = DataObject()
            #                     evaluator = PrecisionRecommenderEvaluator()
            #                     recommender = InfoGainRecommender(data_object)
            #                     recommender.run()
            #                     # recommender.filters = [('Time', 'Weekend')]
            #                     recommender.filters = [(context, condition), (context2, condition2), (context3, condition3)]
            #                     result = evaluator.evaluate(recommender, data_object, 10, 1.0, 3.5)
            #                     print "******** TOPN=", str(topN), " ******"
            #                     print "2D f1Score: ", result['f1Score']
            #                     # print "2D precision: ", result['precision']
            #                     # print "2D recall: ", result['recall']
            #
            #                     print "Contextual f1score:", result['f1Score-ctx']
                                # print "Contextual Precision:", result['precision-ctx']
                                # print "Contextual Recall:", result['recall-ctx']
                                # print result
                                # print "Context: ", context
            print context, condition
            for topN in range(10, 11):
                irStats[topN] = {}
                irStats[topN][context] = irStats[topN][context] or {}
                irStats[topN][context][condition] = irStats[topN][context][condition] or {}
                data_object = DataObject()
                evaluator = PrecisionRecommenderEvaluator()
                recommender = InfoGainRecommender(data_object)
                recommender.run()
                recommender.filters = [(context, condition)]
                result = evaluator.evaluate(recommender, data_object, topN, 1.0, 3.5)
                print "******** TOPN=", str(topN), " ******"
                # print "2D f1Score: ", result['f1Score']
                # print "2D precision: ", result['precision']
                # print "2D recall: ", result['recall']

                # print "Contextual f1score:", result['f1Score-ctx']
                # print "\n\n"
                # print "Contextual Precision:", result['precision-ctx']
                # print "Contextual Recall:", result['recall-ctx']
                print result
                irStats[topN][context][condition] = result
    filename = "rangeTopNEvaluation.json"
    results_to_json(irStats, filename)
    plot_results(result)

if __name__ == "__main__": main()