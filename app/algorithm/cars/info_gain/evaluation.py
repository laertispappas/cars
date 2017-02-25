from app.dataset.data_object import DataObject
from app.dataset.loader import AutoVivification
from app.utils.similarities import sim_pearson

RATING = 0 # Holds the index of rating in user preference array
OPTIMUM = 3.5 # Optimum rating for recommendation > 3.5

from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

def evaluateRecommender(testSet, trainSet, recommender, simMeasure=None, nNeighbors=None, topN=None, type='2d'):
    result = AutoVivification()

    recommender.set_training_set(trainSet)
    # Evaluation metrics
    totalPrecision = 0
    totalRecall = 0
    totalF1score = 0
    totalHit = 0

    for user in recommender.dao.users:
        if type == '2d':
            recommendation = recommender.TraditionalRecommendation(user, simMeasure, nNeighbors, topN)
        else:
            recommendation = recommender.PostFilteringRecommendation(user, simMeasure, nNeighbors, topN)
        hit = 0
        for item in testSet[user].keys():
            for rating, recommended_item in recommendation:
                if recommended_item == item:
                    hit += 1
                    break
        precision = float(hit) / float(topN)
        recall = 0 if len(testSet[user].keys()) == 0 else float(hit) / (len(testSet[user].keys()))
        f1score = 0 if hit == 0 or precision + recall == 0 else float(2 * precision * recall / (precision + recall))

        totalPrecision += precision
        totalRecall += recall
        totalF1score += f1score
        totalHit += hit

    result["Precision"] = float(totalPrecision / (len(testSet)))
    result["Recall"] = float(totalRecall / len(testSet))
    result["F1-score"] = float(totalF1score / len(testSet))
    result["Hit-rate"] = float(totalHit) / len(testSet)
    return result

"""
TBD: broad classes of prediction accuracy measures; measuring
the accuracy of ratings predictions, measuring the accuracy of usage predictions,
and measuring the accuracy of rankings of items.

Measuring Ratings Prediction Accuracy:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Normalized RMSE (NMRSE)
    - Average RMSE
Measuring Usage Prediction:
    - Precision
    - Recall
    - False Positive Rate
    - F-measure
    - Area Under the ROC Curve (AUC)
    - Precision-Recall and ROC for Multiple Users
Ranking Measures
    - Normalized Distance based Performance Measure (NDPM)
    - average precision (AP) correlation
    - Utility-Based Ranking
    - (n)DCG


"""


def KFoldSplit(data, fold, nFolds):  # fold: 0~4 when 5-Fold validation
    trainSet = deepcopy(data)  # data = {user: {item: rating, ...}, ...}
    testSet = {}
    for user in data:
        testSet.setdefault(user, {})
        unitLength = int(len(data[user].keys()) / nFolds)  # data[user] = {item: rating, ...}
        lowerbound = unitLength * fold
        upperbound = unitLength * (fold + 1) if fold < nFolds - 1 else len(data[user])
        testItems = {}
        for i, item in enumerate(data[user].keys()):
            if lowerbound <= i and i < upperbound:
                testItems[item] = trainSet[user][item]
                # delete the item from the training set
                del (trainSet[user][item])
        testSet[user] = testItems
    return trainSet, testSet


def KFold(data, recommender, simMeasure=sim_pearson, nNeighbors=50, topN=10, nFolds=4):
    result = AutoVivification()
    start_time = datetime.now()

    for type in ['2d', 'ctx']:
        # Evaluation metrics
        totalPrecision = 0
        totalRecall = 0
        totalF1score = 0
        totalHitrate = 0

        for fold in range(nFolds):
            trainSet, testSet = KFoldSplit(data, fold, nFolds)

            evaluation = evaluateRecommender(testSet, trainSet, recommender, simMeasure=simMeasure, nNeighbors=nNeighbors, topN=topN, type=type)
            totalPrecision += evaluation["Precision"]
            totalRecall += evaluation["Recall"]
            totalF1score += evaluation["F1-score"]
            totalHitrate += evaluation["Hit-rate"]

            # del (trainSet)
            # del (testSet)
        # Find final results
        result[type]["Precision"] = totalPrecision / nFolds
        result[type]["Recall"] = totalRecall / nFolds
        result[type]["F1-score"] = totalF1score / nFolds
        result[type]["Hit-rate"] = float(totalHitrate) / nFolds
    print("Execution time: {}".format(datetime.now() - start_time))

    print "******** TOP ", str(topN), " ******"
    print result['2d']['F1-score']
    print result['ctx']['F1-score']
    print "Next"

    # plot_results(result, type='F1-score')
    return result


feature_dict = {
    'rating': 0, 'age': 1, 'sex': 2, 'city': 3,
    'country': 4, 'time': 5, 'daytype': 6, 'season': 7,
    'location': 8, 'weather': 9, 'social': 10,
    'endEmo': 11, 'dominantEmo': 12, 'mood': 13,
    'physical': 14, 'decision': 15, 'interaction': 16
}

ContextMappings = {
    '5': 'time',
    '6': 'daytype',
    '7': 'season',
    '10': 'social'
}

ContextConditionMappings = {
    '5': ['','Morning', 'Afternoon', 'Evening', 'Night'],
    '6': ['','Working day', 'Weekend', 'Holiday'],
    '7': ['','Spring', 'Summer', 'Autumn', 'Winter'],
    '8': ['','Home', 'Public place', "Friend's house"],
    '9': ['','Sunny / clear', 'Rainy', 'Stormy', 'Snowy', 'Cloudy'],
    '10': ['','Alone', 'My partner', 'Friends', 'Colleagues', 'Parents', 'Public', 'My family'],
    '11': ['','Sad', 'Happy', 'Scared', 'Surprised', 'Angry', 'Disgusted', 'Neutral'],
    '12': ['', 'Sad', 'Happy', 'Scared', 'Surprised', 'Angry', 'Disgusted', 'Neutral'],
    '13': ['', 'Positive', 'Neutral', 'Negative'],
    '14': ['', 'Healthy', 'Ill'],
    '15': ['', 'User decided which movie to watch', 'User was given a movie'],
    '16': ['', 'first interaction with a movie', 'n-th interaction with a movie']
}

def plot_results(data, type=None):
    import numpy as np
    import matplotlib.pyplot as plt

    for context in data['ctx'].keys():
        labels = []
        baseline_metrics = (data['ctx'][context]['2d']['Precision'], data['ctx'][context]['2d']['Recall'], data['ctx'][context]['2d']['F1-score'])
        contextual_metrics = AutoVivification()

        labels.extend(['Baseline', 'Baseline', 'Baseline'])

        for condition in data['ctx'][context].keys():
            if condition == '2d':
                continue
            contextual_metrics[condition] = (data['ctx'][context][condition]['Precision'],
                                             data['ctx'][context][condition]['Recall'],
                                             data['ctx'][context][condition]['F1-score'])

            label_name = ContextConditionMappings[context][int(condition)]
            labels.extend([label_name, label_name, label_name])
        print baseline_metrics
        print contextual_metrics

        fig, ax = plt.subplots()
        index = np.arange(3)
        bar_width = 0.15
        opacity = 0.8

        rects1 = plt.bar(index, baseline_metrics, bar_width,
                         alpha=opacity,
                         color='b',
                         label='Baseline')
        colors = {
            '1': 'r',
            '2': 'g',
            '3': 'y',
            '4': 'c',
            '5': 'm',
            '6': 'k',
            '7': '#ee1f3f',
            '8': '#123ef2'
        }
        i = 1
        for condition in contextual_metrics.keys():
            plt.bar(index + bar_width * i, contextual_metrics[condition], bar_width,
                    alpha=opacity,
                    color=colors[condition],
                    label=ContextConditionMappings[context][int(condition)])
            i += 1

        rects = ax.patches
        # Now make some labels
        # labels = ["label%d" % i for i in xrange(len(rects))]
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')

        # plt.xlabel('Metric')
        # plt.ylabel('Value')
        plt.title('Evaluation Metrics for ' + ContextMappings[context] + ' context')
        plt.xticks(index + bar_width + 0.15, ('Precision', 'Recall', 'F1Score'))
        plt.legend()
        plt.tight_layout()
        plt.show()

def evaluate():
    from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender
    context_conditions = {
        '5': range(1, 5),
        '6': range(1, 4),
        '7': range(1, 5),
        '10': range(1, 5),
    }
    result = AutoVivification()
    for context in context_conditions.keys():
        for condition in context_conditions[context]:
            print context
            print condition
            for topN in range(1, 11):
                data_object = DataObject()
                recommender = InfoGainRecommender(data_object)
                recommender.run()
                recommender.filters = [(int(context), condition)]
                current_results = KFold(recommender.training_data, recommender, topN=topN)
                result["top-" + str(topN)]['ctx'][str(context)]['2d'] = current_results['2d']
                result["top-" + str(topN)]['ctx'][str(context)][str(condition)] = current_results['ctx']

    filename = "roc_hybrid_filter_weight_all_results.json"
    results_to_json(result, filename)
    plot_results(result)

def results_to_json(data, filename):
    import json
    with open(filename, 'w') as fp:
        json.dump(data, fp)
