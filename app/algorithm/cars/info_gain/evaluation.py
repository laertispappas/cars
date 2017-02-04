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

        # result["users"][user]["Precision"] = precision
        # result["users"][user]["Recall"] = recall
        # result["users"][user]["F1-score"] = f1score
        # result["users"][user]["Hit-rate"] = hit

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


def KFold(data, recommender, simMeasure=sim_pearson, nNeighbors=40, topN=10, nFolds=4):
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
            # result[type][fold] = evaluation["users"] # ctx | 2d: { foldNum:{ userId: { metrics } } }
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
    # print result

    # plot_metrics_bar_for_each_fold(result, nFolds, 'Precision')
    # plot_metrics_bar_for_each_fold(result, nFolds, 'Recall')
    # plot_metrics_bar_for_each_fold(result, nFolds, 'F1-score')
    # plot_metrics_bar_for_each_fold(result, nFolds, 'Hit-rate')

    # plot_avg_metrics_for_all_folds_per_user(result, nFolds, type='Precision')
    # plot_avg_metrics_for_all_folds_per_user(result, nFolds, type='Recall')
    # plot_avg_metrics_for_all_folds_per_user(result, nFolds, type='F1-score')
    # plot_avg_metrics_for_all_folds_per_user(result, nFolds, type='Hit-rate')

    print result['2d']['F1-score']
    print result['ctx']['F1-score']

    # plot_results(result, type='F1-score')

    filters = recommender.filters
    context = str(filters[0][0])
    condition = str(filters[0][1])

    filename = "top10_filter_results_" + context + "__" + condition + ".json"
    results_to_json(result, filename)

    return result


def evaluate():
    from app.algorithm.cars.info_gain.info_gain_recommender import InfoGainRecommender
    # feature_dict = {
    #     'time': 5,
    #     'daytype': 6,
    #     'location': 8,
    #     'social': 10,
    #     'endEmo': 11,
    #     'dominantEmo': 12,
    #     'mood': 13
    # }
    context_conditions = {
        '10': range(1, 5),
        '5': range(1, 5),
        '6': range(1, 4),
        '7': range(1, 5),
        '8': range(1, 4),
    }
    for context in context_conditions.keys():
        for condition in context_conditions[context]:
            print context
            print condition
            data_object = DataObject()
            recommender = InfoGainRecommender(data_object)
            recommender.run()
            recommender.filters = [(int(context), condition)]
            KFold(recommender.training_data, recommender)


def generate_next_context():
    context_conditions = {
        '10': range(1, 5),
        '5': range(1, 5),
        '6': range(1, 4),
        '7': range(1, 5),
        '8': range(1, 4),
    }

    def generate_pairs(context_conditions):
        pairs = []
        for context in context_conditions.keys():
            conditions = context_conditions[context]
            pairs.append((context, conditions))
        return pairs

    # [(context, [conditions]), (ctx2, [cond2])]
    pairs = generate_pairs(context_conditions)

    def __f(pairs):
        pass


def plot_results(data, type=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # data to plot
    n_groups = 4
    baseline = (90, 55, 40, 65)
    context1 = (85, 62, 54, 20)
    context2 = (100, 105, 154, 120)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, baseline, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Frank')

    rects2 = plt.bar(index + bar_width, context1, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Guido')

    rects3 = plt.bar(index + bar_width + bar_width, context2, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Me')

    plt.xlabel('Precision')
    plt.ylabel('Context')
    plt.title('Scores by person')
    plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    exit()

def results_to_json(data, filename):
    import json
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def plot_avg_metrics_for_all_folds_per_user(data, nFolds, type='Precision'):
    users_ids = data['2d'][0].keys()
    users_ids.sort()

    precisions = AutoVivification()
    ctx_precisions = AutoVivification()

    for fold in range(nFolds):
        for user in users_ids:
            precisions.setdefault(user, 0)
            ctx_precisions.setdefault(user, 0)
            precisions[user] += data['2d'][fold][user][type]
            ctx_precisions[user] += data['ctx'][fold][user][type]


    N = len(precisions)
    ind = np.arange(N)  # the x locations for total precision bars
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    avg_precision_per_user = []
    avg__ctx_precision_per_user = []

    for user in users_ids:
        avg_precision_per_user.append(precisions[user])
        avg__ctx_precision_per_user.append(ctx_precisions[user])

    rects1 = ax.bar(ind, avg_precision_per_user, width, color='r')
    rects2 = ax.bar(ind + width, avg__ctx_precision_per_user, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel(type)
    ax.set_xlabel('users')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(users_ids)

    ax.legend((rects1[0], rects2[0]), ('Simple Recommendation ' + '(avg)', 'Contextual Recommendation' + '(avg)'))
    plt.show()


def plot_metrics_bar_for_each_fold(data, nFolds, type='Precision'):
    users_ids = data['2d'][0].keys()
    users_ids.sort()

    for fold in range(nFolds):
        precisions = []
        ctx_precisions = []

        for user in users_ids:
            precisions.append(data['2d'][fold][user][type])
            ctx_precisions.append(data['ctx'][fold][user][type])
        N = len(precisions)
        ind = np.arange(N)  # the x locations for total precision bars
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, precisions, width, color='r')

        rects2 = ax.bar(ind + width, ctx_precisions, width, color='y')

        # add some text for labels, title and axes ticks
        ax.set_ylabel(type)
        ax.set_xlabel('users')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(users_ids)

        ax.legend((rects1[0], rects2[0]), ('Simple Recommendation ' + '(fold=' + str(fold) + ')' , 'Contextual Recommendation' + '(fold=' + str(fold) + ')'))
        plt.show()


def precision(user, recommendations, udb, all_udb):
    """
    recommendations: generated by get_recommendations
    udb: test udb

    precision = #tp / #tp + #fp
    where:
        #tp => Recommended and Used
        #fp => Recommended and not Used
    """
    all = len(recommendations) + 0.00000000000000000001 # === tp + fp
    tp = 0
    fp = 0
    for (rec_rating, rec_movie) in recommendations:
        for movie in udb[user].keys():
            if movie == rec_movie and udb[user][movie][RATING] >= OPTIMUM:
                tp += 1
                break
        if rec_movie not in udb[user].keys():
            fp += 1
    return tp/float(tp + fp + 0.00000000000000000001)

def recall(user, recommendations, udb):
    """
    :param user: Target user
    :param recommendations: recommendation list generated for target user
    :param udb: test dataset
    :return:


    recall = #tp / #tp + #fn
    where:
        #tp => Recommended and Used
        #fn => Not recommended but Used
    """
    tp = 0
    fn = 0
    good_movies = 0

    for movie in udb[user].keys():
        if udb[user][movie][RATING] >= OPTIMUM:
            good_movies += 1
            found = False
            for (x, y) in recommendations:
                if movie == y:
                    tp += 1
                    found = True
                    break
            if not found:
                fn += 1
    return tp/float(tp + fn)

def f1score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall)/(precision + recall)


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