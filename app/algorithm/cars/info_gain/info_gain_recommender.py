from collections import Counter

from app.algorithm.cars.context_recommender import ContextRecommender
from app.algorithm.cars.info_gain.plotter import Plotter
from app.dataset.data_object import DataObject
from app.dataset.loader import AutoVivification
from app.algorithm.cars.info_gain.evaluation import KFold
import numpy

import os

from app.utils.similarities import sim_pearson

RATING = 0 # Holds the index of rating in user preference array
OPTIMUM = 3.5 # Optimum rating for recommendation > 3.5
VERBOSE = True

class InfoGainRecommender(ContextRecommender):
    def __init__(self, data_object):
        super(self.__class__, self).__init__(data_object)
        self.filters = [(5,2)]

    def init_model(self):
        pass

    def build_model(self):
        # TODO: Remove after code cleanup
        self.set_training_set()

    def set_training_set(self, trainingSet = None):
        if trainingSet == None:
            self.training_data = self.dao.users
        else:
            self.training_data = trainingSet

    # TODO implement on concrete class
    def TraditionalRecommendation(self, user, simMeasure=sim_pearson, nNeighbors=None, topN=10):
        return self.cf_recs(user, simMeasure, nNeighbors, topN)

    def PostFilteringRecommendation(self, user, simMeasure=sim_pearson, nNeighbors=None, topN=10):
        # TODO: remove all 999999999
        recs = self.cf_recs(user, simMeasure, nNeighbors, topN=99999999999)
        return self.__contextual_filter(recs, topN)[0:topN]


    # Returns top N context aware recommendations for the given user.
    #
    def top_recommendations(self, user, N = 10):
        recs = self.cf_recs(user)

        # Context - Value tuple
        filter_recs = self.__contextual_filter(recs, topN=N)
        return filter_recs[1:N]

    def getPredictedRating(self, user, item, nearestNeighbors):
        if item in self.training_data[user].keys():
            return self.training_data[user][item]

        movie_ratings = []
        for movie_details in self.training_data[user].values():
            movie_ratings.append(movie_details[RATING])
        meanRating = numpy.mean(movie_ratings)

        weightedSum = 0
        normalizingFactor = 0
        for neighbor, similarity in nearestNeighbors.items():
            if item not in self.training_data[neighbor]:
                continue
            neighborsRatings = []
            for movie_details in self.training_data[neighbor].values():
                neighborsRatings.append(movie_details[RATING])
            meanRatingOfNeighbor = numpy.mean(neighborsRatings)

            weightedSum += similarity * (self.training_data[neighbor][item][RATING] - meanRatingOfNeighbor)
            normalizingFactor += numpy.abs(similarity)
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)

    def cf_recs(self, user, similarity=sim_pearson, nNeighbors=20, topN=10):
        self.target_user = user

        predictedScores = []
        self.similarities = self.getNearestNeighbors(user, similarity, nNeighbors)
        for item in self.dao.movies.keys():
            if item in self.training_data[user]:
                continue
            itemRaters = {}  # Nearest neighbors who rated on the item
            for similarity, neighbor in self.similarities:
                if similarity <= 0 or len(itemRaters) == nNeighbors:
                    break
                if item in self.training_data[neighbor].keys():
                    itemRaters[neighbor] = similarity
            predicted_rating = self.getPredictedRating(user, item, itemRaters)
            predictedScores.append((predicted_rating, item))
        predictedScores.sort(reverse=True)

        return predictedScores[0:topN]

    def getNearestNeighbors(self, target, simMeasure, nNeighbors=None):
        prefs = self.training_data
        # sim = similarity(prefs, person, other)
        similarities = [(simMeasure(prefs, target, other), other) for other in prefs if target != other]
        similarities.sort(reverse=True)
        if nNeighbors != None:
            similarities = similarities[0:nNeighbors]
        return similarities  # similarities = [(similarity, neighbor), ...]

    def getContextualNearestNeighbors(self, target, simMeasure, nNeighbors=None):
        prefs = self.training_data

        context, filter_ctx_value = self.filters[0]
        # if movie in self.training_data[neighbor] and context in self.training_data[neighbor][movie] and self.training_data[neighbor][movie][context] == filter_ctx_value:

        similarities = []
        for other in prefs:
            if other == target:
                next
            for movie in self.training_data[other]:
                if context in self.training_data[other][movie] and self.training_data[other][movie][context] == filter_ctx_value:
                    similarities.append((simMeasure(prefs, target, other), other))
        # similarities = [(simMeasure(prefs, target, other), other) for other in prefs if target != other]
        similarities.sort(reverse=True)
        if nNeighbors != None:
            similarities = similarities[0:nNeighbors]
        return similarities  # similarities = [(similarity, neighbor), ...]

    def __contextual_filter(self, recommendations, topN=10):
        # Relevance of item i for target user u in a particular context c
        # is approximated by the propability Pc(u,i,c) = |Uu,i,c| / k where k is the number
        # of neighbors used by kNNand Uu,i,c = { v in N(u)|Rv,i,c != 0} that is the user's neighbor v
        # in the neighborhood of u, N(u) who have rated / consumed item i in context c. The item relevance
        # is determined by the threshold value tpc (0.1) that is used to contextualize the ratings as follows:
        #
        # F(u,i,c) = F(u,i)       if Pc(u,i,c) >= tpc
        #          = F(u,i) -0.5  if Pc(u,i,c) < tpc
        #
        # where F(u,i) denotes the context-unaware rating prediction by RS and F(u,i,c) denotes
        # the context-aware rating prediction.
        #
        filtered_recs = []
        nNeighbors = len(self.similarities)
        self.contextual_similarities = self.getContextualNearestNeighbors(self.target_user, sim_pearson, nNeighbors)
        tpc = 0.2

        for rating, movie in recommendations:
            nNeighbors_rated_item_in_same_context = 0
            for sim, neighbor in self.contextual_similarities:
                for context, filter_ctx_value in self.filters:
                    if movie in self.training_data[neighbor] and context in self.training_data[neighbor][movie] and self.training_data[neighbor][movie][context] == filter_ctx_value:
                            nNeighbors_rated_item_in_same_context += 1
            puic = float(nNeighbors_rated_item_in_same_context) / float(nNeighbors)
            # Weight
            filtered_recs.append((rating + rating * puic, movie))
            # Filter
            # if puic >= tpc:
            #   filtered_recs.append((rating, movie))
            # else:
            #   filtered_recs.append((rating - 3.25, movie))

            # # Filter - weight
            # if puic >= tpc:
            #     filtered_recs.append((rating + rating * puic, movie))
            # else:
            #     filtered_recs.append((rating - 0.25, movie))

        filtered_recs.sort(reverse=True)
        return filtered_recs[0:topN]