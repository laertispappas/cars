from operator import itemgetter
from scipy.stats.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from algs.recommender import Recommender
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import average_precision_score
from math import sqrt


class ItemKNN(Recommender):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.db = Database(self)
        self.db.calculate_similarities(self.pearson_sim)
        self.evaluator = Evaluator(self)

    """
    Function
    --------
    get_top_recos_for_user

    Parameters
    ----------
    userid : string
        The id of the user for whom we want the top recommendations
    df : Dataframe
        The dataframe of movie ratings such
    dbase : instance of Database class.
        A database of similarities, on which the get method can be used to get the similarity
      of two moviees. e.g. dbase.get(rid1,rid2)
    n: int
        the n top choices of the user by star rating
    k : int
        the number of nearest neighbors desired, default 8
    reg: float
        the regularization.


    Returns
    --------
    A sorted list
        of the top recommendations tuples (movie_id, movie_average).
        k-nearest recommendations for each of the user's n top choices,
        removing duplicates and the ones the user has already rated.
    """
    def get_top_recos_for_user(self, userid, df, dbase, n=5, k=7, reg=3.):
        moviez=self.get_user_top_choices(userid, df, numchoices=n)['movie_id'].values
        rated_by_user=df[df.user_id==userid].movie_id.values
        tops=[]
        for ele in moviez:
            t=self.knearest(ele, df.movie_id.unique(), dbase, k=k, reg=reg)
            for e in t:
                if e[0] not in rated_by_user:
                    tops.append(e)

        #there might be repeats. unique it
        ids=[e[0] for e in tops]
        uids={k:0 for k in list(set(ids))}

        topsu=[]
        for e in tops:
            if uids[e[0]] == 0:
                topsu.append(e)
                uids[e[0]] =1
        topsr=[]
        for r, s,nc in topsu:
            average_rate=df[df.movie_id==r].rating.mean()
            topsr.append((r,average_rate))

        topsr=sorted(topsr, key=itemgetter(1), reverse=True)

        if n < len(topsr):
            return topsr[0:n]
        else:
            return topsr

    def get_user_top_choices(self, user_id, df, numchoices=5):
        "get the sorted top 5 movies for a user by the star rating the user gave them"
        udf = df[df.user_id == user_id][['movie_id', 'rating']].sort(['rating'], ascending=False).head(numchoices)
        return udf

    """
    Function
    --------
    calculate_similarity

    Parameters
    ----------
    movie1 : string
        The id of movie 1
    movie2 : string
        The id of movie 2
    df : DataFrame
      A dataframe of reviews, such as the smalldf above
    similarity_func : func
      A function like pearson_sim above which takes two dataframes of individual
      movie reviews made by a common set of reviewers, and the number of
      common reviews. This function returns the similarity of the two movies
      based on the common reviews.

    Returns
    --------
    A tuple
      The first element of the tuple is the similarity and the second the
      common support n_common. If the similarity is a NaN, set it to 0
    """

    def calculate_similarity(self, movie1, movie2, df, similarity_func):
        # find common reviewers
        movie1_reviewers = df[df.movie_id == movie1].user_id.unique()
        movie2_reviewers = df[df.movie_id == movie2].user_id.unique()
        common_reviewers = set(movie1_reviewers).intersection(movie2_reviewers)
        n_common = len(common_reviewers)
        # get reviews
        movie1_reviews = self.get_movie_ratings(movie1, df, common_reviewers)
        movie2_reviews = self.get_movie_ratings(movie2, df, common_reviewers)
        sim = similarity_func(movie1_reviews, movie2_reviews, n_common)
        if np.isnan(sim):
            return 0, n_common
        return sim, n_common


    def get_movie_ratings(self, movie_id, df, set_of_users):
        """
        given a movie id and a set of reviewers, return the sub-dataframe of their
        reviews.
        """
        mask = (df.user_id.isin(set_of_users)) & (df.movie_id == movie_id)
        reviews = df[mask]
        reviews = reviews[reviews.user_id.duplicated() == False]
        return reviews

    """
    Function
    --------
    knearest_amongst_userrated

    Parameters
    ----------
    movie_id : string
        The id of the movie whose nearest neighbors we want
    user_id : string
        The id of the user, in whose reviewed movies we want to find the neighbors
    df: Dataframe
        The dataframe of reviews such as smalldf
    dbase : instance of Database class.
        A database of similarities, on which the get method can be used to get the similarity
      of two movieed. e.g. dbase.get(rid1,rid2)
    k : int
        the number of nearest neighbors desired, default 7
    reg: float
        the regularization.


    Returns
    --------
    A sorted list
        of the top k similar movies. The list is a list of tuples
        (movie_id, shrunken similarity, common support).
    """
    def knearest_amongst_userrated(self, movie_id, user_id, df, dbase, k=7, reg=3.):
        dfuser=df[df.user_id==user_id]
        moviez_user_has_rated = dfuser.movie_id.unique()
        return self.knearest(movie_id, moviez_user_has_rated, dbase, k=k, reg=reg)

    def knearest(self, movie_id, set_of_movies, dbase, k=7, reg=3.):
        """
        Given a movie_id, dataframe, and database, get a sorted list of the
        k most similar movies from the entire database.
        """
        similars = []
        for other_rest_id in set_of_movies:
            if other_rest_id != movie_id:
                sim, nc = dbase.get(movie_id, other_rest_id)
                ssim = self.shrunk_sim(sim, nc, reg=reg)
                similars.append((other_rest_id, ssim / 2.0 + float(nc) / (float(nc) + reg), nc))
        similars = sorted(similars, key=itemgetter(1), reverse=True)
        return similars[0:k]

    """
    Function
    --------
    rating

    Parameters
    ----------
    df: Dataframe
        The dataframe of reviews such as smalldf
    dbase : instance of Database class.
        A database of similarities, on which the get method can be used to get the similarity
      of two movieed. e.g. dbase.get(rid1,rid2)
    movie_id : string
        The id of the movie whose nearest neighbors we want
    user_id : string
        The id of the user, in whose reviewed movies we want to find the neighbors
    k : int
        the number of nearest neighbors desired, default 7
    reg: float
        the regularization.


    Returns
    --------
    A float
        which is the impued rating that we predict that user_id will make for movie_id
    """
    def rating(self, df, dbase, movie_id, user_id, k=7, reg=3.):
        mu = df.rating.mean()
        users_reviews = df[df.user_id == user_id]
        nsum = 0.
        scoresum = 0.
        nears = self.knearest_amongst_userrated(movie_id, user_id, df, dbase, k=k, reg=reg)
        movie_mean = df[df.movie_id == movie_id].movie_average.values[0]
        user_mean = users_reviews.user_average.values[0]
        scores = []
        for r, sold, nc in nears:
            s = sold / 2.0
            shrink_factor = float(nc) / (float(nc) + reg)
            s = s + shrink_factor / 2.0
            scoresum = scoresum + s
            scores.append(s)
            r_reviews_row = users_reviews[users_reviews['movie_id'] == r]
            r_rating = r_reviews_row.rating.values[0]
            r_average = r_reviews_row.movie_average.values[0]
            # rminusb = (r_rating - (r_average + user_mean - mu))
            rminusb = (r_rating - movie_mean)
            nsum = nsum + s * rminusb
        #baseline = (user_mean + movie_mean - mu)
        baseline = movie_mean
        # we might have nears, but there might be no commons, giving us a pearson of 0
        if scoresum > 0.:
            val = nsum / scoresum + baseline
        else:
            val = baseline
        return val

    """
    Function
    --------
    pearson_sim

    Parameters
    ----------
    movie1_ratings: SubFrame
        Rating subframe for first movie
    movie2_ratings: SubFrame.
        Rating subframe for second movie
    n_common : int
        the number of common support

    Returns
    --------
    the pearson correlation coefficient between the user average subtracted ratings
    """
    def pearson_sim(self, movie1_ratings, movie2_ratings, n_common):
        if n_common == 0:
            rho = 0.
        else:
            diff1 = movie1_ratings['rating'] - movie1_ratings['user_average']
            diff2 = movie2_ratings['rating'] - movie2_ratings['user_average']
            rho = pearsonr(diff1, diff2)[0]
        return rho

    def shrunk_sim(self, sim, n_common, reg=3.):
        "takes a similarity and shrinks it down by using the regularizer"
        ssim = (n_common * sim) / (n_common + reg)
        return ssim


class Database:
    "A class representing a database of similaries and common supports"
    def __init__(self, recommender):
        "the constructor, takes a reviews a recommender like ItemKNN that holds the data object and related similarity functions"
        database={}
        self.df = recommender.data_object.rating_data
        self.recommender = recommender
        self.unique_movie_ids={v:k for (k,v) in enumerate(self.df.movie_id.unique())}

        keys=self.unique_movie_ids.keys()
        l_keys=len(keys)
        self.database_sim=np.zeros([l_keys,l_keys])
        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)

    def calculate_similarities(self, similarity_func):
        items=self.unique_movie_ids.items()
        for b1, i1 in items:
            for b2, i2 in items:
                if i1 < i2:
                    sim, nsup=self.recommender.calculate_similarity(b1, b2, self.df, similarity_func)
                    self.database_sim[i1][i2]=sim
                    self.database_sim[i2][i1]=sim
                    self.database_sup[i1][i2]=nsup
                    self.database_sup[i2][i1]=nsup
                elif i1==i2:
                    nsup=self.df[self.df.movie_id==b1].user_id.count()
                    self.database_sim[i1][i1]=1.
                    self.database_sup[i1][i1]=nsup

    def get(self, b1, b2):
        "returns a tuple of similarity,common_support given two movie ids"
        sim=self.database_sim[self.unique_movie_ids[b1]][self.unique_movie_ids[b2]]
        nsup=self.database_sup[self.unique_movie_ids[b1]][self.unique_movie_ids[b2]]
        return (sim, nsup)

class Evaluator(object):
    def __init__(self, item_knn):
        self.item_knn = item_knn

    def evaluate_ratings(self):
        raise NotImplementedError

    def evaluate_ranking(self):
        raise NotImplementedError

    def evaluate(self):
        print "k=2, reg=1."
        self.make_results_plot(2, 1.)
        plt.title("k=2, reg=1.")

        print "k=2, reg=15."
        self.make_results_plot(2, 15., )
        plt.title("k=2, reg=15.")

        print "k=15, reg=1."
        self.make_results_plot(15, 1.)
        plt.title("k=15, reg=1.")

        print "k=15, reg=15."
        self.make_results_plot(15, 15., )
        plt.title("k=15, reg=15.")

    def evaluate_all(self):
        self.rmse();
        self.f_score1()

    def rmse(self):
        print "Calculating RMSE"
        actual = self.actual_ratings()
        predicted = self.predicted_ratings()
        rmse = sqrt(mean_squared_error(actual, predicted))
        print "RMSE= ", rmse

    # TODO
    def f_score1(self):
        actual = self.actual_ratings().astype(int)
        predicted = self.predicted_ratings().astype(int)
        score = f1_score(actual, predicted, average='micro')
        print "F1 score= ", score

    # TODO
    def map(self):
        print "calculatinf MAP"

    # TODO
    def groc_curves(self):
        """
        We use a global ROC (GROC) curve to measure performance
        when we are allowed to recommend more often to
        some users than others. GROC curves are constructed in
        the following manner:
        1. Order the predictions pred(pi, mj ) in a list
            by magnitude, imposing an ordering: (p, m) k
        2. Pick n, calculate hit/miss rates caused by predicting the top n (p, m)k by magnitude, and
            plot the point
            By selecting different n (e.g. incrementing n by a fixed amount) we draw a curve on the graph.
        """

        print "test"

    # TODO
    def croc_curves(self):
        """
        Customer ROC (CROC) curves measure performance of
        a recommender system when we are constrained to recommend
        the same number of items to each user.:
        """
    def k_fold_cross_val_poly(folds, degrees, X, y):
        n = len(X)
        kf = KFold(n, n_folds=folds)
        kf_dict = dict([("fold_%s" % i, []) for i in range(1, folds + 1)])
        fold = 0
        for train_index, test_index in kf:
            fold += 1
            print "Fold: %s" % fold
            X_train, X_test = X.ix[train_index], X.ix[test_index]
            y_train, y_test = y.ix[train_index], y.ix[test_index]
            # Increase degree of linear regression polynomial order
            for d in range(1, degrees + 1):
                print "Degree: %s" % d
                # Create the model and fit it
                polynomial_features = PolynomialFeatures(
                    degree=d, include_bias=False
                )
                linear_regression = LinearRegression()
                model = Pipeline([
                    ("polynomial_features", polynomial_features),
                    ("linear_regression", linear_regression)
                ])
                model.fit(X_train, y_train)
                # Calculate the test MSE and append to the
                # dictionary of all test curves
                y_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred)
                kf_dict["fold_%s" % fold].append(test_mse)
            # Convert these lists into numpy arrays to perform averaging
            kf_dict["fold_%s" % fold] = np.array(kf_dict["fold_%s" % fold])
        # Create the "average test MSE" series by averaging the
        # test MSE for each degree of the linear regression model,
        # across each of the k folds.
        kf_dict["avg"] = np.zeros(degrees)
        for i in range(1, folds + 1):
            kf_dict["avg"] += kf_dict["fold_%s" % i]
        kf_dict["avg"] /= float(folds)
        return kf_dict

    def plot_test_error_curves_kf(kf_dict, folds, degrees):
        fig, ax = plt.subplots()
        ds = range(1, degrees + 1)
        for i in range(1, folds + 1):
            ax.plot(ds, kf_dict["fold_%s" % i], lw=2, label='Test MSE - Fold %s' % i)
        ax.plot(ds, kf_dict["avg"], linestyle='--', color="black", lw=3, label='Avg Test MSE')
        ax.legend(loc=0)
        ax.set_xlabel('Degree of Polynomial Fit')
        ax.set_ylabel('Mean Squared Error')
        ax.set_ylim([0.0, 4.0])
        fig.set_facecolor('white')
        plt.show()

    def compare_results(sefl, rating_actual, rating_predicted, ylow=-10, yhigh=15, title=""):
        """
        plot predicted results against actual results. Takes 2 arguments: a
        numpy array of actual ratings and a numpy array of predicted ratings
        scatterplots the predictions, a unit slope line, line segments joining the mean,
        and a filled in area of the standard deviations."
        """
        df = pd.DataFrame(dict(actual=rating_actual, predicted=rating_predicted))
        ax = plt.scatter(df.actual, df.predicted, alpha=0.2, s=30, label="predicted")
        plt.ylim([ylow, yhigh])
        plt.plot([1, 5], [1, 5], label="slope 1")
        xp = [1, 2, 3, 4, 5]
        yp = df.groupby('actual').predicted.mean().values
        plt.plot(xp, yp, 'k', label="means")
        sig = df.groupby('actual').predicted.std().values
        plt.fill_between(xp, yp - sig, yp + sig,
                         color='k', alpha=0.2)
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.legend(frameon=False)
        plt.grid(False)
        plt.title(title)
        print "fraction between -15 and 15 rating", np.mean(np.abs(df.predicted) < 15)
        plt.show()

    def make_results_plot(self, k, reg):
        actual = self.actual_ratings()
        predicted = self.predicted_ratings(k=k, reg=reg)
        self.compare_results(actual, predicted, ylow=1, yhigh=5)

    def actual_ratings(self):
        return self.item_knn.data_object.rating_data.rating.values

    def predicted_ratings(self, k = 10, reg = 3):
        uid = self.item_knn.data_object.rating_data.user_id.values
        bid = self.item_knn.data_object.rating_data.movie_id.values
        actual = self.actual_ratings()
        predicted = np.zeros(len(actual))
        counter = 0
        for user_id, biz_id in zip(uid, bid):
            predicted[counter] = self.item_knn.rating(self.item_knn.data_object.rating_data, self.item_knn.db, biz_id, user_id, k=k, reg=reg)
            counter = counter + 1
        return predicted