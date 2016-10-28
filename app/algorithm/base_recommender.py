class BaseRecommender(object):
    def __init__(self, data_object):
        self.dao = data_object

    def run(self):
        self.init_model()
        self.build_model()

        # TODO: Add evaluation logic here
        # ...

    """
    Method
    ---------
    init_model
        Initilize recommender model
    Returns
    ---------
        Nothing
    """
    def init_model(self):
        raise NotImplementedError("Abstract method")

    """
    Method
    ---------
    build_model
         Learning method: override this method to build a model, for a model-based method.
    Returns
    ---------
        Nothing
    """
    def build_model(self):
        raise NotImplementedError("Abstract method")


    """
    Method
    ---------
    predict
         predict a specific rating for user u on item j. This default implementation returns the average user rating
    Returns
    ---------
        Nothing
    """
    def predict(self, u):
        return self.dao.average_rating


    """
    Method
    --------
    get_eval_info

    Returns
    --------
     the evaluation information of a recommend
    """
    def get_eval_info(self):
        raise NotImplementedError("TODO implement me")

    """
    Method
    ------
    correlation
        Compute the correlation between two vectors for a specific method

    Returns
    -------
        The correlation between vectors i and j; return NaN if the correlation is not computable.
    """
    # TODO To be implemented
    def correlation(self):
        raise NotImplementedError("TODO implement me")

    def print_algo_config(self):
        raise NotImplementedError("TODO implemente me")

    # Serializing a learned model (i.e., variable data) to files.
    # def save_model(self):
    #     raise NotImplementedError("Abstract method")
    #
    # Deserializing a learned model (i.e., variable data) from files.
    # def load_model(self):
    #     raise NotImplementedError("Abstract method")