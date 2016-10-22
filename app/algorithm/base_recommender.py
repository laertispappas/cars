class BaseRecommender(object):
    def __init__(self, data_object):
        self.dao = data_object

    def init_model(self):
        raise NotImplementedError("Abstract method")

    def build_model(self):
        raise NotImplementedError("Abstract method")

    def predict(self):
        raise NotImplementedError("Abstract method")

    # def save_model(self):
    #     raise NotImplementedError("Abstract method")
    #
    # def load_model(self):
    #     raise NotImplementedError("Abstract method")