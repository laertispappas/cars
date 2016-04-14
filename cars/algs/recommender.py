from data.data_reader import DataReader

class Recommender(object):
    def __init__(self):
        self.data_object = DataReader().load()