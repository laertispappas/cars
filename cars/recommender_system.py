from matplotlib import rcParams
from algs.item_knn import ItemKNN
from algs.camf_cuci import CAMF_CUCI
from data.plotter import Plotter

class RecommenderSystem(object):
    def __init__(self):
        self.recommender = CAMF_CUCI()
        self.plotter = Plotter(self.recommender.data_object)

    def recomend(self, user, context):
        print("RecommenderSystem#recomend")

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'


system = RecommenderSystem()
data_object = system.recommender.data_object
# plotter = system.plotter

# data_object.number_of_movies()
# data_object.number_of_ratings()
# data_object.number_of_users()
# data_object.mean_rating_value()
#
# plotter.plot_average_rating_hist()
# plotter.plot_user_and_movie_hist()
# plotter.plot_average_user_rating()
# plotter.plot_average_movie_rating()

# testuserid=42
# print "For user", testuserid, "the top recommendations are:"
# toprecos=system.recommender.get_top_recos_for_user(testuserid, system.recommender.data_object.rating_data, system.recommender.db, n=5, k=7, reg=3.)
# for biz_id, biz_avg in toprecos:
#     print biz_id, "| Average Rating |", biz_avg
#
# system.recommender.evaluator.evaluate()
# system.recommender.evaluator.evaluate_all()
# print data_object.movie_metadata

system.recommender.data_object.print_specs()
system.recommender.build_model()
# print "Train Matrix:"
# print system.recommender.train_matrix
#
# print "Test Matrix:"
# print system.recommender.test_matrix
#
# print "Rating Matrix:"
# print system.recommender.rating_matrix
#
# print "Rating Data:"
# print data_object.rating_data
# print system.recommender.train_matrix
