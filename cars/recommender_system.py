from data.plotter import Plotter
from recommender import Recommender
from data.data_reader import DataReader

from collections import defaultdict
import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl


class RecommenderSystem(object):
  def __init__(self):
    self.data_object = DataReader().load()
    self.recommender = Recommender()
    self.plotter = Plotter(self.data_object)

  def recomend(self, user, context):
    print("RecommenderSystem#recomend")


##### TEST
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


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks

    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

fulldf=pd.read_csv("/home/lapis/projects/my/python34/cars/dataset/ldos/LDOS-CoMoD_ratings.csv")
print "Number of Ratings", fulldf.shape[0]
print "Number of Users", fulldf.userID.unique().shape[0], "Number of Movies", fulldf.itemID.unique().shape[0]

system = RecommenderSystem()
data_object = system.data_object
plotter = system.plotter

data_object.number_of_movies()
data_object.number_of_ratings()
data_object.number_of_users()
data_object.mean_rating_value()

# plotter.plot_average_rating_hist()
# plotter.plot_user_and_movie_hist()
# plotter.plot_average_user_rating()
# plotter.plot_average_movie_rating()