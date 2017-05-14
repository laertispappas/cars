import sys
import numpy as np
import pylab as pl
from sklearn.metrics import precision_recall_curve

from app.dataset.loader import AutoVivification

colors = "bgrcmyk"  # 7 is a prime, so we'll loop over all combinations of colors and markers, when zipping their cycles
markers = "so^>v<dph8"  # +x taken out, as no color.

class Plotter(object):
    def __init__(self, metrics):
        """
        :param metrics: Dictionary irStats of the evaluation:
        { Context: { condition: { precision: value, precision-ctx: valus-ctx, recall: r_val, recall-ctx: rctx_val } } }
        """
        self.metrics = metrics

    def plotf1Score(self):
        import matplotlib.pyplot as plt;
        plt.rcdefaults()
        import numpy as np
        import matplotlib.pyplot as plt

        objects = ['Baseline'] + self.metrics.keys()
        base_avg = self.metrics['Time']['Weekend']['f1Score']
        time_avg = (self.metrics['Time']['Weekend']['f1Score-ctx'] + self.metrics['Time']['Weekday']['f1Score-ctx']) / 2
        location_avg = (self.metrics['Location']['Home']['f1Score-ctx'] + self.metrics['Location']['Cinema']['f1Score-ctx']) / 2
        companion_avg = (self.metrics['Companion']['Alone']['f1Score-ctx'] + self.metrics['Companion']['Partner']['f1Score-ctx'] + self.metrics['Companion']['Family']['f1Score-ctx']) / 3

        avgs = [base_avg, time_avg, location_avg, companion_avg]
        y_pos = np.arange(len(objects))

        plt.bar(y_pos, avgs, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('F1Score')
        plt.xlabel('Context')
        plt.title('F1-Score Evaluation Metrics')

        plt.show()

    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt

        for context in self.metrics.keys():
            labels = []
            baseline_metrics = (self.metrics[context][self.metrics[context].keys()[0]]['precision'],
                                self.metrics[context][self.metrics[context].keys()[0]]['recall'],
                                self.metrics[context][self.metrics[context].keys()[0]]['f1Score'])
            contextual_metrics = AutoVivification()
            labels.extend(['Baseline', 'Baseline', 'Baseline'])

            for condition in self.metrics[context].keys():
                contextual_metrics[condition] = (self.metrics[context][condition]['precision-ctx'],
                                                 self.metrics[context][condition]['recall-ctx'],
                                                 self.metrics[context][condition]['f1Score-ctx'])

                labels.extend([condition, condition, condition])
            fig, ax = plt.subplots()
            index = np.arange(3)
            bar_width = 0.15
            opacity = 0.8

            rects1 = plt.bar(index, baseline_metrics, bar_width,
                             alpha=opacity,
                             color='b',
                             label='Baseline')
            colors = {
                'Weekend': 'y',
                'Weekday': 'c',
                'Home': 'y',
                'Cinema': 'c',
                'Alone': 'y',
                'Partner': 'c',
                'Family': '#ee1f3f',
            }
            i = 1
            for condition in contextual_metrics.keys():
                plt.bar(index + bar_width * i, contextual_metrics[condition], bar_width,
                        alpha=opacity,
                        color=colors[condition],
                        label=condition)
                i += 1

            rects = ax.patches
            # Now make some labels
            # labels = ["label%d" % i for i in xrange(len(rects))]
            for rect, label in zip(rects, labels):
                height = rect.get_height()
                # ax.text(rect.get_x() + rect.get_width() / 2, height, None, ha='center', va='bottom')

            # plt.xlabel('Metric')
            # plt.ylabel('Value')
            plt.title('Evaluation Metrics for ' + context)
            plt.xticks(index + bar_width + 0.15, ('Precision', 'Recall', 'F1Score'))

            import math
            plt.ylim([0, 1])
            plt.legend()
            plt.tight_layout()
            plt.show()