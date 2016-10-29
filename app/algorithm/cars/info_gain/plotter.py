import scipy as sc
import pylab as pl
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

def fmeasure(p, r):
    """ Calculates the fmeasure for precision p and recall r. """
    return 2 * p * r / (p + r)

def _fmeasureCurve(f, p):
    """ The f1 measure is defined as: f(p,r) = 2*p*r / (p + r)
        If you want to plot "equipotential-lines" into a
        precision/recall diagramm (recall (y) over precision (x)),
        for a given fixed f value we get the function:"""
    return f * p / (2 * p - f)


def _plotFMeasures(fstepsize=.01, stepsize=0.0001):
    """ Plots 10 fmeasure Curves into the current canvas. """
    p = sc.arange(0.001, 0.09, stepsize)[1:]  # @UndefinedVariable
    for f in sc.arange(0.001, 0.09, fstepsize)[1:]:  # @UndefinedVariable
        points = [(x, _fmeasureCurve(f, x)) for x in p
                  if 0 < _fmeasureCurve(f, x) <= 1.5]
        xs, ys = zip(*points)
        curve, = pl.plot(xs, ys, "--", color="gray",
                         linewidth=0.5)  # , label=r"$f=%.1f$"%f) # exclude labels, for legend @UnusedVariable
        # bad hack:
        # gets the 10th last datapoint, from that goes a bit to the left, and a bit down
        pl.annotate(r"$f=%.1f$" % f, xy=(xs[-10], ys[-10]), xytext=(xs[-10] - 0.05, ys[-10] - 0.035), size="small",
                    color="gray")


colors = "bgrcmyk"  # 7 is a prime, so we'll loop over all combinations of colors and markers, when zipping their cycles
markers = "so^>v<dph8"  # +x taken out, as no color.

def plotPrecisionRecallDiagram(title="title", points=None, labels=None, loc="center right"):
    """ Plots 10 f-Measure equipotential lines plus the (precision,recall) points
        into the current canvas. Points is a list of (precision,recall) pairs.
        Optionally you can also provide a labels (list of strings), which will be
        used to create a legend, which is located at loc."""
    if labels != None:
        ax = pl.axes([0.1, 0.1, 0.7, 0.8])
        # pl.axes([0.1, 0.1, 0.7, 0.8]) # llc_x, llc_y, width, height
    else:
        ax = pl.gca()
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 0.1])
    pl.title(title)
    pl.xlabel("Precision")
    pl.ylabel("Recall")
    _plotFMeasures()

    # _contourPlotFMeasure()

    if points != None:
        getColor = it.cycle(colors).next
        getMarker = it.cycle(markers).next

        scps = []  # scatter points
        for i, (x, y) in enumerate(points):
            label = None
            if labels: label = labels[i]
            print i, x, y, label
            scp = ax.scatter(x, y, label=label, s=50, linewidths=0.75,
                             facecolor=getColor(), alpha=0.75, marker=getMarker())
            scps.append(scp)
            # pl.plot(x,y, label=label, marker=getMarker(), markeredgewidth=0.75, markerfacecolor=getColor())
            # if labels: pl.text(x, y, label, fontsize="x-small")
        if labels:
            # pl.legend(scps, labels, loc=loc, scatterpoints=1, numpoints=1, fancybox=True) # passing scps & labels explicitly to work around a bug with legend seeming to miss out the 2nd scatterplot
            pl.legend(scps, labels, loc=(1.01, 0), scatterpoints=1, numpoints=1,
                      fancybox=True)  # passing scps & labels explicitly to work around a bug with legend seeming to miss out the 2nd scatterplot
    pl.axis([-0.02, 1.02, -0.02, 1.02])  # xmin, xmax, ymin, ymax


class Plotter(object):
    def __init__(self, metrics):
        self.metrics = metrics
        self.user_labels = ["User " + str(u) for u in metrics.keys()]

    def plot_precision_recall_curves(self, contextual = False, fig_label = 'PrecRecallFigure', file_name = 'precRecallResult.png'):
        precision_recalls = []
        if contextual:
            precision_recalls = [(self.metrics[u]['precision'][1], self.metrics[u]['recall'][1]) for u in self.metrics.keys()]
        else:
            precision_recalls = [(self.metrics[u]['precision'][0], self.metrics[u]['recall'][1]) for u in self.metrics.keys()]

        plotPrecisionRecallDiagram(fig_label, precision_recalls, self.user_labels)
        pl.savefig(file_name, dpi=300)
        pl.show()

    def plot_precision_bar(self, type='precision'):
        precisions = []
        ctx_precisions = []
        for user in self.metrics.keys():
            precisions.append(self.metrics[user][type][0])
            ctx_precisions.append(self.metrics[user][type][1])

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
        ax.set_xticklabels(self.user_labels)

        ax.legend((rects1[0], rects2[0]), ('Simple Recommendation', 'Contextual Recommendation'))
        plt.show()
