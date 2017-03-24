import argparse
import os
import sys
import logging
import math
import re
import time
from collections import Counter
from itertools import izip_longest
import numpy as np
import matplotlib as mpl
from textwrap import wrap
mpl.use('Agg')
import matplotlib.pyplot as plt

legend_dic = {
    'upper_left': 2,
    'upper_right': 1
}

def computeStats(numbers):
    _count = len(numbers)
    _sum = sum(numbers)
    _min = min(numbers)
    _lq = np.percentile(numbers, 25)
    _mean = float(_sum) / _count
    _median = np.median(numbers)
    _mode = Counter(numbers).most_common(1)
    _uq = np.percentile(numbers, 75)
    _max = max(numbers)
    return {
        'count': _count,
        'sum': _sum,
        'min': _min,
        'lower_quartile': _lq,
        'mean': _mean,
        'median': _median,
        'mode': _mode,
        'upper_quartile': _uq,
        'max': _max
    }

def dict2String(dic):
    return ('\n').join(['{} = {}'.format(key, dic[key]) for key in dic])

parser = argparse.ArgumentParser(description='Given a list of lengths in a file (one per line), generate a histogram.')
parser.add_argument('-lengths', nargs='+', help='File containing lengths, 1 per line. If more than one file is provided the "-sof" flag will be forced on')
parser.add_argument('-labels', nargs='+', help='Give a name for each file/series')
parser.add_argument('-e', '--edge_color', default='none', help='Set the edge color. Default = "none".')
parser.add_argument('-b', '--binsize', type=int, default=100, help='Bin size. Default = 100')
parser.add_argument('-ns', '--no_stats', action='store_true', help='Set this flag to disable the computation of statistics and improve runtime')
parser.add_argument('-normed', action='store_true', help='Set this flag to enable normalization of data')
parser.add_argument('-sof', '--stats_on_fig', action='store_false', help='Set this flag to disable the plotting of stats to the figure')
parser.add_argument('-r', '--xarange', nargs=2, type=int, default=None, help='Range of lengths to plot. Default = None')
parser.add_argument('-t', '--title', default='Length Distribution', help='Title of plot')
parser.add_argument('-xlab', default='Length', help='Label for x-axis')
parser.add_argument('-ylab', default='Frequency', help='Label for y-axis')
parser.add_argument('-gs', '--genome_size', type=int, default=None, help='If input will return fold-coverage')
parser.add_argument('-gr', '--no_grid', action='store_false', help='Disable grid')
parser.add_argument('-lpos', '--legend_position', default='upper_left', choices=('upper_left', 'upper_right'), help='Set the legend position. Default = \'upper_left\'')
parser.add_argument('-l', '--log', dest='logLevel', default='WARNING', choices=('debug', 'info', 'warning', 'error', 'critical'), help='Set the logging level. Default = \'warning\'')

args = parser.parse_args()

if not args.lengths:
    parser.error('Must have -lengths specified')

if not args.labels:
    args.labels = ['series_{}'.format(i) for i in xrange(len(args.lengths))]

if len(args.lengths) != len(args.labels):
    parser.error('Number of data files not equal to number of labels')

# Logging
logging.basicConfig(level=getattr(logging, args.logLevel.upper()))

series = []
for i,item in enumerate(args.lengths):
    logging.debug('Reading and sorting dataset {} ...'.format(i))
    start = time.time()
    data = sorted([int(round(float(x))) for x in open(item, 'r').readlines()])
    logging.debug('DONE. Time = {}'.format(time.time()-start))
    stat = None
    if not args.no_stats:
        start = time.time()
        logging.debug('Computing stats for dataset {} ...'.format(i))
        stat = computeStats(data)
        logging.debug('DONE. Time = {}'.format(time.time()-start))
    series.append((data, stat))

if args.no_stats and args.genome_size:
    for data, stat in series:
        stat['Coverage'] = stat['sum']/args.genome_size

fig, ax = plt.subplots()
logging.debug('Creating axis ...')
logging.debug('Series preview:\n{}'.format([zip(*[x[0][:50] for x in series])]))
#n, bins, patches = ax.hist(np.array([y for y in izip_longest(*[x[0] for x in series])]), args.binsize, args.xarange, normed=args.normed, histtype='bar', label=args.labels)
colors = plt.cm.Set1(np.linspace(0.1, 0.9, len(series)))
n, bins, patches = ax.hist([x[0] for x in series], args.binsize, args.xarange, normed=args.normed, histtype='bar', label=args.labels, color=colors, edgecolor=args.edge_color)
legend = ax.legend(bbox_to_anchor=(1.2,1))
xtextpos = ax.get_xlim()[1] * 0.5
ytextpos = ax.get_ylim()[1] * 0.25
if args.xarange:
    xtextpos = (args.xarange[1] + args.xarange[0]) * 0.5
    logging.debug('xtextpos = {}'.format(xtextpos))
ax.set_title(args.title)
ax.set_xlabel(args.xlab)
ax.set_ylabel(args.ylab)
ax.grid(args.no_grid)
#ax.legend(loc=legend_dic[args.legend_position])
if args.stats_on_fig and (len(args.lengths) == 1):
    ax.annotate(stats.replace('\t','='), (xtextpos,ytextpos), backgroundcolor=(1,1,1,0.7))
figname = '{}.png'.format(('\n').join(wrap(re.sub(r'\s+', '_', args.title),60)))
#fig.tight_layout()
fig.savefig(figname, bbox_extra_artists=(legend,), bbox_inches='tight')

print 'Figure saved: {}'.format(figname)

for i, item in enumerate(series):
    stats = item[1]
    print args.labels[i]
    print '-'*len(args.labels[i])
    print dict2String(stats)
    print
