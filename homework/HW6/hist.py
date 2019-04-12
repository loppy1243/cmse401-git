#!/usr/bin/env python

import argparse
import os
import csv
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-x', dest='excluded_fields', type=str, action='append',
                    help='exclude the specified field from the plot', default=[])
parser.add_argument('-o', dest='outfile', type=str, help='output file')
parser.add_argument('-p', dest='prefix', type=str, help='timing sets prefix')
parser.add_argument('timing_sets', type=str, nargs='+',
                    help='timings sets to include in the plot')
args = parser.parse_args()

times = {}
fields = []
all_fields = None
for tset in args.timing_sets:
    times[tset] = {}
    with open(args.prefix+'/'+tset+'.dat', newline='') as strm:
        reader = csv.reader(strm, delimiter=' ', skipinitialspace=True)
        if fields == []:
            all_fields = next(reader)
            for field in all_fields:
                if field not in args.excluded_fields:
                    fields.append(field)
                    times[tset][field] = []
        else:
            next(reader)
            for field in fields:
                times[tset][field] = []
        for row in reader:
            if row[0] == "AVG":
                continue
            for field, val in zip(all_fields, row):
                if field not in args.excluded_fields:
                    times[tset][field].append(float(val))

#plot_dims = (len(fields), len(args.timing_sets))
plot_dims = (len(fields), 1)
splot_x, splot_y = 1, 1
def splot(x, y):
    return x + (y-1)*plot_dims[1]

plt.figure(figsize=(7*plot_dims[1], 2*plot_dims[0]))

for tset in args.timing_sets:
    first = True
    ax = None
    for field in fields:
        if first:
            ax = plt.subplot(*plot_dims, splot(splot_x, splot_y))
        else:
            plt.subplot(*plot_dims, splot(splot_x, splot_y), sharex=ax)

        plt.hist(times[tset][field], label=tset, histtype='bar')
        plt.title(field)
        if first:
            plt.legend()
        splot_y = splot_y % (plot_dims[0]) + 1
        first = False
    splot_x = splot_x % (plot_dims[1]) + 1
plt.xlabel('Time (s)')

plt.tight_layout()
plt.savefig(args.outfile)
