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
parser.add_argument('-s', dest='separate', action='store_true')
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
                if (len(field) < 3 or field[0:3] != "err") and field not in args.excluded_fields:
                    fields.append(field)
                    times[tset][field] = []
        else:
            next(reader)
            for field in fields:
                times[tset][field] = []
        for row in reader:
            if row != [] and row[0] == "AVG":
                continue
            for field, val in zip(all_fields, row):
                if field in fields:
                    times[tset][field].append(float(val))

if args.separate:
    plt.figure(figsize=(1.5*len(fields), 4*len(args.timing_sets)))
else:
    plt.figure(figsize=(1.5*len(fields), 2*len(args.timing_sets)))

splot = 1

for tset in args.timing_sets:
    if args.separate:
        plt.subplot(2*len(args.timing_sets), 1, 2*splot - 1)
    else:
        plt.subplot(len(args.timing_sets), 1, splot)
    bp = plt.boxplot([times[tset][field] for field in fields], labels=fields)
    for median_line in bp['medians']:
        x, y = median_line.get_xydata()[1]
        plt.annotate("{:.3g}".format(y), (x*1.01, y), fontsize=5)
    plt.title(tset)
    plt.ylabel('Time (s)')

    if args.separate:
        first = True
        for i in range(len(fields)):
            plt.subplot(2*len(args.timing_sets), len(fields), len(fields)*(2*splot-1) + 1 + i)
            bp = plt.boxplot(times[tset][fields[i]])
            if first:
                plt.ylabel('Time (s)')
                first = False

    splot += 1

plt.tight_layout()
plt.savefig(args.outfile)
