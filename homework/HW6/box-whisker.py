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

plot_dims = (len(args.timing_sets), 1)
splot = 1

plt.figure(figsize=(7, 2*plot_dims[0]))

first = True
for tset in args.timing_sets:
    plt.subplot(*plot_dims, splot)
    plt.boxplot(
        [[times[tset][field][i] for field in fields] for i in range(len(fields))],
        labels=fields
    )
    plt.title(tset)
    plt.ylabel('Time (s)')
    if first:
        plt.legend()
        first = False

plt.tight_layout()
plt.savefig(args.outfile)
