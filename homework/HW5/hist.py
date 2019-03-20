import argparse
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-x', dest='excluded_fields', type=str, action='append',
                    help='exclude the specified field from the plot', default=[])
parser.add_argument('-o', dest='outfile', type=str, help='output file')
parser.add_argument('timing_sets', type=str, nargs='+',
                    help='timings sets to include in the plot')
args = parser.parse_args()

prefix = 'timings/compiled/'
nodes = os.listdir(prefix+'/'+args.timing_sets[0])
for i in range(len(nodes)):
    nodes[i] = '.'.join(nodes[i].split('.')[:-1])

times = {}
fields = []
all_fields = None
for node in nodes:
    times[node] = {}
    for tset in args.timing_sets:
        times[node][tset] = {}
        with open(prefix+'/'+tset+'/'+node+'.dat', newline='') as strm:
            reader = csv.reader(strm, delimiter=' ', skipinitialspace=True)
            if fields == []:
                all_fields = next(reader)
                for field in all_fields:
                    if field not in args.excluded_fields:
                        fields.append(field)
                        times[node][tset][field] = []
            else:
                next(reader)
                for field in fields:
                    times[node][tset][field] = []
            for row in reader:
                for field, val in zip(all_fields, row):
                    if field not in args.excluded_fields:
                        times[node][tset][field].append(float(val))

plot_dims = (len(fields), len(args.timing_sets))
splot_x, splot_y = 1, 1
def splot(x, y):
    return x + (y-1)*plot_dims[1]

for tset in args.timing_sets:
    first = True
    ax = None
    for field in fields:
        if first:
            first = False
            ax = plt.subplot(*plot_dims, splot(splot_x, splot_y))
        else:
            plt.subplot(*plot_dims, splot(splot_x, splot_y), sharex=ax)

        for node in nodes:
            plt.hist(times[node][tset][field], label=node)
        plt.title(tset+' '+field)
        plt.legend()
        splot_y = splot_y % (plot_dims[0]) + 1
    splot_x = splot_x % (plot_dims[1]) + 1

plt.tight_layout()
plt.savefig(args.outfile)
