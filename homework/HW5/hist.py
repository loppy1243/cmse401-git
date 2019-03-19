import csv
import numpy as np
from matplotlib import pyplot as plt

prefix = 'timings/compiled/serial'
nodes = ['dev-intel14-k20', 'dev-intel16-k80']

walltimes = {}
cputimes = {}
for node in nodes:
    walltimes[node] = []
    cputimes[node] = []
    with open(prefix+'/'+node+'.dat', newline='') as strm:
        reader = csv.reader(strm, delimiter=' ', skipinitialspace=True)
        next(reader)
        for row in reader:
            walltimes[node].append(float(row[1]))
            cputimes[node].append(float(row[2]))

ax = plt.subplot(2, 1, 1)
for node in nodes:
    plt.hist(walltimes[node], label=node)
    plt.title('Serial Wall Times')
    plt.legend()

plt.subplot(2, 1, 2, sharex=ax)
for node in nodes:
    plt.hist(cputimes[node], label=node)
    plt.title('Serial CPU Times')
    plt.legend()

plt.tight_layout()
plt.savefig('serial_hist.pdf')
