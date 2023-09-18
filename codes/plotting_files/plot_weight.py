import json 
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import configparser


taxa_list = ["phylum", 'class', 'order', 'family', 'genus']


# Create a ConfigParser object and read the .ini file
config = configparser.ConfigParser()
config.read('configs.ini')

# index = argument_parser().parse_args().index
seed_list = [4, 13, 42, 71, 265]

seed = config.get('model', 'seed')
taxa_name = config.get('model', 'taxa')
csv1 = 'logs_class_4/nohost/data.csv'
csv2 = 'logs_w1/nohost/data.csv'
csv3 = 'logs_w2/nohost/data.csv'
csv4 = 'logs_w3/nohost/data.csv'
csv_list = [
            csv1, 
            csv2,
            csv3,
            csv4
        ]


# plot lines
labels = [
    'ALG(GCN)_NoHost',
    'ALG(GCN)_NoHost_w1',
    'ALG(GCN)_NoHost_w2',
    'ALG(GCN)_NoHost_focal'
    ]
markers = ['o', 'v', 's', 'p', '*', 'x', 'D', 'h', 'X', 'd']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
cnt = 0
num_min = 100
num_max = 0
for j in csv_list:
    data = pd.read_csv(j)
    plt.plot(data['num'].values, data['acc'].values, label = labels[cnt], color = colors[cnt], marker= markers[cnt])
    num_min = min(num_min, data['num'].values[0])
    num_max = max(num_max, data['num'].values[-1])
    cnt+=1

plt.title('Taxonomy: {}, Random Seed: {}'.format(taxa_name, seed))
plt.xlabel('Number of labeled nodes')
plt.ylabel('Test accuracy')
if num_max > 500:
    step = 50
else:
    step = 25
    
plt.xticks([i for i in range(25, (int(num_max / 25) + 1)*25, step)])
plt.legend()
print("Drawing figure for {}, seed {}".format(taxa_name, seed))
plt.savefig('figs/paper/weight_comp.png')
