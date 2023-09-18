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
csv1 = 'logs/algmlp/data.csv'
csv2 = 'logs/alggcn/data.csv'
csv3 = 'logs/entropy/data.csv'
csv4 = 'logs/random/data.csv'
csv6 = 'logs/trainall/data.csv'
csv5 = 'logs/nohost/data.csv'
csv7 = 'logs/allnohosts/data.csv'
csv8 = 'logs/entropynohost/data.csv'
csv9 = 'logs/randomnohost/data.csv'
csv10 = 'logs/algmlpnohost/data.csv'
# csv11 = 'logs/inverse/data.csv'
csv_list = [
            csv1, 
            csv10, 
            csv2, 
            csv5, 
            csv3, 
            csv8, 
            csv4, 
            csv9,
            # csv11
        ]


# plot lines
labels = [
    'ALG(MLP)', 'ALG(MLP)_NoHost', 
    'ALG(GCN)', 'ALG(GCN)_NoHost', 
    'Baseline_Entropy', 'Baseline_Entropy_NoHost', 
    'Baseline_Random', 'Baseline_Random_NoHost', 
    # 'ALG(GCN)_Inverse'
    ]
markers = ['o', 'v', 's', 'p', '*', 'x', 'D', 'h', 'X', 'd']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
cnt = 0
num_min = 100
num_max = 0
for j in csv_list:
    # if 'nohost' in j and cnt != 3:
    #     cnt += 1
    #     continue
    # if cnt == 8:
    #     cnt+=1
    #     continue
    data = pd.read_csv(j)
    plt.plot(data['num'].values, data['acc'].values, label = labels[cnt], color = colors[cnt], marker= markers[cnt])
    num_min = min(num_min, data['num'].values[0])
    num_max = max(num_max, data['num'].values[-1])
    cnt+=1

baseline1 = pd.read_csv(csv6)['acc'].values[0]
num1 = pd.read_csv(csv6)['num'].values[0]
plt.axhline(y=baseline1, color='grey', linestyle='--', label = 'Baseline_UseAll ({})'.format(num1))
baseline2 = pd.read_csv(csv7)['acc'].values[0]
num2 = pd.read_csv(csv7)['num'].values[0]
plt.axhline(y=baseline2, color='black', linestyle='-', label = 'Baseline_UseAll_NoHost ({})'.format(num2))

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
plt.savefig('figs/test/seed_{}_{}_al.png'.format(seed, taxa_name))