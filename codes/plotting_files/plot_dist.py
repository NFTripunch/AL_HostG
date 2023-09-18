import pickle as pkl
import numpy as np 
import configparser
import pandas as pd
from data_util import load_data
import random 
import torch 
import seaborn as sns
import matplotlib.pyplot as plt
from upsetplot import from_contents 
# Load data 
config = configparser.ConfigParser()
config.read('configs.ini')
taxa = config.get('model', 'taxa')
seed = config.get('model', 'seed')
special = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if special:
    # Special case for class 
    index  =[i for i in range(11)]
    # count = [776, 504, 431, 125, 90, 62, 48, 49, 29, 20, 11]
    class_weight_w1 = [0.2740, 1.0000, 0.7407, 0.8696, 1.1765, 1.8182, 2.0000, 1.6667, 1.6667,
        2.0000, 4.0000]
    class_weight_w2 = [0.6143, 0.8771, 0.9535, 0.8305, 1.0000, 1.1547, 1.4142, 1.0847, 1.2403,
        1.4142, 2.0000]
    plt.bar(index, class_weight_w2)
    plt.xticks(index)
    plt.title("Class weight taxa: {}".format('class'))
    plt.savefig('figs/paper/skw2_dist.png')
else:
    # Load selected nodes
    name = 'randomnohost'
    seeds = [4, 13, 42, 71, 265]
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="hostg.{}".format(taxa))
    idx = pkl.load(open('selected/{}_{}_{}_selected.pkl'.format(taxa, name, seed), 'rb'))

    labels_selected = labels[idx]
    # sort by number of nodes using numpy and pandas and plot in bar plot 

    sorted = pd.Series(labels_selected.cpu().numpy()).value_counts().sort_index()
    plt.bar(sorted.index, sorted.values)
    plt.xticks(sorted.index)
    plt.title("Taxa: {}, Model name: Random_NoHost, Seed: {}".format(taxa, seed))
    plt.savefig('figs/paper/{}_{}_{}_selected.png'.format(taxa, name, seed))
