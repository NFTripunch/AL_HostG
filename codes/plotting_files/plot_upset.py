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

# Load selected nodes
name = 'nohost'
seeds = [4, 13, 42, 71, 265]

selected = {}
cnt = 0
for s in seeds:
    idx = pkl.load(open('selected/{}_{}_{}_selected.pkl'.format(taxa, name, s), 'rb'))
    print(len(idx))
    selected[s] = idx


nodes = from_contents(selected)
print(nodes)
from upsetplot import plot
plot(nodes, sort_by='cardinality', show_counts='%d', show_percentages=True, element_size=50)
from matplotlib import pyplot as plt 
plt.title("Taxa: {}, Model name: {}".format(taxa, name))
plt.savefig('figs/upset/upset_{}_{}.png'.format(taxa, name))