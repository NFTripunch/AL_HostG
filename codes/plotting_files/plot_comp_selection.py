import pickle as pkl
import numpy as np 
import configparser
import pandas as pd
from data_util import load_data
import random 
import torch 
import seaborn as sns
import matplotlib.pyplot as plt

# Load data 
config = configparser.ConfigParser()
config.read('configs.ini')
taxa = config.get('model', 'taxa')
seed = config.get('model', 'seed')
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="hostg.{}".format(taxa))
seed= int(seed)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Load selected nodes
name = 'nohost'
seeds = [4, 13, 42, 71, 265]
data = []
len_labeled = len(labels)
# features = features[:len_labeled]

# """
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

kpca = KernelPCA(n_components=50, kernel='rbf')
features = kpca.fit_transform(features.cpu().numpy())
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(features)
new_labels = np.zeros(len(labels), dtype = int)
new_labels[idx_train] = labels[idx_train] + 1 
# selected = pkl.load(open('selected/{}_{}_selected.pkl'.format(name, seed), 'rb'))
selected = []
importance = np.zeros(len(labels), dtype = int)
labels_counts = np.zeros(len(seeds))
cnt = 0
for s in seeds:
    idx = pkl.load(open('selected/{}_{}_selected.pkl'.format(name, s), 'rb'))
    importance[idx] += 1
    labels_counts[cnt] = len(idx)
    cnt+=1
idx = [i for i in range(len(labels)) if importance[i] > 0]
new_labels = importance[idx]

choice = 0
if choice == 0:
    palette = sns.color_palette("Set2", len(set(new_labels)))
    df = pd.DataFrame({'tsne-2d-one': tsne_results[idx,0], 'tsne-2d-two': tsne_results[idx,1], 'y': new_labels.astype(int)})
else:
    palette = sns.color_palette("husl", len(set(labels.cpu().numpy()))),
    df = pd.DataFrame({'tsne-2d-one': tsne_results[:len(labels),0], 'tsne-2d-two': tsne_results[:len(labels),1], 'y': labels})
plt.figure(figsize = (6,6))
# fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
grey = [(0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

g = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    # ax = axes[0],
    hue="y",
    # palette=grey+color_list,
    # palette = sns.color_palette("Paired", len(set(new_labels))),
    # palette = sns.color_palette("YlOrBr", len(set(new_labels))),
    # palette = sns.color_palette("Set2", len(set(new_labels))),
    palette = palette,
    # style = 'y', 
    data=df,
    legend="full",
    alpha=0.8
)
sns.move_legend(g, loc='upper left', ncol=1)

# figure 2 
# sns.histplot(ax = axes[1], x=seeds, y=labels_counts, legend = 'full')
if choice == 0:
    plt.savefig('figs/comp_{}_{}_tsne.png'.format(seed, taxa,name))
else:
    plt.savefig('figs/seed_{}_{}_{}_tsne.png'.format(seed, taxa,name))
"""
fig = plt.figure()
ax = fig.add_subplot(111)
 
# for every class, we'll add a scatter plot separately
for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]
 
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
 
    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=np.float) / 255
 
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)
 
# build a legend using the labels we set previously
ax.legend(loc='best')
 
# finally, show the plot
plt.show()
"""

# for s in seeds:
#     data.append(pkl.load(open('selected/{}_{}_selected.pkl'.format(name, s), 'rb')))
# data = np.array(data)

 

# sorted = pd.Series(data.flatten()).value_counts().sort_index()
# print(sorted.shape)
# print(sorted[sorted == 5].shape)
# # As a result, it is possible that different runs give you different solutions. Notice that it is perfectly fine to run t-SNE a number of times (with the same data and parameters), and to select the visualization with the lowest value of the objective function as your final visualization.

