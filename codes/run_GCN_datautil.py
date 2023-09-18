import  numpy as np

import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F


from    data import load_data, preprocess_features, preprocess_adj, sample_mask
import  model
from    config import  args
from    utils import masked_loss, masked_acc, masked_ECE
import  pickle as pkl
import pandas as pd
import  scipy.sparse as sp
import argparse
from scipy.special import softmax
from sklearn.metrics import classification_report
from collections import Counter
import networkx as nx

import random


################################################################################
###############################  Input Params  #################################
################################################################################

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.set_device(args.gpus)
else:
    print("Running with cpu")

################################################################################
############################  Loading dataset  #################################
################################################################################

def GCN_split(taxa):
    print("Spliting taxa: {}".format(taxa))
    adj        = pkl.load(open("GCN_data/"+taxa+"_contig.graph",'rb'))
    labels     = pkl.load(open("GCN_data/"+taxa+"_contig.label",'rb'))
    print(pd.value_counts(labels))
    features   = pkl.load(open("GCN_data/"+taxa+"_contig.feature",'rb'))
    print("| # of edges : {}".format(adj.sum().sum() / 2))
    features = sp.csc_matrix(features)
    # idx_test   = pkl.load(open("GCN_data/"+taxa+"_contig.test_id",'rb'))

    # idx_test = np.array(idx_test, dtype=np.int64)
    labels = np.array(labels)
    labels = np.squeeze(labels)
    print("Number of unlabeled nodes in the graph: {}".format(np.count_nonzero(np.isnan(labels))))
    num_classes = len(np.unique(labels))
    # print("Before one-hot encoding labels.shape: {}".format(labels.shape))
    # TODO: one-hot encoding of labels
    labels = pd.get_dummies(labels).values
    # print("After one-hot encoding labels.shape: {}".format(labels.shape))
    len_test = 500
    len_allx = len(labels) - len_test
    # TODO: the num_classes are [10, 19, 43, 78, 136] for [phylum, class, order, family, genus]
    maximum_num_per_class = 20
    len_train = num_classes * maximum_num_per_class

    initial_node_index = [] # initially labeled 
    initial_num_per_class = 2
    for i in range(num_classes):
        initial_node_index.append(np.where(labels[:, i] == 1)[0])
    tmp_node_index = []
    ori_node_index = [i for i in range(len(labels))]
    print(len(ori_node_index))
    for i in range(num_classes):
        append_list = []
        if (len(initial_node_index[i]) > initial_num_per_class):
            append_list = random.sample(list(initial_node_index[i]),initial_num_per_class)
        else:
            append_list = initial_node_index[i]
        for i in append_list:
            # O(num_classes * initial_num_per_class)
            ori_node_index.remove(i)
            tmp_node_index.append(i)
            # print("remove: ", i)
    # print("length comp: ", len(tmp_node_index), len(ori_node_index))
    perm_node_index = tmp_node_index + np.random.permutation(ori_node_index).tolist()
    indices = np.array(perm_node_index)
    # TODO: find the permutation matrix
    permutation_matrix = np.zeros((labels.shape[0], labels.shape[0]), dtype = int)
    for i in range(labels.shape[0]):
        permutation_matrix[indices[i]][i] = 1
    assert (indices == np.matmul(np.arange(labels.shape[0]).reshape(1, -1), permutation_matrix).reshape(-1)).all()
    
    # permutation_matrix = np.random.permutation(np.identity(labels.shape[0], dtype = int))
    # indices = np.matmul(np.arange(labels.shape[0]).reshape(1, -1), permutation_matrix).reshape(-1)
    
    
    # TODO: reorder feature, adj matrix
    # complexity O(n^2)
    perm_csc = sp.csr_matrix(permutation_matrix)
    adj = perm_csc @ adj @ perm_csc.transpose()
    # adj = perm_csc * adj * perm_csc.transpose() # though doc says it is element-wise multiplication, it's still working
    # adj = perm_csc.multiply(adj) # element-wise multiplication
    # adj = adj.multiply(perm_csc.transpose())
    print("| # of edges after permutation: {}".format(adj.toarray().sum().sum() / 2))
    adj = nx.to_dict_of_dicts(nx.from_scipy_sparse_matrix(adj))
    idx_train, idx_allx, idx_test = indices[:len_train], indices[:len_allx], indices[len_allx:]
    
    x, allx, tx = features[idx_train], features[idx_allx], features[idx_test]
    y, ally, ty = labels[idx_train], labels[idx_allx], labels[idx_test]
    
    # idx_train = np.array([i for i in range(len(labels)) if i not in idx_test])
    print('features:', features.shape)
    print('labels:', labels.shape)
    print("num_classes: {}".format(num_classes))
    print("idx_train.shape: {}, idx_allx.shape: {}, idx_test.shape: {}".format(idx_train.shape, idx_allx.shape, idx_test.shape))
    print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))
    print("allx.shape: {}, ally.shape: {}".format(allx.shape, ally.shape))
    print("tx.shape: {}, ty.shape: {}".format(tx.shape, ty.shape))
    # print(type(x), type(allx), type(adj))
    pkl.dump(x, open("GCN_data/ind.hostg."+taxa+".x", "wb" ))
    pkl.dump(y, open("GCN_data/ind.hostg."+taxa+".y", "wb" ))
    pkl.dump(allx, open("GCN_data/ind.hostg."+taxa+".allx", "wb" ))
    pkl.dump(ally, open("GCN_data/ind.hostg."+taxa+".ally", "wb" ))
    pkl.dump(adj, open("GCN_data/ind.hostg."+taxa+".graph", "wb" ))
    pkl.dump(tx, open("GCN_data/ind.hostg."+taxa+".tx", "wb" ))
    pkl.dump(ty, open("GCN_data/ind.hostg."+taxa+".ty", "wb" ))
    f = open("GCN_data/ind.hostg."+taxa+".test.index", "w")
    for i in idx_test:
        f.write("{}\n".format(i))


taxa_list = ["phylum", 'class', 'order', 'family', 'genus']
# for taxa in taxa_list:
#     GCN_split(taxa)

GCN_split('phylum')
