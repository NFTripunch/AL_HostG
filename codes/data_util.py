import sys
import pickle as pkl
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import argparse
import pandas as pd



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(path="../data", dataset="hostg.phylum"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    
    indices = [idx_host, initially_labeled_virus_node, unlabeled_virus_node, test_virus_node]
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for i in range(len(names)):
        if "hostg" in dataset:
            data_path = "{}/hostg/{}/ind.{}.{}".format(path, dataset.split('.')[1], dataset, names[i])
        else:
            data_path = "{}/ind.{}.{}".format(path, dataset, names[i])
        with open(data_path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    if "hostg" in dataset:
        test_idx_reorder = parse_index_file("{}/hostg/{}/ind.{}.test.index".format(path, dataset.split('.')[1], dataset))
    else:
        test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))
    features = normalize(features)
    print("| # of features : {}".format(features.shape))
    print("| # of classes   : {}".format(ally.shape[1]))
    print("number of nan: {}".format(np.count_nonzero(np.isnan(ally))))
    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1] # onehot to label encoding 
    
    labels = torch.LongTensor(np.where(labels)[1])
    print("Label distribution of the dataset:")
    print(pd.value_counts(np.array(labels)))
    idx_train = range(len(y)) # only 20 * num_classes labeled data index may be wrong
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of maximum train set : {}".format(len(idx_train))) # 20 * num_classes
    print("| # of all train set (including the val set): {}".format(ally.shape[0]))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))
    # print("Intersection of the train, val, and test sets:")
    # print(set(idx_train).intersection(idx_val))
    # print(set(idx_train).intersection(idx_test))
    # print(set(idx_val).intersection(idx_test))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)
    print("| # of labels : {}".format(labels.shape))
    return adj, features, labels, idx_train, idx_val, idx_test

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='hostg.phylum', help='dataset')
# args = parser.parse_args()
# load_data(dataset = args.dataset)