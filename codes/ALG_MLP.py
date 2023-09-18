import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
from scipy.sparse import csgraph
import sys
import pickle
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch.optim import lr_scheduler
import copy
import json
import sklearn
from utils import *
from cluster import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.backends import cudnn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from data_util import load_data
from models import GCN, MLP

cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
name = 'algmlp'


import configparser
config = configparser.ConfigParser()
config.read('configs.ini')
taxa = config.get('model', 'taxa')

def argument_parser():
    # Data processing
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=128, help='batch size')
    parser.add_argument('--dataset', type=str, default='citeseer', choices = ['hostg.phylum', 'hostg.genus','hostg.class','hostg.family','hostg.order','citeseer'])
    parser.add_argument('--num_classes', type=int, default = 6, help='number of classes')
    parser.add_argument('--num_nodes', type=int, default = 3327, help='number of nodes')
    parser.add_argument('--seeds', type=int, default = 4, help='seeds')
    parser.add_argument('--no_hosts', action = 'store_true', help='whether to inlcude host nodes in the initially labeled datasets')
    return parser


parser = argument_parser()
args = parser.parse_args()

if args.no_hosts:
    name += "_nohost"

MLPPATH = 'logs/{0}/{1}_MLP_checkpoint.pt'.format(name, 'ALG_MLP')
GCNPATH = 'logs/{0}/{1}_GCN_checkpoint.pt'.format(name, 'ALG_MLP')

def main():  
    """
    ------------------------------------------------------------------------
    --------------------- Model Training -----------------------------------
    ------------------------------------------------------------------------ 
    """

    set_seed(args.seeds)
    print("Running with seed: ", args.seeds)
    T = 19 
    if args.dataset == 'hostg.phylum':
        max_node = 2286 
        # num_hosts = 118 
        # num_class = 10
        # num_virus = 18
        num_node = 2170
        num_hosts = 11
        num_class = 7
        num_virus = 14
        max_num_per_class = 20
    elif args.dataset == 'hostg.genus':
        max_node = 2286 
        # num_hosts = 118 
        # num_class = 135
        # num_virus = 239
        num_node = 1947
        num_hosts = 46
        num_class = 41
        num_virus = 82
        max_num_per_class = 20

    elif args.dataset == 'hostg.family':
        max_node = 2286 
        # num_hosts = 118 
        # num_virus = 139
        # num_class = 77
        num_node = 2119
        num_hosts = 54
        num_class = 39
        num_virus = 78
        max_num_per_class = 20
        
    elif args.dataset == 'hostg.order':
        max_node = 2286 
        # num_hosts = 118 
        # num_class = 42
        # num_virus = 76
        num_node = 2168 
        num_hosts = 39 
        num_class = 25
        num_virus = 50
        max_num_per_class = 20
    
    elif args.dataset == 'hostg.class':
        max_node = 2286 
        # num_hosts = 118 
        # num_class = 19
        # num_virus = 35
        num_node = 2163 
        # num_hosts = 114
        num_hosts = 18 
        num_class = 11
        num_virus = 22
        max_num_per_class = 20

    else:
        max_node = args.num_nodes
        num_node = args.num_nodes
        num_class = args.num_classes
        max_num_per_class = 20

    initial_num_per_class = 2
    ititial_labeled_nodes = num_hosts + num_virus
    num_for_label = num_class*max_num_per_class # maximum labeled nodes for training 
    print("Running ALG model")

    print("T:",T)
    print("Number of classes:",num_class)
    print("Total number of nodes:",num_node)
    print("The number of nodes for label:",num_for_label)
    print("initial_num_per_class:",initial_num_per_class)
    print("ititial_labeled_nodes:",ititial_labeled_nodes)
    
    total_training_epochs = 1500 
    
    proxy_dp = 0.6
    proxy_lr = 0.2
    # hidden_size = 128

    batch_sizes = np.ones(T)*2
    batch_sizes = np.array([2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    print('batch_sizes_per_class:',batch_sizes )
    print('batch_sizes:',batch_sizes*num_class )

    alphas = [] 
    for temp_T in range(T+1):
        alpha = np.cos(np.pi*temp_T/(2*(T)))
        alphas.append(alpha)
    print('alphas:',alphas)

    batch_epochs = get_adaptive_training_epochs(total_training_epochs,num_iterations = T, m_T=0.3,m_1=1)
    print('batch_epochs:',batch_epochs)

    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)
    import configparser
    config = configparser.ConfigParser()
    config.read('configs.ini')
    dropout_rate = float(config.get('model', 'dropout'))
    print("tested on a dropout graph with rate: ", dropout_rate) 
    G = nx.from_numpy_matrix(adj.toarray())
    G.remove_edges_from(random.sample(G.edges(),k=int(dropout_rate*G.number_of_edges())))
    adj = nx.adjacency_matrix(G)
    print("Number of edges after deletion: ", adj.sum().sum()/2)
    # print("Degree of node 0: ", adj[0].sum())
    # print("Degree of node -1: ", adj[-1].sum())
    
    labels_npy = labels.cpu().numpy()

    if args.no_hosts:
        # Exclude host nodes
        initial_node_index = list([i for i in range(num_hosts, num_hosts + num_virus)])
    else:
        initial_node_index = list([i for i in range(num_hosts + num_virus)])
    
    # print("Initial index: ", initial_node_index)
    nodes_index = []
    for i in range(num_node): 
        if i not in initial_node_index:
            nodes_index.append(i)
    nodes_index = list(initial_node_index)+nodes_index

    idx_available = []
    if args.no_hosts:
        lb = num_hosts
    else:
        lb = 0
    for i in range(lb, num_node):
        if i in initial_node_index:
            continue
        if i not in idx_val and i not in idx_test:
            idx_available.append(i)


    # print(nodes_index)
    idx_train = torch.LongTensor(nodes_index[:len(initial_node_index)])
    # temporarily unlabeled nodes
    idx_unlabeled = nodes_index[len(initial_node_index):] 
    ori_ini_node_index = copy.deepcopy(initial_node_index)
    # Adj operation
    # adj = adj ^ K 
    adj = aug_normalized_adjacency(adj)
    adj_tensor = torch.FloatTensor(adj.todense())
    adj_tensor = adj_tensor.cuda()
    adj_matrix = np.array(torch.mm(adj_tensor,adj_tensor).cpu()) 
    adj_tensor = torch.mm(adj_tensor,adj_tensor) 
    adj = sparse_mx_to_torch_sparse_tensor(adj).float() 
    adj = adj.cuda()


    features_GCN = copy.deepcopy(features)
    # print("features shape: ", features.shape)
    
    # MLP aggregated features
    features = features.cuda()
    features = np.array(torch.mm(adj_tensor, features).cpu()) 
    # print("idx train: ", idx_train)
    # print("features shape", features.shape)
    train_label = np.array(labels[idx_train]).reshape(-1,1)
    train_embeddings = np.array(features[idx_train])
    # print("Shape comp: {}, {}".format(train_embeddings.shape,train_label.shape))

    labeled_data = np.hstack((train_embeddings,train_label))
    unlabeled_data =  np.array(features[idx_unlabeled])

    # Semi-supervised K-means for clustering
    clusterAssment,centroids = semi_kMeans(labeled_data,unlabeled_data)
    print(clusterAssment.shape)
    print(centroids.shape)
    dic = {} 
    i = 0
    for index in nodes_index:
        dic[index] = clusterAssment[i]
        i+=1
    dic = sorted(dic.items(), key=lambda d: d[0])
    clusterAssment = np.array([ key_value[1] for key_value in dic])
    print(clusterAssment.shape)
    
    dist = [{} for j in range(num_class)]
    for i in range(num_node):
        clus_i = int(clusterAssment[i]) 
        centroids_j = centroids[clus_i] 
        dist_ij = distEclud(features[i],centroids_j) 
        dist[clus_i][i] = dist_ij 

    normalized_similarities = {}
    for i in range(num_class):
        normalized_similarity = get_normalized_similarity(dist[i])
        normalized_similarities.update(normalized_similarity)

    features = torch.FloatTensor(features).cuda()
    features_GCN = torch.FloatTensor(features_GCN).cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    labels_npy = labels.cpu().numpy()
    features_npy = features.cpu().numpy()
    """
    ------------------------------------------------------------------------
    --------------------- Model Training -----------------------------------
    ------------------------------------------------------------------------ 
    """
    # Section 2 MLP training for approximate QBC
    # Iteration 0
    """
    Train one iteration first so that in the next round there will be at two committee for calculating inf(v_i)
    """


    num_epochs = batch_epochs[0]
    model1 = MLP(nfeat=features.shape[1],
            nhid = args.hidden_size,
            nclass=labels.max().item() + 1,
            dropout=proxy_dp)
    model1.cuda()
    optimizer = optim.Adam(model1.parameters(),lr=0.2, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopper(patience=30, min_delta = 0.01)
    t_total = time.time()
    record = {}
    for epoch in range(num_epochs):
        cur_model, loss_val = train(epoch+1, model1,optimizer,features,labels,idx_train, idx_val, idx_test, record)
        
        if early_stopping.early_stop(cur_model, loss_val, PATH = MLPPATH):
            print("Early stopped at epoch: ", epoch)
            # load best model
            model1.load_state_dict(torch.load(MLPPATH))
            break
        
        scheduler.step()
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    bit_list = sorted(record.keys())
    bit_list.reverse()
    for key in bit_list[:10]:
        value = record[key]
        print(key,value)

    # Evaluation 1 
    with torch.no_grad():
        output = model1(features)
        # A minor difference is that the implementation of CrossEntrypyLoss 
        # implicitly applies a softmax activation followed by a log transformation but NLLLoss does not
        # loss_val1 = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val1 = accuracy(output[idx_val], labels[idx_val])

        # loss_test1 = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test1 = accuracy(output[idx_test], labels[idx_test])

        print("Result of 1st iteration of MLP",'acc_val: {:.4f}'.format(acc_val1),
            'acc_test: {:.4f}'.format(acc_test1))
        if not isinstance(acc_val1, np.floating):
            weight1 = 0.5*np.log(acc_val1.cpu().numpy() /(1-acc_val1.cpu().numpy() ))
        else: 
            weight1 = 0.5*np.log(acc_val1/(1-acc_val1))
        output1 = F.softmax(model1(features),1).cpu().detach().numpy()
    
    ensemble_output = output1
    normalized_differences = np.zeros(num_node) 

    predicted_class = np.argmax(ensemble_output,1)
    pd.Series(predicted_class).value_counts().sort_index()

    alpha = alphas[0]
    weighted_score = {}
    cluster_index = []
    for i in range(num_class):
        cluster_index.append({})

    # min_score = 100000
    # max_score = 0
    for node in range(num_node):
        node_rep = normalized_similarities[node]
        node_inf = normalized_differences[node]
        weighted_score[node] = alpha*node_rep+(1-alpha)*node_inf
        # if weighted_score[node] < min_score:
        #     min_score = weighted_score[node]
        # if weighted_score[node] > max_score:
        #     max_score = weighted_score[node]
    
    for node in range(num_node, max_node):
        weighted_score[node] = 0
    
    # print("min_score:",min_score)
    # print("max_score:",max_score)
    
    for node in idx_available:
        cluster = predicted_class[node] 
        node_rep = normalized_similarities[node]
        node_inf = normalized_differences[node]
        cluster_index[cluster].update({node:weighted_score[node]})

    for i in range(num_class):
        cluster_index[i] = sorted(cluster_index[i].items(), key=lambda d: d[1], reverse=True)

    batch_size = batch_sizes[1]
    num_labeled_nodes_per_class = pd.Series(labels[idx_train].cpu().numpy()).value_counts().sort_index().values
    print("number of labeled nodes per class:",num_labeled_nodes_per_class)
    print("Batch size:",batch_size)
    num_labeled_nodes_per_class = len(idx_train)/num_class-num_labeled_nodes_per_class+batch_size
    num_labeled_nodes_per_class = num_labeled_nodes_per_class.astype(int)
    # ensure all class has non-negative labeled node
    print("number of nodes to be labeled per class:",num_labeled_nodes_per_class)
    cut_zero(num_class, num_labeled_nodes_per_class) 
    # number of available nodes per class 
    class_pd = pd.Series(predicted_class[idx_available]).value_counts().sort_index()
    print("number of available nodes per class:\n",class_pd)
    index = class_pd.index # class index
    values = class_pd.values # num nodes
    nodes_per_class = np.zeros(num_class)
    for i in range(len(index)):
        nodes_per_class[index[i]] = values[i]
    print("number of labeled nodes per class after cut_zero:",num_labeled_nodes_per_class)
    
    average_ab(num_labeled_nodes_per_class,nodes_per_class)
    print("number of labeled nodes per class after average_ab:",num_labeled_nodes_per_class)


    # Node Selection 
    t = time.time()
    print("iniitially labeled len of idx_train:",len(idx_train))
    test_accs = []
    idx_train_list = [len(idx_train)]
    test_acc = evaluate(args.hidden_size, features_GCN, labels, idx_train, idx_val, idx_test, adj)
    # test_accs.append(test_acc.cpu().numpy())
    test_accs.append(test_acc)
    idx_train, idx_available = node_selection(idx_train, idx_available, num_class, predicted_class, weighted_score, num_labeled_nodes_per_class, adj_matrix)
    print("len of idx_train after node selection 1:",len(idx_train))
    print('node selection finished')
    if not isinstance(acc_val1, np.floating):
        weight1 = 0.5*np.log(acc_val1.cpu().numpy() /(1-acc_val1.cpu().numpy() ))
    else: 
        weight1 = 0.5*np.log(acc_val1 /(1-acc_val1))
    output1 = F.softmax(model1(features[:num_node]),1).cpu().detach().numpy()
    weights = [weight1]
    outputs = [output1]

    
    idx_train_list.append(len(idx_train))
    test_acc = evaluate(args.hidden_size, features_GCN, labels, idx_train, idx_val, idx_test, adj)
    test_accs.append(test_acc)
    # test_accs.append(test_acc.cpu().numpy())
    model = copy.deepcopy(model1)
    for iteration in range(1,T-1):
        # Iteration 2 ~ T - 1
        print('iteration:', iteration+1)
        initial_node_index = copy.deepcopy(idx_train)
        nodes_index = []
        for i in range(num_node): 
            if i not in initial_node_index:
                nodes_index.append(i)
        nodes_index = initial_node_index+nodes_index
        idx_train = torch.LongTensor(nodes_index[:len(initial_node_index)])
        idx_unlabeled = nodes_index[len(initial_node_index):] 
        train_label = np.array(labels_npy[idx_train]).reshape(-1,1)
        train_embeddings = np.array(features_npy[idx_train])


        labeled_data = np.hstack((train_embeddings,train_label))
        unlabeled_data =  np.array(features_npy[idx_unlabeled])

        # model-free representativeness
        clusterAssment,centroids = semi_kMeans(labeled_data,unlabeled_data)
        dic = {} 
        i = 0
        for index in nodes_index:
            dic[index] = clusterAssment[i]
            i+=1
        dic = sorted(dic.items(), key=lambda d: d[0])
        clusterAssment = np.array([ key_value[1] for key_value in dic])

        dist = [{} for j in range(num_class)]
        for i in range(num_node):
            clus_i = int(clusterAssment[i]) 
            centroids_j = centroids[clus_i] 
            dist_ij = distEclud(features_npy[i],centroids_j) 
            dist[clus_i][i] = dist_ij 

        normalized_similarities = {}
        for i in range(num_class):
            normalized_similarity = get_normalized_similarity(dist[i])
            normalized_similarities.update(normalized_similarity)
            
        # Training MLP
        idx_train = idx_train.cuda()
        best_val_acc = 0
        num_epochs = batch_epochs[iteration]
        model.cuda()
        # Incremental training 
        optimizer = optim.Adam(model.parameters(),lr=0.2, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        t_total = time.time()
        record = {}
        # Use MLP to calculate informativeness

        for epoch in range(num_epochs):
            cur_model, loss_val = train(epoch+1, model,optimizer,features,labels,idx_train, idx_val, idx_test, record)
            
            if early_stopping.early_stop(cur_model, loss_val, PATH = MLPPATH):
                print("Early stopped at epoch: ", epoch)
                # load best model
                model.load_state_dict(torch.load(MLPPATH))
                break
            
            scheduler.step()
        # Forward pass
        with torch.no_grad():
            output = model(features)
            
            # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

            # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print('Iteration: {:04d}'.format(iteration+1),
                'acc_val: {:.4f}'.format(acc_val),
                'acc_test: {:.4f}'.format(acc_test))

            output = F.softmax(model(features[:num_node]),1).cpu().detach().numpy()
            outputs.append(output)
        if not isinstance(acc_val, np.floating):
            weight = 0.5*np.log(acc_val.cpu().numpy() /(1-acc_val.cpu().numpy() ))
        else: 
            weight = 0.5*np.log(acc_val /(1-acc_val))
        weights.append(weight)
        sum_weighted_dist = np.zeros(num_node)

        for i in range(iteration+1):
            for j in range(i+1,iteration+1):
                # different committee members are just models in different iterations
                weighted_dist = get_weighted_dist(num_node, outputs[i],outputs[j],alphas[i],alphas[j])
                # weighted_dist = get_weighted_dist(num_node, outputs[i],outputs[j],weights[i],weights[j])
                sum_weighted_dist+=weighted_dist

        print("Weights before softmax transformation: ", weights)
        soft_weights = F.softmax(torch.from_numpy(np.array(weights)).to(device),0).cpu().numpy()
        print("Weights after softmax transformation: ", soft_weights)
        ensemble_output = np.zeros((num_node,num_class))

        for i in range(len(weights)):
            ensemble_output += soft_weights[i]*outputs[i]
        predicted_class = np.argmax(ensemble_output,1)

        dist = [{} for _ in range(num_class) ]
        for i in range(num_node):
            clus_i = int(predicted_class [i]) 
            dist[clus_i][i] = sum_weighted_dist[i] 
        normalized_differences = {}
        for i in range(num_class):
            normalized_difference = get_normalized_difference(dist[i])
            normalized_differences.update(normalized_difference)

        
        alpha = alphas[iteration]
        weighted_score = {}
        for node in range(num_node):
            node_rep = normalized_similarities[node]
            node_inf = normalized_differences[node]
            weighted_score[node] = alpha*node_rep+(1-alpha)*node_inf
            # if weighted_score[node] < min_score:
            #     min_score = weighted_score[node]
            # if weighted_score[node] > max_score:
            #     max_score = weighted_score[node]
        for node in range(num_node, max_node):
            weighted_score[node] = 0
    

        batch_size = batch_sizes[iteration]
        num_labeled_nodes_per_class = pd.Series(labels[idx_train].cpu().numpy()).value_counts().sort_index().values
        print("Current number of labeled nodes per class", num_labeled_nodes_per_class)
        num_labeled_nodes_per_class = len(idx_train)/num_class-num_labeled_nodes_per_class+batch_size
        print("Number of labeled nodes per class", num_labeled_nodes_per_class)

        num_labeled_nodes_per_class = num_labeled_nodes_per_class.astype(int)
        # ensure all class has non-negative labeled node 
        print("number of labeled nodes per class before cut_zero:",num_labeled_nodes_per_class)
        cut_zero(num_class, num_labeled_nodes_per_class)
        # One possible explanation is that during the training process, we cannot access the labels of unlabled nodes 
        class_pd = pd.Series(predicted_class[idx_available]).value_counts().sort_index()
        # class_pd = pd.Series(labels[idx_available].cpu().numpy()).value_counts().sort_index()
        print("number of available nodes per class:\n",class_pd)
        index = class_pd.index
        values = class_pd.values
        nodes_per_class = np.zeros(num_class)
        for i in range(len(index)):
            nodes_per_class[index[i]] = values[i]

        # ensure that number of nodes to be labeled doesn't exceed the number of available nodes
        print("number of labeled nodes per class after cut_zero:",num_labeled_nodes_per_class)
        
        average_ab(num_labeled_nodes_per_class,nodes_per_class)
        print("number of labeled nodes per class after average_ab:",num_labeled_nodes_per_class)

        print("len of idx_train: {} before node selection {}".format(len(idx_train), iteration + 1))
        print("Actual labels of idx_train selected nodes: \n", pd.Series(labels[idx_train].cpu().numpy()).value_counts().sort_index())
        idx_train, idx_available = node_selection(idx_train, idx_available, num_class, predicted_class, weighted_score, num_labeled_nodes_per_class, adj_matrix)
        print("len of idx_train: {} after node selection {}".format(len(idx_train), iteration + 1))
        print("Actual labels of idx_train selected nodes: \n", pd.Series(labels[idx_train].cpu().numpy()).value_counts().sort_index())
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  Evaluation begin  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        test_acc = evaluate(args.hidden_size, features_GCN, labels, idx_train, idx_val, idx_test, adj)
        # test_accs.append(test_acc.cpu().numpy())
        test_accs.append(test_acc)
        idx_train_list.append(len(idx_train))
        print("Test accuracy of model with best validation metric: \n{}".format(np.array(test_accs)))

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  Evaluation end  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # Model Evaluation 
    print("******* Model Evaluation ******")
    evaluate(args.hidden_size, features_GCN, labels, idx_train, idx_val, idx_test, adj)
    
    for i in idx_train:
        if i in ori_ini_node_index:
            idx_train.remove(i)
    file = open('selected/{0}_{1}_{2}_selected.pkl'.format(taxa, name, args.seeds), 'wb')
    pickle.dump(np.array(idx_train), file)
    file.close()
    pd.DataFrame({"num": idx_train_list, "acc": test_accs}).to_csv("logs/{}/data.csv".format(name))
    
def node_selection(idx_train, idx_available, num_class, predicted_class, weighted_score, num_labeled_nodes_per_class, adj_matrix):
    idx_train = list(idx_train.cpu().numpy())
    idx_available_temp = copy.deepcopy(idx_available)
    print("len of idx_available_temp initial: {}".format(len(idx_available_temp)))
    remove_list = []
    for i in idx_available_temp:
        if num_labeled_nodes_per_class[predicted_class[i]] == 0:
            remove_list.append(i)

    for node in remove_list:
        idx_available_temp.remove(node)
    count = np.zeros(num_class) 
    while True:
        print("len of idx_available_temp: {}".format(len(idx_available_temp)))
        max_reception_node, max_field = get_max_receptive_node_dense(idx_train,idx_available_temp,weighted_score, adj_matrix)
        if max_reception_node == -1:
            print("Receptive field is ", max_field)
            print("node is invalid")
            break
        if max_reception_node not in idx_available_temp:
            print("Max reception node {} not in index pool".format(max_reception_node))
            print(pd.Series(predicted_class[idx_available]).value_counts().sort_index())
        node_class = predicted_class[max_reception_node]
        if count[node_class] < num_labeled_nodes_per_class[node_class]:
            idx_train.append(max_reception_node) 
            print("New node added with receptive field: {}, class :{}".format(max_field, node_class))
            idx_available.remove(max_reception_node)
            if max_reception_node in idx_available_temp:
                idx_available_temp.remove(max_reception_node)
            count[node_class] +=1
        else:
            print("No quota for this class with quota {}".format(num_labeled_nodes_per_class[node_class]))
            print("Node with receptive field: {}, class :{} node: {}".format(max_field, node_class, max_reception_node))
            idx_available_temp.remove(max_reception_node)
            node_class = predicted_class[max_reception_node]
            remove_list = []
            for i in idx_available_temp:
                if predicted_class[i] == node_class:
                    remove_list.append(i)
            for node in remove_list:
                idx_available_temp.remove(node)  
        if np.sum(count) >=np.sum(num_labeled_nodes_per_class):
            break
    return idx_train, idx_available

def train(epoch, model, optimizer, features, labels, idx_train, idx_val, idx_test, record, adj = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if adj is None:
        output = model(features)
    else:
        output = model(features, adj)

    class_weight = cal_class_weight(idx_train, labels)
    
    # Method 1 use unweighted CE 
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    
    # Method 2 use weighted CE where the weight is the sqrt version of weights by sklearn
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train], weight= None)
    
    # Method 3 Use focal loss
    focal_loss = FocalLoss(gamma = 2)
    # loss_train = focal_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    # validation
    model.eval()
    with torch.no_grad(): 
        if adj is None:
            output = model(features)
        else:
            output = model(features, adj)
        val_class_weight = cal_class_weight(idx_val, labels)
        loss_val = F.cross_entropy(output[idx_val], labels[idx_val], weight = None)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        # loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    
    if record is not None:
        record[acc_val.item()] = acc_test.item()

    if epoch % 10 == 0:
        print("############################################################################")
        if adj == None:
            print("MLP Epoch {:03d}".format(epoch))
        else:
            print("GCN Epoch {:03d}".format(epoch))
        print("train_loss: {:.4f}, val_loss: {:.4f}".format(loss_train.item(), loss_val.item()))
        print("current val_acc: {:.4f} ".format(acc_val.item()))
        print("current test_acc: {:.4f} ".format( acc_test.item()))
        print("############################################################################")

    return model, loss_val

def evaluate(hidden_size, features, labels, idx_train, idx_val, idx_test, adj):
    model2 = GCN(nfeat=features.shape[1],
            nhid=hidden_size,
            nclass=labels.max().item() + 1,
            dropout=0.6)
    model2.cuda()
    optimizer = optim.Adam(model2.parameters(),
                        lr=0.2, weight_decay=5e-4)
    num_epochs = 400
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    t_total = time.time()
    record = {}
    test_acc = 0
    early_stopping = EarlyStopper(patience=100, min_delta = 0.005)
    for epoch in range(num_epochs):
        # print("Epoch {}".format(epoch))
        # model_best, test_acc, loss_val = train(epoch+1, model2,optimizer,features, labels, idx_train, idx_val, idx_test, best_val_acc, record, adj)
        cur_model, loss_val = train(epoch+1, model2,optimizer,features,labels,idx_train, idx_val, idx_test, record, adj)
        
        if early_stopping.early_stop(cur_model, loss_val, PATH = GCNPATH):
            print("Early stopped at epoch: ", epoch)
            # load best model
            model2.load_state_dict(torch.load(GCNPATH))
            break
        scheduler.step()

    
    with torch.no_grad():
        output = model2(features, adj)
        test_acc = accuracy(output[idx_test], labels[idx_test])
    
    print("Optimization Finished!")
    print("Test accuracy of the model with best loss: {:.4f}".format(test_acc))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    bit_list = sorted(record.keys())
    bit_list.reverse()

    print("Best 10 checkpoints: ")
    for key in bit_list[:10]:
        value = record[key]
        print(key,value)

    y_predict = np.argmax(F.softmax(model2(features, adj)[idx_test],1).cpu().detach().numpy(), 1)
    labels_npy = labels[idx_test].cpu().numpy()
    
    cm=ConfusionMatrixDisplay.from_predictions(labels_npy, y_predict)
    fig, ax = plt.subplots(figsize=(30,30))
    cm.plot(ax=ax)
    cm.figure_.savefig('logs/{0}/{0}_confusion_matrix.png'.format(name))
    return record[bit_list[0]]

if __name__ == '__main__':
  main() 