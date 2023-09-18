import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
from scipy.sparse import csgraph
import sys
import time
import argparse
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision

from torch.optim import lr_scheduler
import copy
from utils import *
from cluster import *
import sklearn

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F

from data_util import load_data
from models import GCN, MLP

# TODO :Implement the loss function for imbalanced data
# TODO : Decide the training size
# TODO : Decide the hyper-parameters for GCN training

"""
drop_out rate: 0.85 or 0.6
Training epochs： 400 
Learning rate: 0.2 or 0.05 
L2 regularization: 5e-4 
"""
# TODO : Output the confusion matrix
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
GCNPATH = 'logs/allnohosts/b_allnohosts_GCN_checkpoint.pt'
# use all training data

def argument_parser():
    # Data processing
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=128, help='batch size')
    parser.add_argument('--dataset', type=str, default='citeseer', choices = ['hostg.phylum', 'hostg.genus','hostg.class','hostg.family','hostg.order','citeseer'])
    parser.add_argument('--num_classes', type=int, default = 6, help='number of classes')
    parser.add_argument('--num_nodes', type=int, default = 3327, help='number of nodes')
    # parser set seed
    parser.add_argument('--seeds', type=int, default=4, help='random seed')
    return parser

def main():  
    """
    ------------------------------------------------------------------------
    --------------------- Model Training -----------------------------------
    ------------------------------------------------------------------------ 
    """
    parser = argument_parser()
    args = parser.parse_args()
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
    ititial_labeled_nodes = num_class*initial_num_per_class
    num_for_label = num_class*20 # maximum labeled nodes for training 
    print("Running ALG model")

    print("T:",T)
    print("Number of classes:",num_class)
    print("Total number of nodes:",num_node)
    print("The number of nodes for label:",num_for_label)

    
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

    idx_train = []
    for i in range(num_hosts, num_node):
        if i not in idx_test and i not in idx_val:
            idx_train.append(i)
    idx_train = torch.LongTensor(idx_train)

    labels_npy = labels.cpu().numpy()

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


    features = torch.FloatTensor(features).cuda()
    features_GCN = torch.FloatTensor(features_GCN).cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    labels_npy = labels.cpu().numpy()
    features_npy = features.cpu().numpy()
    print("Size of training set: ", len(idx_train))
    print("Size of validation set: ", len(idx_val))
    print("Size of test set: ", len(idx_test))

    """
    ------------------------------------------------------------------------
    --------------------- Model Training -----------------------------------
    ------------------------------------------------------------------------ 
    """

    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  Evaluation begin  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    test_acc = evaluate(args.hidden_size, features_GCN, labels, idx_train, idx_val, idx_test, adj)
    # print("Test accuracy: \n{}".format(test_acc.cpu().numpy()))
    print("Test accuracy of model with best validation metric: \n{}".format(test_acc))

    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  Evaluation end  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # Model Evaluation 
    # pd.DataFrame({"num": [len(idx_train)], "acc": [test_acc.cpu().numpy()]}).to_csv("logs/allnohosts/data.csv")
    pd.DataFrame({"num": [len(idx_train)], "acc": [test_acc]}).to_csv("logs/allnohosts/data.csv")


def train(epoch, model, optimizer, features, labels, idx_train, idx_val, idx_test, record, adj = None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if adj is None:
        output = model(features)
    else:
        output = model(features, adj)

    class_weight = cal_class_weight(idx_train, labels)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train], weight= None)
    # focal_loss = FocalLoss(gamma = 0.5)
    # loss_train = focal_loss(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        if adj is None:
            output = model(features)
        else:
            output = model(features, adj)
        val_class_weight = cal_class_weight(idx_val, labels)

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val], weight= None)
        # loss_val = focal_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        test_class_weight = cal_class_weight(idx_test, labels)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test], weight= None)
        # loss_test = focal_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])


    if record is not None:
        record[acc_val.item()] = acc_test.item()
    if epoch % 10 == 0:
        print("############################################################################")
        if adj == None:
            print("MLP Epoch {:03d}".format(epoch))
        else:
            print("GCN Epoch {:03d}".format(epoch))
        
        print("train_loss: {:.4f}, val_loss: {:.4f}, val_test: {:.4f}".format(loss_train.item(), loss_val.item(), loss_test.item()))
        print("current val_acc: {:.4f} ".format(acc_val.item()))
        print("current test_acc: {:.4f} ".format( acc_test.item()))
        print("############################################################################")
    return model, loss_val

def evaluate(hidden_size, features, labels, idx_train, idx_val, idx_test, adj):
    best_val_acc = 0
    model2 = GCN(nfeat=features.shape[1],
            nhid=hidden_size,
            nclass=labels.max().item() + 1,
            dropout=0.6)
    model2.cuda()
    optimizer = optim.Adam(model2.parameters(),
                        lr=0.2, weight_decay=5e-4)
    t_total = time.time()
    record = {}
    test_acc = 0
    num_epochs = 400
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopper(patience=100, min_delta = 0.005)
    for epoch in range(num_epochs):
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
    print("Test accuracy {:.4f}".format(test_acc))
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    bit_list = sorted(record.keys())
    bit_list.reverse()
    print("Best 10 checkpoints")
    for key in bit_list[:10]:
        value = record[key]
        print(key,value)
    
    y_predict = np.argmax(F.softmax(model2(features, adj)[idx_test],1).cpu().detach().numpy(), 1)
    print("Confusion matrix for test set")

    labels_npy = labels[idx_test].cpu().numpy()
    
    cm=ConfusionMatrixDisplay.from_predictions(labels_npy, y_predict)
    cm.figure_.savefig('logs/allnohosts/b_allnohosts_confusion_matrix.png')
    return record[bit_list[0]]

if __name__ == '__main__':
  main() 