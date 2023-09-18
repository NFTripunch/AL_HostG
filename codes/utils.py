import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import sys
import time
import argparse
import pandas as pd
import torch
import random
import copy
import sklearn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import balanced_accuracy_score

# Others 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def average_ab(num_selected,all_left):
    # ensure that number of nodes to be labeled doesn't exceed the number of available nodes

    # does all left include the one selected?

    target_sum=sum(num_selected)
    iter = 0
    while(True):
        iter += 1
        # print("round {}".format(iter))
        # print("num selected: ", num_selected)
        # print("all left: ", all_left)
        for i in range(len(num_selected)):
            num_selected[i]=min(num_selected[i],all_left[i])
        sum_num=sum(num_selected)
        if(sum_num==target_sum):
            # print("sum_num==target_sum")
            break # != indicating that there exists j \in [0, len(num_selected) - 1], s.t. all_left[j] < num_selected[j]
        cnt=0
        for i in range(len(num_selected)):
            if(num_selected[i]<all_left[i]):
                # print("item ",i," num train < num available")
                cnt+=1
        a_tmp=(target_sum-sum_num)//cnt
        b_tmp=(target_sum-sum_num)%cnt
        # print("a_tmp: ", a_tmp)
        # print("b_tmp: ", b_tmp)
        cnt_tmp=0
        for i in range(len(num_selected)):
            if(num_selected[i]<all_left[i]):
                if(cnt_tmp<b_tmp):
                    num_selected[i]+=a_tmp+1
                    cnt_tmp+=1
                else:
                    num_selected[i]+=a_tmp


# Math Tools 
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

# Evaluation 
def accuracy(output, labels, balanced = False):
    # if balanced:
    #     preds = output.max(1)[1].type_as(labels)
    #     if torch.is_tensor(labels):
    #         labels = labels.cpu().detach().numpy()
    #     if torch.is_tensor(preds):
    #         preds = preds.cpu().detach().numpy()
    #     return balanced_accuracy_score(labels, preds)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def cal_class_weight(idx, labels):
    pad_y = labels[idx].cpu().numpy()
    classes = len(set(labels.cpu().numpy()))
    for i in range(classes):
        # some class may not be in the training/val/test set
        if i not in pad_y:
            pad_y = np.append(pad_y, i)
    class_weight = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced',classes = torch.unique(labels).cpu().numpy(), y = pad_y)
    class_weight = torch.tensor(class_weight, dtype = torch.float).cuda()
    class_weight = torch.sqrt(class_weight)
    return class_weight
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose = True):
        self.patience = patience # number of epochs to wait before stopping
        self.min_delta = min_delta # minimum change in the monitored quantity to qualify as an improvement
        self.counter = 0
        self.verbose = verbose
        self.min_validation_loss = np.inf

    def save_checkpoint(self, model, PATH = 'logs/checkpoint.pt'):
        torch.save(copy.deepcopy(model.state_dict()), PATH)

    def early_stop(self, current_model, validation_loss, PATH = 'logs/checkpoint.pt'):
        if validation_loss < self.min_validation_loss - self.min_delta:
            if self.verbose:
                print("Validation loss decreased ({:.3f} --> {:.3f}).  Saving model ...".format(self.min_validation_loss, validation_loss))
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(current_model, PATH)
        else: 
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Node Selection Measures
def get_normalized_similarity(dist):
    # scale to [0,1]
    if len(dist) == 0:
        return {}
    elif len(dist) == 1:
        return {list(dist.keys())[0]:1}
    else:
        index_value = dist
        simi_values = []
        for index,value in index_value.items():
            simi_values.append(1-value)
        
        denom = max(simi_values) - min(simi_values)
        if denom != 0:
            normalized_values = (simi_values - min(simi_values))/(max(simi_values)-min(simi_values))
        else:
            normalized_values = np.ones(len(simi_values))
        
        count = 0
        for index,value in index_value.items():
            index_value[index] = normalized_values[count]
            count+=1
        return index_value

def get_normalized_difference(dist):
    # scale to [0,1]
    if len(dist) == 0:
        return {}
    elif len(dist) == 1:
        return {list(dist.keys())[0]:1}
    else:
        index_value = dist
        diff_values = []
        for index,value in index_value.items():
            diff_values.append(value)
        if max(diff_values) - min(diff_values) == 0:
            normalized_values = np.ones(len(diff_values))
        else:
            normalized_values = (diff_values - min(diff_values))/(max(diff_values)-min(diff_values))
        count = 0
        for index,value in index_value.items():
            index_value[index] = normalized_values[count]
            # count+=1
        return index_value

# Graph operation 

def get_adaptive_training_epochs(total_training_epochs, num_iterations, m_T, m_1):
    T = num_iterations
    m = np.zeros(T)
    for t in range(T):
        m_t = (m_T-m_1)*(t)/(T-1) +1
        m[t] = m_t
    m = m/np.sum(m) # similar to softmax, make sure the sum of m is 1

    return (m * total_training_epochs).astype(int)



def get_weighted_dist(num_node, output1,output2,alpha1,alpha2):
    # eq. 11

    dist12 = output1 - output2
    weighted_dist = []
    for node in range(num_node):
        weighted_dist.append(np.sqrt(sum(np.power(dist12[node], 2))))
    weighted_dist = np.array(weighted_dist)
    weighted_dist = weighted_dist*alpha1*alpha2
    return weighted_dist

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def graph_decompose(adj,graph_name,k,metis_p,strategy="edge"):
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","number" (depending on metis preprocessing) 
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed subgraphs
    '''
    print("Skeleton:",metis_p)
    print("Strategy:",strategy)
    g,g_rest,edges_rest,gs=get_graph_skeleton(adj,graph_name,k,metis_p)
    gs=allocate_edges(g_rest,edges_rest, gs, strategy)
       
    re=[]       
   
    #print the info of nodes and edges of subgraphs 
    edge_num_avg=0
    compo_num_avg=0
    print("Subgraph information:")
    for i in range(k):
        nodes_num=gs[i].number_of_nodes()
        edge_num=gs[i].number_of_edges()
        compo_num=nx.number_connected_components(gs[i])
        print("\t",nodes_num,edge_num,compo_num)
        edge_num_avg+=edge_num
        compo_num_avg+=compo_num
        re.append(nx.to_scipy_sparse_matrix(gs[i])) 
        
    #check the shared edge number in all subgrqphs
    edge_share=set(sort_edge(gs[0].edges()))
    for i in range(k):        
        edge_share&=set(sort_edge(gs[i].edges()))
        
    print("\tShared edge number is: %d"%len(edge_share))
    print("\tAverage edge number:",edge_num_avg/k) 
    print("\tAverage connected component number:",compo_num_avg/k)
    print("\n"+"-"*70+"\n")
    return re

def sort_edge(edges):
    edges=list(edges)
    for i in range(len(edges)):
        u=edges[i][0]
        v=edges[i][1]
        if u > v:
            edges[i]=(v,u)
    return edges

def get_graph_skeleton(adj,graph_name,k,metis_p): 
    '''
    Input:
        adj:the adjacency matrix of original graph
        graph_name:"cora","citeseer","pubmed"
        k:decompose into k subgraphs
        metis_p:"no_skeleton","all_skeleton","k" 
    Output:
        g:the original graph
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
    '''
    g=nx.from_numpy_matrix(adj.todense())
    num_nodes=g.number_of_nodes()
    print("Original nodes number:",num_nodes)
    num_edges=g.number_of_edges()
    print("Original edges number:",num_edges)  
    print("Original connected components number:",nx.number_connected_components(g),"\n")    
    
    g_dic=dict()
    
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()] 
            
    #initialize all the subgrapgs, add the nodes
    gs=[nx.Graph() for i in range(k)]
    for i in range(k):
        gs[i].add_nodes_from([i for i in range(num_nodes)])
    
    if metis_p=="no_skeleton":
        #no skeleton
        g_rest=g
        edges_rest=list(g_rest.edges())
    else:    
        if metis_p=="all_skeleton":
            #doesn't use metis to cut any edge
            graph_cut=g
        else:
            #read the cluster info from file
            f=open("metis_file/"+graph_name+".graph.part.%s"%metis_p,'r')
            cluster=dict()  
            i=0
            for lines in f:
                cluster[i]=eval(lines.strip("\n"))
                i+=1
           
            #get the graph cut by Metis    
            graph_cut=nx.Graph()
            graph_cut.add_nodes_from([i for i in range(num_nodes)])  
            
            for v in range(num_nodes):
                v_class=cluster[v]
                for u in g_dic[v]:
                    if cluster[u]==v_class:
                        graph_cut.add_edge(v,u)
            
        subgs=list(nx.connected_component_subgraphs(graph_cut))
        print("After Metis,connected component number:",len(subgs))
        
                
        #add the edges of spanning tree, get the skeleton
        for i in range(k):
            for subg in subgs:
                T=get_spanning_tree(subg)
                gs[i].add_edges_from(T)
        
        #get the rest graph including all the edges except the shared egdes of spanning trees
        edge_set_share=set(sort_edge(gs[0].edges()))
        for i in range(k):
            edge_set_share&=set(sort_edge(gs[i].edges()))
        edge_set_total=set(sort_edge(g.edges()))
        edge_set_rest=edge_set_total-edge_set_share   
        edges_rest=list(edge_set_rest)
        g_rest=nx.Graph()
        g_rest.add_nodes_from([i for i in range(num_nodes)])
        g_rest.add_edges_from(edges_rest)
       
          
    #print the info of nodes and edges of subgraphs
    print("Skeleton information:")
    for i in range(k):
        print("\t",gs[i].number_of_nodes(),gs[i].number_of_edges(),nx.number_connected_components(gs[i])) 
        
    edge_set_share=set(sort_edge(gs[0].edges()))
    for i in range(k):
        edge_set_share&=set(sort_edge(gs[i].edges()))
    print("\tShared edge number is: %d\n"%len(edge_set_share))
    
    return g,g_rest,edges_rest,gs

def get_spanning_tree(g):
    '''
    Input:Graph
    Output:list of the edges in spanning tree
    '''
    g_dic=dict()
    for v,nb in g.adjacency():
        g_dic[v]=[u[0] for u in nb.items()]
        np.random.shuffle(g_dic[v])
    flag_dic=dict()
    if g.number_of_nodes() ==1:
        return []
    gnodes=np.array(g.nodes)
    np.random.shuffle(gnodes)
    
    for v in gnodes:
        flag_dic[v]=0
    
    current_path=[]
    
    def dfs(u):
        stack=[u]
        current_node=u
        flag_dic[u]=1
        while len(current_path)!=(len(gnodes)-1):
            pop_flag=1
            for v in g_dic[current_node]:
                if flag_dic[v]==0:
                    flag_dic[v]=1
                    current_path.append((current_node,v))  
                    stack.append(v)
                    current_node=v
                    pop_flag=0
                    break
            if pop_flag:
                stack.pop()
                current_node=stack[-1]     
    dfs(gnodes[0])        
    return current_path

def allocate_edges(g_rest,edges_rest, gs, strategy):
    '''
    Input:
        g_rest:the rest graph
        edges_rest:the rest edges
        gs:the skeleton of the graph for every subgraph
        strategy:"edge" (for edge_decomposition),"node" (for node_decomposition)
    Output:
        the decomposed graphs after allocating rest edges
    '''
    k=len(gs)
    if strategy=="edge":  
        print("Allocate the rest edges randomly and averagely.")
        np.random.shuffle(edges_rest)
        t=int(len(edges_rest)/k)
        
        #add edges
        for i in range(k):       
            if i == k-1:
                gs[i].add_edges_from(edges_rest[t*i:])
            else:
                gs[i].add_edges_from(edges_rest[t*i:t*(i+1)])        
        return gs
    
    elif strategy=="node":
        print("Allocate the edges of each nodes randomly and averagely.")
        g_dic=dict()    
        for v,nb in g_rest.adjacency():
            g_dic[v]=[u[0] for u in nb.items()]
            np.random.shuffle(g_dic[v])
        
        def sample_neighbors(nb_ls,k):
            np.random.shuffle(nb_ls)
            ans=[]
            for i in range(k):
                ans.append([])
            if len(nb_ls) == 0:
                return ans
            if len(nb_ls) > k:
                t=int(len(nb_ls)/k)
                for i in range(k):
                    ans[i]+=nb_ls[i*t:(i+1)*t]
                nb_ls=nb_ls[k*t:]
            '''
            if len(nb_ls)>0:
                for i in range(k):
                    ans[i].append(nb_ls[i%len(nb_ls)])
            '''
            
            
            if len(nb_ls)>0:
                for i in range(len(nb_ls)):
                    ans[i].append(nb_ls[i])
            
            np.random.shuffle(ans)
            return ans
        
        #add edges
        for v,nb in g_dic.items():
            ls=np.array(sample_neighbors(nb,k))
            for i in range(k):
                gs[i].add_edges_from([(v,j) for j in ls[i]])
        
        return gs


def cut_zero(num_class, last_node):
    # ensure all class has non-negative labeled node
    cnt_zero=0
    for i in range(last_node.shape[0]):
        if(last_node[i]<0):
            cnt_zero+=-last_node[i]
            last_node[i]=0
    i=0
    while(True):
        #print(last_node, cnt_zero)
        if(last_node[i]>0 and cnt_zero>0):
            last_node[i]-=1
            cnt_zero-=1
        if(cnt_zero==0):
            break
        i=(i+1)%num_class

def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_mat): 
    receptive_vector=((cur_neighbors+adj_mat[selected_node])!=0)+0
    # print(receptive_vector.shape)
    count=weighted_score.dot(receptive_vector)
    return count

def get_entropy(softmax_output):
    return -np.sum(softmax_output*np.log(softmax_output))
    
def get_current_neighbors_dense(cur_nodes, adj_mat):
    neighbors=(adj_mat[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def get_min_receptive_node_dense(idx_used,high_score_nodes,weighted_score, adj_mat, labels): 
    t=time.time()
    max_receptive_field = np.inf
    max_receptive_node = -1
    _weighted_score=np.array(list(weighted_score.values()))
    cur_neighbors=get_current_neighbors_dense(idx_used, adj_mat)

    # print(cur_neighbors.shape)
    # print(type(cur_neighbors)) # <class 'numpy.ndarray'>
    num_dup = 0
    for node in high_score_nodes:
        receptive_field=get_receptive_fields_dense(cur_neighbors,node,_weighted_score,adj_mat)
        len_labels = len(labels)
        neighbors = (adj_mat[node,:len_labels]!=0)+0
        label_node = labels[node].cpu().numpy()
        print("Label of the node: ", label_node)
        index = np.where(neighbors==1)[0]
        # print(labels.shape)
        # print(labels[index])
        class_pd = pd.Series(labels[index].cpu().numpy()).value_counts().sort_index()
        class_num = class_pd[label_node]
        class_total = pd.Series(labels.cpu().numpy()).value_counts().sort_index()[label_node]
        print("Number of neighbors: {} / total nodes {} in class {}".format(class_num, class_total, label_node))
        # print("Number of neighbors of the node to be selected: {}".format(neighbors.sum()))
        # print("Receptive field of new neighbors:", np.dot(_weighted_score,neighbors))
        # print("current receptive_field {} node {}".format(receptive_field, node))
        # print("Receptive field of original nodes: {}".format(np.dot(_weighted_score,cur_neighbors)))
        if receptive_field < max_receptive_field:
            max_receptive_field = receptive_field
            max_receptive_node = node
            num_dup = 0
        elif receptive_field == max_receptive_field:
            num_dup += 1
    print("Number of duplicates: {} / total nodes: {}".format(num_dup, len(high_score_nodes)))
    return max_receptive_node, max_receptive_field

def get_max_receptive_node_dense(idx_used,high_score_nodes,weighted_score, adj_mat
                                #  , labels = None
                                 ): 
    t=time.time()
    max_receptive_field = 0
    max_receptive_node = -1
    _weighted_score=np.array(list(weighted_score.values()))
    cur_neighbors=get_current_neighbors_dense(idx_used, adj_mat)

    # print(cur_neighbors.shape)
    # print(type(cur_neighbors)) # <class 'numpy.ndarray'>
    num_dup = 0
    for node in high_score_nodes:
        receptive_field=get_receptive_fields_dense(cur_neighbors,node,_weighted_score,adj_mat)
        # if labels is not None:
        #     len_labels = len(labels)
        #     neighbors = (adj_mat[node,:len_labels]!=0)+0
        #     # print("Label of the node: ", labels[node].cpu().numpy())
        #     index = np.where(neighbors==1)[0]
        #     # print(labels.shape)
        #     # print(labels[index])
        #     # print(pd.Series(labels[index].cpu().numpy()).value_counts().sort_index())
        # print("Number of neighbors of the node to be selected: {}".format(neighbors.sum()))
        # print("Receptive field of new neighbors:", np.dot(_weighted_score,neighbors))
        # print("current receptive_field {} node {}".format(receptive_field, node))
        # print("Receptive field of original nodes: {}".format(np.dot(_weighted_score,cur_neighbors)))
        if receptive_field > max_receptive_field:
            max_receptive_field = receptive_field
            max_receptive_node = node
            num_dup = 0
        elif receptive_field == max_receptive_field:
            num_dup += 1
    # print("Number of duplicates: {} / total nodes: {}".format(num_dup, len(high_score_nodes)))
    return max_receptive_node, max_receptive_field


# Not used

# def normalize_adj(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
#     r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
#     return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


# def laplacian(mx, norm):
#     """Laplacian-normalize sparse matrix"""
#     assert (all(len(row) == len(mx) for row in mx)), "Input should be a square matrix"

#     return csgraph.laplacian(adj, normed=norm)

# def aug_random_walk(adj):
#    adj = adj + sp.eye(adj.shape[0])
#    adj = sp.coo_matrix(adj)
#    row_sum = np.array(adj.sum(1))
#    d_inv = np.power(row_sum, -1.0).flatten()
#    d_mat = sp.diags(d_inv)
#    return (d_mat.dot(adj)).tocoo()

  
# from https://github.com/clcarwin/focal_loss_pytorch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()