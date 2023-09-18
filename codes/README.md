# Active Learning Framework for Graph Convolutional Networks and their application to virus classification




## Requirements


To install requirements:

```setup
pip install -r requirements.txt
```
## Datasets
To train using the following models, first put your dataset under the directory of "../data"

## Training

To train the model(s) in the paper:
### Training ALG models 

Run the following command to train ALG GCN/MLP models 

```
python3 ALG.py --dataset=[hostg.phylum, hostg.genus, hostg.class, hostg.family, hostg.order] --seeds $((seeds))
python3 ALG_MLP.py --dataset=[hostg.phylum, hostg.genus, hostg.class, hostg.family, hostg.order] --seeds $((seeds))

```


### Training baseline active learning methods

We implemented two baseline active learning strategies: random and selection by entropy:
```
python3 baseline_unified.py --dataset=[hostg.phylum, hostg.genus, hostg.class, hostg.family, hostg.order] --method=[random, entropy] --seeds $((seeds))

```

### Training using all data
Instead of picking nodes using active learning, we also provided models traing using all data:
```
python3 baseline_train_all.py --dataset=[hostg.phylum, hostg.genus, hostg.class, hostg.family, hostg.order] --seeds $((seeds))
```

Parameters:
- dataset: dataset used in training 
- hidden_size: batch size
- no_hosts: whether to inlcude host nodes in the initially labeled datasets
  
  