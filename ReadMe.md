# Short ReadMe.py for reproduction of Table 1
The whole environment is in requirements.txt. It's highly recommended to mainly check the important packages: pytorch, dgl, pytorch_geometric, ogb.
## Cora,Citeseer and Pubmed.
```python run_small_graph.py --alp 1 --dataset cora --K 3 --prop_step 16 --hidden 32 --metric hits@k --gpu 0 --dynamic dynamic --lam0 0.01 --lr 0.01```

```python run_small_graph.py --alp 1 --dataset citeseer --K 3 --prop_step 4 --hidden 32 --metric hits@k --gpu 0 --dynamic dynamic --lam0 0.01 --lr 0.01```

```python run_small_graph.py --alp 1 --dataset pubmed --K 3  --prop_step 4 --hidden 32 --metric hits@k --gpu 0 --dynamic dynamic --lam0 0.01 --lr 0.01```
## OGB datasets.
```python run_large_graph.py --dataset ogbl-ppa --gpu 0 --K 1 --lr 0.005 --hidden 128 --batch_size 65536 --model yinyang_l --dropout 0.2 --num_neg 6 --epochs 300 --prop_step 2 --lam0 1 --lam 1 --seed 3 --alp 1  --sign_k 2 --split_negs``` 

```python run_large_graph.py --dataset ogbl-citation2 --K 1 --lr 0.003 --hidden 128 --batch_size 131072 --model yinyang_l --dropout 0 --num_neg 6 --epochs 200 --prop_step 3 --lam0 0.1 --seed 0 --alp 1  --lam_K 0.1 --step_lr_decay --loss_negtype source --gpu 0 --split_negs ```

```python run_large_graph.py --dataset ogbl-collab --K 3 --lr 0.0004 --hidden 512 --batch_size 16384 --model yinyang --dropout 0.2 --num_neg 3 --epochs 800 --prop_step 6 --lam0 0.01 --lam 1 --alp 1  --filter_year 2010 --contain_valid_edge --train_valid_edge --gpu 0 --seed 3``` 

```python run_large_graph.py --dataset ogbl-ddi --K 1 --lr 0.001 --hidden 1024 --batch_size 8192 --model yinyang --dropout 0.6 --num_neg 6 --epochs 300 --prop_step 2 --lam0 0.01 --step_lr_decay --as_decoder_input --gpu 0 --seed 9 ``` 
