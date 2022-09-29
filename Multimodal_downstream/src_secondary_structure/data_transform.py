import os
import json
from tokenize import Special
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, df):
        self.examples = df

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        df = self.examples.iloc[index]
        protein_tokens_pad = torch.tensor(df['pro_tokens_pad'], dtype=torch.long)
        go_tokens_pad = torch.tensor(df['go_tokens_pad'], dtype=torch.long)
        pro_ids = df['pro_ids']
        goterms = torch.tensor(df['goterms'], dtype=torch.float32)
        proteins_num = torch.tensor(df['proteins_num'], dtype=torch.long)
        goterms_num = torch.tensor(df['goterms_num'], dtype=torch.long)
        labels = torch.tensor(df['labels_pad'], dtype=torch.long)
        data_list = [protein_tokens_pad, go_tokens_pad, pro_ids, goterms, proteins_num, goterms_num, labels]
        return data_list

def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_tensor_for_pro_ids(file_name, dtype):
    with open(file_name) as f:
        seqs_id = f.read().strip().split('\n')
    return seqs_id

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def pack(pro_tokens, go_tokens, pro_ids, goterms, labels, dataset_path):
    pro2id_path = os.path.join(dataset_path, "pro2id.json")
    with open(pro2id_path) as f:
        pro2id_dict = json.load(f)
    go2id_path = os.path.join(dataset_path, "go2id.json")
    with open(go2id_path) as f:
        go2id_dict = json.load(f)

    proteins_len = 0
    goterms_len = 0
    N = len(pro_tokens)
            
    proteins_num = []
    for pro_token in pro_tokens:
        proteins_num.append(len(pro_token))
        if len(pro_token) >= proteins_len:
            proteins_len = len(pro_token)
    proteins_num = torch.tensor(proteins_num, dtype=torch.long)

    goterms_num = []
    for goterm in goterms:
        goterms_num.append(goterm.shape[0])
        if goterm.shape[0] >= goterms_len:
            goterms_len = goterm.shape[0]
    goterms_num = torch.tensor(goterms_num, dtype=torch.long)
 
    # pro_tokens与labels对齐 mark
    pro_tokens_pad, labels_pad = [], []
    for pro_token, label in zip(pro_tokens, labels):
        assert len(pro_token) == len(label)
        pro_tokens_pad.append([pro2id_dict[t] if t in pro2id_dict else -100 for t in pro_token] + [-100]*(proteins_len-len(pro_token)))
        labels_pad.append([int(l) for l in label] + [-100]*(proteins_len-len(label)))
    pro_tokens_pad = torch.tensor(pro_tokens_pad, dtype=torch.long)
    labels_pad = torch.tensor(labels_pad, dtype=torch.long)

    go_tokens_pad = []
    for go_token in go_tokens:
        go_token_list = go_token.strip().split(';')
        go_tokens_pad.append([go2id_dict[t] if t in go2id_dict else -100 for t in go_token_list] + [-100]*(goterms_len-len(go_token_list)))
    go_tokens_pad = torch.tensor(go_tokens_pad, dtype=torch.long)

    goterms_new = torch.zeros((N, goterms_len, goterms[0].shape[1]))
    for i, goterm in enumerate(goterms):
        a_len = goterm.shape[0]
        goterms_new[i, :a_len, :] = goterm


    # proteins_new: [batch, max_proteins_len, w2v_emb=768], torch.float32
    # goterms_new: [batch, max_proteins_len, w2v_emb=64], torch.float32
    # protein_num: [batch], represents the actual length of each protein, torch.int64
    # goterms_num: [batch], represents the actual length of each goterm, torch.int64
    data_pack_dict = {}
    data_pack_dict["pro_tokens_pad"] = list(pro_tokens_pad.numpy())
    data_pack_dict["go_tokens_pad"] = list(go_tokens_pad.numpy())
    data_pack_dict["pro_ids"] = pro_ids
    data_pack_dict["goterms"] = list(goterms_new.numpy())
    data_pack_dict["proteins_num"] = list(proteins_num.numpy())
    data_pack_dict["goterms_num"] = list(goterms_num.numpy())
    data_pack_dict["labels_pad"] = list(labels_pad.numpy())  # mark
    data_pack_df = pd.DataFrame(data_pack_dict)
    return data_pack_df

def transform_data(dataset, max_pro_seq_len, dataset_path):
    seq_tokens, go_tokens, pro_sequences, goterms, pro_structures, labels = [], [], [], [], [], []
    for data in dataset:
        seq_token, go_token, pro_sequence, goterm, pro_structure, label = data
        seq_tokens.append(seq_token)
        go_tokens.append(go_token)
        pro_sequences.append(pro_sequence)
        goterms.append(goterm)
        pro_structures.append(pro_structure)
        labels.append(label)
    data_pack_df = pack(seq_tokens, go_tokens, pro_sequences, goterms, labels, dataset_path)
    dataset_pack = MyDataset(data_pack_df)
    dataset_structure = ProteinGraphDataset(pro_structures, max_seq_len=max_pro_seq_len)
    return dataset_pack, dataset_structure

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])  # D_mu=[1, D_count]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)  # D_expand=[edge_num, 1]

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)  # RBF=[edge_num, D_count]
    return RBF

class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.
    
    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.
    
    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path):
        """
        path:
        line 1: "{"seq": "MKTA...", "coords":{"N": [[14, 39, 26], ...], "CA": [[...], ...], 
                "C": [[...], ...], "O": [[...], ...]}, "num_chains": 8, "name": "12as.A", "CATH":["3.30.930", ...]}"
        line 2: ...
        ...
        """
        self.data = []
        
        with open(path) as f:
            lines = f.readlines()
        
        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            # name = entry['name']
            coords = entry['coords']
            
            entry['coords'] = list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            ))
            
            self.data.append(entry)

class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, max_seq_len=1024, device="cpu"):
        
        super(ProteinGraphDataset, self).__init__()

        """
        data_list:
            [{"seq": "MKTA...", "coords":{"N": [[14, 39, 26], ...], "CA": [[...], ...], 
            "C": [[...], ...], "O": [[...], ...]}, "num_chains": 8, "name": "12as.A", "CATH":["3.30.930", ...]},
            ...]
        """
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.max_seq_len = max_seq_len
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20, '#': 21}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])
    
    def _featurize_as_graph(self, protein):
        name = protein['name']  # 1ri5.A
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], 
                                     device=self.device, dtype=torch.float32)  
            coords = coords[: self.max_seq_len]
            # coords=[seq_len, 4, 3] 
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)
            seq = seq[: self.max_seq_len]
            seq_len = torch.tensor([seq.shape[0]])
            # seq=[seq_len]
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            # mask=[seq_len]
            coords[~mask] = np.inf
            # coords=[seq_len, 4, 3] 
            X_ca = coords[:, 1]
            # X_ca=[seq_len, 3]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            # edge_index=[2, (seq_len-infinite_num)*top_k]
            pos_embeddings = self._positional_embeddings(edge_index)
            # pos_embeddings=[(seq_len-infinite_num)*top_k, num_positional_embeddings=16]
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            # E_vectors=[(seq_len-infinite_num)*top_k, 3]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # rbf=[(seq_len-infinite_num)*top_k, D_count=16]
            dihedrals = self._dihedrals(coords)  # dihedrals=[seq_len, 6]                 
            orientations = self._orientations(X_ca)  # orientations=[seq_len, 2, 3]   
            sidechains = self._sidechains(coords)  # orientations=[seq_len, 3]   
            
            node_s = dihedrals  # node_s=[seq_len, 6]       
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            # node_v=[seq_len, 3, 3]
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            # edge_s=[(seq_len-infinite_num)*top_k, num_positional_embeddings+D_count=32]
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            # edge_v=[(seq_len-infinite_num)*top_k, 1, 3]
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
            # if name == "xxx":
            #     edge_s = torch.ones((515, 32), dtype=torch.float)
            #     edge_v = torch.ones((515, 1, 3), dtype=torch.float)
            #     edge_index = torch.ones((2, 572), dtype=torch.long)
        data = torch_geometric.data.Data(x=X_ca, seq=seq, seq_len=seq_len, name=name,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index, mask=mask)
        return data
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        # X=[seq_len, 4, 3] 
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])  # X=[seq_len*3, 3]
        dX = X[1:] - X[:-1]  # dX=[seq_len*3-1, 3]
        U = _normalize(dX, dim=-1)  # U=[seq_len*3-1, 3]
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

