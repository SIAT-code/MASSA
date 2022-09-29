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
from compound_process import *

class MyDataset(Dataset):
    def __init__(self, df):
        self.examples = df

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        df = self.examples.iloc[index]
        protein_tokens_pad_1 = torch.tensor(df['pro_tokens_pad_1'], dtype=torch.long)
        go_tokens_pad_1 = torch.tensor(df['go_tokens_pad_1'], dtype=torch.long)
        pro_ids_1 = df['pro_ids_1']
        goterms_1 = torch.tensor(df['goterms_1'], dtype=torch.float32)
        proteins_num_1 = torch.tensor(df['proteins_num_1'], dtype=torch.long)
        goterms_num_1 = torch.tensor(df['goterms_num_1'], dtype=torch.long)
        labels = torch.tensor(df['labels'], dtype=torch.float32)
        protein_tokens_pad_2 = torch.tensor(df['pro_tokens_pad_2'], dtype=torch.long)
        go_tokens_pad_2 = torch.tensor(df['go_tokens_pad_2'], dtype=torch.long)
        pro_ids_2 = df['pro_ids_2']
        goterms_2 = torch.tensor(df['goterms_2'], dtype=torch.float32)
        proteins_num_2 = torch.tensor(df['proteins_num_2'], dtype=torch.long)
        goterms_num_2 = torch.tensor(df['goterms_num_2'], dtype=torch.long)
        data_list = [protein_tokens_pad_1, go_tokens_pad_1, pro_ids_1, goterms_1, proteins_num_1, goterms_num_1, labels, protein_tokens_pad_2, go_tokens_pad_2, pro_ids_2, goterms_2, proteins_num_2, goterms_num_2]
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

# k折交叉验证所需, 仅SKEMPI任务需要k折交叉验证, 仅在SKEMPI任务里才有这个函数
def split_kfolds_dataset(dataset, k):
        n = int(1.0 / k * len(dataset))
        dataset_kfolds = []
        for i in range(k-1):
            dataset_i = dataset[i * n: (i+1) * n]
            dataset_kfolds.append(dataset_i)
        dataset_kfolds.append(dataset[(k-1) * n:])

        return dataset_kfolds

def pack(pro_tokens_1, go_tokens_1, pro_ids_1, goterms_1, pro_tokens_2, go_tokens_2, pro_ids_2, goterms_2, labels, dataset_path):
    pro2id_path = os.path.join(dataset_path, "pro2id.json")
    with open(pro2id_path) as f:
        pro2id_dict = json.load(f)
    go2id_path = os.path.join(dataset_path, "go2id.json")
    with open(go2id_path) as f:
        go2id_dict = json.load(f)

    N = len(pro_tokens_1)

    proteins_len_1 = 0
    goterms_len_1 = 0
    
    proteins_num_1 = []
    for pro_token in pro_tokens_1:
        proteins_num_1.append(len(pro_token))
        if len(pro_token) >= proteins_len_1:
            proteins_len_1 = len(pro_token)
    proteins_num_1 = torch.tensor(proteins_num_1, dtype=torch.long)

    goterms_num_1 = []
    for goterm in goterms_1:
        goterms_num_1.append(goterm.shape[0])
        if goterm.shape[0] >= goterms_len_1:
            goterms_len_1 = goterm.shape[0]
    goterms_num_1 = torch.tensor(goterms_num_1, dtype=torch.long)

    pro_tokens_pad_1 = []
    for pro_token in pro_tokens_1:
        pro_tokens_pad_1.append([pro2id_dict[t] if t in pro2id_dict else -100 for t in pro_token] + [-100]*(proteins_len_1-len(pro_token)))
    pro_tokens_pad_1 = torch.tensor(pro_tokens_pad_1, dtype=torch.long)

    go_tokens_pad_1 = []
    for go_token in go_tokens_1:
        go_token_list = go_token.strip().split(';')
        go_tokens_pad_1.append([go2id_dict[t] if t in go2id_dict else -100 for t in go_token_list] + [-100]*(goterms_len_1-len(go_token_list)))
    go_tokens_pad_1 = torch.tensor(go_tokens_pad_1, dtype=torch.long)

    goterms_new_1 = torch.zeros((N, goterms_len_1, goterms_1[0].shape[1]))
    for i, goterm in enumerate(goterms_1):
        a_len = goterm.shape[0]
        goterms_new_1[i, :a_len, :] = goterm

    proteins_len_2 = 0
    goterms_len_2 = 0
    
    proteins_num_2 = []
    for pro_token in pro_tokens_2:
        proteins_num_2.append(len(pro_token))
        if len(pro_token) >= proteins_len_2:
            proteins_len_2 = len(pro_token)
    proteins_num_2 = torch.tensor(proteins_num_2, dtype=torch.long)

    goterms_num_2 = []
    for goterm in goterms_2:
        goterms_num_2.append(goterm.shape[0])
        if goterm.shape[0] >= goterms_len_2:
            goterms_len_2 = goterm.shape[0]
    goterms_num_2 = torch.tensor(goterms_num_2, dtype=torch.long)

    pro_tokens_pad_2 = []
    for pro_token in pro_tokens_2:
        pro_tokens_pad_2.append([pro2id_dict[t] if t in pro2id_dict else -100 for t in pro_token] + [-100]*(proteins_len_2-len(pro_token)))
    pro_tokens_pad_2 = torch.tensor(pro_tokens_pad_2, dtype=torch.long)

    go_tokens_pad_2 = []
    for go_token in go_tokens_2:
        go_token_list = go_token.strip().split(';')
        go_tokens_pad_2.append([go2id_dict[t] if t in go2id_dict else -100 for t in go_token_list] + [-100]*(goterms_len_2-len(go_token_list)))
    go_tokens_pad_2 = torch.tensor(go_tokens_pad_2, dtype=torch.long)


    goterms_new_2 = torch.zeros((N, goterms_len_2, goterms_2[0].shape[1]))
    for i, goterm in enumerate(goterms_2):
        a_len = goterm.shape[0]
        goterms_new_2[i, :a_len, :] = goterm

    labels = torch.tensor(labels, dtype=torch.float32)

    # proteins_new: [batch, max_proteins_len, w2v_emb=768], torch.float32
    # goterms_new: [batch, max_proteins_len, w2v_emb=64], torch.float32
    # protein_num: [batch], represents the actual length of each protein, torch.int64
    # goterms_num: [batch], represents the actual length of each goterm, torch.int64
    data_pack_dict = {}
    data_pack_dict["pro_tokens_pad_1"] = list(pro_tokens_pad_1.numpy())
    data_pack_dict["go_tokens_pad_1"] = list(go_tokens_pad_1.numpy())
    data_pack_dict["pro_ids_1"] = pro_ids_1
    data_pack_dict["goterms_1"] = list(goterms_new_1.numpy())
    data_pack_dict["proteins_num_1"] = list(proteins_num_1.numpy())
    data_pack_dict["goterms_num_1"] = list(goterms_num_1.numpy())

    data_pack_dict["pro_tokens_pad_2"] = list(pro_tokens_pad_2.numpy())
    data_pack_dict["go_tokens_pad_2"] = list(go_tokens_pad_2.numpy())
    data_pack_dict["pro_ids_2"] = pro_ids_2
    data_pack_dict["goterms_2"] = list(goterms_new_2.numpy())
    data_pack_dict["proteins_num_2"] = list(proteins_num_2.numpy())
    data_pack_dict["goterms_num_2"] = list(goterms_num_2.numpy())

    data_pack_dict["labels"] = labels
    data_pack_df = pd.DataFrame(data_pack_dict)
    return data_pack_df

def transform_data(dataset, max_pro_seq_len, dataset_path):
    seq_tokens_1, go_tokens_1, pro_sequences_1, goterms_1, pro_structures_1, seq_tokens_2, go_tokens_2, pro_sequences_2, goterms_2, pro_structures_2, labels = [], [], [], [], [], [], [], [], [], [], []
    for data in dataset:
        seq_token_1, go_token_1, pro_sequence_1, goterm_1, pro_structure_1, seq_token_2, go_token_2, pro_sequence_2, goterm_2, pro_structure_2, label = data

        seq_tokens_1.append(seq_token_1), seq_tokens_2.append(seq_token_2)
        go_tokens_1.append(go_token_1), go_tokens_2.append(go_token_2)
        pro_sequences_1.append(pro_sequence_1), pro_sequences_2.append(pro_sequence_2)
        goterms_1.append(goterm_1), goterms_2.append(goterm_2)
        pro_structures_1.append(pro_structure_1), pro_structures_2.append(pro_structure_2)
        labels.append(label)
    
    data_pack_df = pack(seq_tokens_1, go_tokens_1, pro_sequences_1, goterms_1, seq_tokens_2, go_tokens_2, pro_sequences_2, goterms_2, labels, dataset_path)
    dataset_pack = MyDataset(data_pack_df)
    dataset_structure_1 = ProteinGraphDataset(pro_structures_1, max_seq_len=max_pro_seq_len)
    dataset_structure_2 = ProteinGraphDataset(pro_structures_2, max_seq_len=max_pro_seq_len)
    return dataset_pack, dataset_structure_1, dataset_structure_2

def data_read(tokens_input, dataset_type, max_pro_seq_len, seed):
    file_path = os.path.join(tokens_input, "samples_seq_go_%s.txt" % dataset_type)
    df_tokens = pd.read_csv(file_path, sep=',')
    df_tokens.columns = ['id1', 'uid1', 'seq1', 'go1', 'id2', 'uid2', 'seq2', 'go2', 'label']

    seq_tokens_1, go_tokens_1, pro_ids_1, seq_tokens_2, go_tokens_2, pro_ids_2, labels = df_tokens['seq1'].fillna('').values.tolist(), df_tokens['go1'].fillna('').values.tolist(), df_tokens['id1'].fillna('').values.tolist(), df_tokens['seq2'].fillna('').values.tolist(), df_tokens['go2'].fillna('').values.tolist(), df_tokens['id2'].fillna('').values.tolist(), df_tokens['label'].fillna('').values.tolist()

    seq_tokens_tmp = []
    for index, seq_token in enumerate(seq_tokens_1):
        seq_tokens_tmp.append(seq_token[: max_pro_seq_len])
    seq_tokens_1 = seq_tokens_tmp

    seq_tokens_tmp = []
    for index, seq_token in enumerate(seq_tokens_2):
        seq_tokens_tmp.append(seq_token[: max_pro_seq_len])
    seq_tokens_2 = seq_tokens_tmp

    """Load preprocessed data."""
    goterms_1 = load_tensor(os.path.join(tokens_input, 'goterms_%s_1' % dataset_type), torch.FloatTensor)
    goterms_2 = load_tensor(os.path.join(tokens_input, 'goterms_%s_2' % dataset_type), torch.FloatTensor)

    cath_1 = CATHDataset(os.path.join(tokens_input, 'structure_%s_1.jsonl' % dataset_type))
    cath_2 = CATHDataset(os.path.join(tokens_input, 'structure_%s_2.jsonl' % dataset_type))
    print("Check whether the length of sequence and structure is the same.")
    unequal_count, problem_samples_count = 0, 0
    for i in reversed(range(len(seq_tokens_1))):
        if (len(seq_tokens_1[i]) != len(cath_1.data[i]['seq']) and (len(seq_tokens_1[i]) < max_pro_seq_len or len(cath_1.data[i]['seq']) < max_pro_seq_len)) or (len(seq_tokens_2[i]) != len(cath_2.data[i]['seq']) and (len(seq_tokens_2[i]) < max_pro_seq_len or len(cath_2.data[i]['seq']) < max_pro_seq_len)):
            unequal_count += 1
            del seq_tokens_1[i]
            del go_tokens_1[i]
            del pro_ids_1[i]
            del goterms_1[i]
            del cath_1.data[i]
            del labels[i]
            del seq_tokens_2[i]
            del go_tokens_2[i]
            del pro_ids_2[i]
            del goterms_2[i]
            del cath_2.data[i]
            del labels[i]

    print("Unequal protein length count: %d, delete %d samples!!!" % (unequal_count, unequal_count))
    print("Problem samples count: %d, delete %d samples!!!" % (problem_samples_count, problem_samples_count))
    dataset = list(zip(seq_tokens_1, go_tokens_1, pro_ids_1, goterms_1, cath_1.data, seq_tokens_2, go_tokens_2, pro_ids_2, goterms_2, cath_2.data, labels))

    dataset = shuffle_dataset(dataset, seed)

    return dataset

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
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20, '#': 21, 'U': 22}
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

