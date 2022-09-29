from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import math
import numpy as np
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead
from gvp_gnn import StructureEncoder
from AttentiveFP import Fingerprint

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]

        self.scale = self.scale.to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]

        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]

        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, trg_mask=None):
        # trg = [batch_size, seq len, atom_dim]
        # trg_mask = [batch size, seq len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg

class SeqEncoder(nn.Module):
    """protein sequence feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, max_pro_seq_len, dropout=0.3):
        super().__init__()
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.pos_embedding = nn.Embedding(max_pro_seq_len, protein_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        device = protein.device
        pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(device)
        protein = protein + self.pos_embedding(pos)
        # protein = [batch size, protein len, protein_dim]
        conv_input = self.fc(protein)
        # conv_input = [batch size, protein len,hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            self.scale = self.scale.to(device)
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size, protein len, hid dim]
        conved = self.ln(conved)
        return conved

class SeqStrucFusion(nn.Module):
    def __init__(self, n_layers, n_heads, pf_dim, seq_hid_dim, struc_hid_dim, seq_struc_hid_dim, dropout):
        super().__init__()
        self.multimodal_transform = nn.Linear(seq_hid_dim+struc_hid_dim, seq_struc_hid_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(seq_struc_hid_dim, n_heads, pf_dim, dropout)
             for _ in range(n_layers)])
        
        self.ln = nn.LayerNorm(seq_struc_hid_dim)

    def forward(self, seq_feat, struc_feat, protein_mask):
        multimodal_emb = torch.cat((seq_feat, struc_feat), dim=-1)
        multimodal_emb = self.multimodal_transform(multimodal_emb)
        multimodal_emb = F.relu(multimodal_emb)
        
        for layer in self.layers:
            multimodal_emb = layer(multimodal_emb, protein_mask)
        
        multimodal_emb = self.ln(multimodal_emb)
        return multimodal_emb

class GotermEncoder(nn.Module):
    """protein sequence feature extraction."""
    def __init__(self, input_dim, goterm_dim, n_heads, n_layers, pf_dim, dropout=0.3):
        super().__init__()
        self.fc = nn.Linear(input_dim, goterm_dim)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(goterm_dim, n_heads, pf_dim, dropout)
             for _ in range(n_layers)])
        
        self.ln = nn.LayerNorm(goterm_dim)

    def forward(self, goterms, goterms_mask):
        goterms = self.fc(goterms)

        for layer in self.layers:
            goterms = layer(goterms, goterms_mask)
        
        goterms = self.ln(goterms)
        return goterms

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, 1, compound sent len, 1]
        # src_mask = [batch size, 1, 1, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg

class ProGoCrossFusion(nn.Module):
    def __init__(self, pro_hid_dim, go_hid_dim, pro_n_heads, go_n_heads, pro_n_layers, go_n_layers, pro_pf_dim, go_pf_dim, pro_dropout, go_dropout):
        super().__init__()
        self.pro_layers = nn.ModuleList(
            [TransformerDecoderLayer(pro_hid_dim, pro_n_heads, pro_pf_dim, pro_dropout)
             for _ in range(pro_n_layers)])

        self.go_layers = nn.ModuleList(
            [TransformerDecoderLayer(go_hid_dim, go_n_heads, go_pf_dim, go_dropout)
             for _ in range(go_n_layers)])

        self.pro2go = nn.Linear(pro_hid_dim, go_hid_dim)
        self.go2pro = nn.Linear(go_hid_dim, pro_hid_dim)

    def forward(self, pro_feat, go_feat, pro_mask=None, go_mask=None):
        go_feat_for_pro = self.go2pro(go_feat)
        for layer in self.pro_layers:
            pro_feat_new = layer(pro_feat, go_feat_for_pro, pro_mask, go_mask)

        pro_feat_for_go = self.pro2go(pro_feat)
        for layer in self.go_layers:
            go_feat_new = layer(go_feat, pro_feat_for_go, go_mask, pro_mask)

        return pro_feat_new

class ProProCrossFusion(nn.Module):
    def __init__(self, pro_hid_dim, pro_n_heads, pro_n_layers, pro_pf_dim, pro_dropout):
        super().__init__()
        self.pro_layers = nn.ModuleList(
            [TransformerDecoderLayer(pro_hid_dim, pro_n_heads, pro_pf_dim, pro_dropout)
             for _ in range(pro_n_layers)])

    def forward(self, pro_feat_1, pro_feat_2, pro_mask_1=None, pro_mask_2=None):
        for layer in self.pro_layers:
            pro_feat_1_new = layer(pro_feat_1, pro_feat_2, pro_mask_1, pro_mask_2)

        for layer in self.pro_layers:
            pro_feat_2_new = layer(pro_feat_2, pro_feat_1, pro_mask_2, pro_mask_1)

        return pro_feat_1_new, pro_feat_2_new

class RegressionLayer(nn.Module):
    def __init__(self, pro_hid_dim):
        super().__init__()
        self.fc1 = nn.Linear(pro_hid_dim, int(pro_hid_dim/2))
        self.fc2 = nn.Linear(pro_hid_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def attention_module(self, pro_feature_1, pro_feature_2, pro_mask_1, pro_mask_2):
        ## pro_feature(batch * pro_len * hid)
        ## pro_mask(batch * pro_len)

        batch_size = pro_feature_1.shape[0]
        # batch * pro_len_1 * pro_len_2
        pairwise_pred = torch.sigmoid(torch.matmul(pro_feature_1, pro_feature_2.transpose(1, 2)))
        # batch * pro_len_1 * pro_len_2
        pairwise_mask = torch.matmul(pro_mask_1.view(batch_size, -1, 1), pro_mask_2.view(batch_size, 1, -1))
        pairwise_attention = pairwise_pred * pairwise_mask

        return pairwise_attention

    def forward(self, pro_feat_1, pro_feat_2, pro_mask_1, pro_mask_2):
        norm_1 = torch.norm(pro_feat_1, dim=2)
        pro_mask_1 = pro_mask_1.squeeze(-2).squeeze(-2)
        norm_1 = norm_1.masked_fill(pro_mask_1==0, -1e10)

        norm_1 = F.softmax(norm_1, dim=1)
        pro_feat_1 = torch.bmm(pro_feat_1.permute(0, 2, 1), norm_1.unsqueeze(-1))
        pro_feat_1 = pro_feat_1.squeeze(-1)
        pro_feat_1 = F.relu(self.fc1(pro_feat_1))

        norm_2 = torch.norm(pro_feat_2, dim=2)
        pro_mask_2 = pro_mask_2.squeeze(-2).squeeze(-2)
        norm_2 = norm_2.masked_fill(pro_mask_2==0, -1e10)

        norm_2 = F.softmax(norm_2, dim=1)
        pro_feat_2 = torch.bmm(pro_feat_2.permute(0, 2, 1), norm_2.unsqueeze(-1))
        pro_feat_2 = pro_feat_2.squeeze(-1)
        pro_feat_2 = F.relu(self.fc1(pro_feat_2))

        pro_feat = torch.cat((pro_feat_1, pro_feat_2), dim=-1)
        
        label = self.fc2(pro_feat)
        label = self.sigmoid(label)
        label = label.squeeze(1)
        return label

class Predictor(nn.Module):
    def __init__(self, seq_encoder, struc_encoder, seq_struc_fusion, goterm_encoder, pro_go_cross_fusion, pro_pro_cross_fusion, regression_layer, add_structure, add_goterm):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.struc_encoder = struc_encoder
        self.seq_struc_funsion = seq_struc_fusion
        self.goterm_encoder = goterm_encoder
        self.pro_go_cross_fusion = pro_go_cross_fusion
        self.pro_pro_cross_fusion = pro_pro_cross_fusion
        self.regression_layer = regression_layer
        self.add_structure = add_structure
        self.add_goterm = add_goterm

    def make_masks(self, proteins_num, goterms_num, protein_max_len, goterm_max_len):
        N = len(proteins_num)  # batch size
        protein_mask = torch.zeros((N, protein_max_len))
        goterm_mask = torch.zeros((N, goterm_max_len))
        for i in range(N):
            protein_mask[i, :proteins_num[i]] = 1
            goterm_mask[i, :goterms_num[i]] = 1
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        goterm_mask = goterm_mask.unsqueeze(1).unsqueeze(2)
        protein_mask, goterm_mask = protein_mask.to(proteins_num.device), goterm_mask.to(goterms_num.device)
        return protein_mask, goterm_mask

    def struc_data_format_change(self, sample_num, sample_len, struc_emb, pro_seq_lens, device):
        struc_emb_new = None
        seq_len_1, seq_len_2 = 0, 0
        for i in range(sample_num):
            if i == 0:
                seq_len_1, seq_len_2 = 0, pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = modal2_emb_one.unsqueeze(0)
            else:
                seq_len_1, seq_len_2 = seq_len_2, seq_len_2+pro_seq_lens[i]
                seq_len = pro_seq_lens[i]
                modal2_emb_one = struc_emb[seq_len_1: seq_len_2, :]
                pads = torch.zeros((sample_len-seq_len, struc_emb.shape[-1]), dtype=torch.float).to(device)
                modal2_emb_one = torch.cat((modal2_emb_one, pads), dim=0)
                struc_emb_new = torch.cat((struc_emb_new, modal2_emb_one.unsqueeze(0)), dim=0)
        struc_emb = struc_emb_new
        return struc_emb

    def forward(self, seq_feat_1, struc_feat_1, proteins_num_1, goterms_1, goterms_num_1, seq_feat_2, struc_feat_2, proteins_num_2, goterms_2, goterms_num_2):
        device = seq_feat_1.device
        proteins_mask_1, goterms_mask_1 = self.make_masks(proteins_num_1, goterms_num_1, seq_feat_1.shape[1], goterms_1.shape[1])
        proteins_mask_2, goterms_mask_2 = self.make_masks(proteins_num_2, goterms_num_2, seq_feat_2.shape[1], goterms_2.shape[1])

        start_time_1 = time.time()
        seq_emb_1 = self.seq_encoder(seq_feat_1)
        seq_emb_2 = self.seq_encoder(seq_feat_2)
        end_time_1 = time.time()
        # print("time 1: %g" % (end_time_1-start_time_1))

        start_time_2 = time.time()
        struc_emb_1 = self.struc_encoder(*struc_feat_1)
        struc_emb_2 = self.struc_encoder(*struc_feat_2)
        end_time_2 = time.time()
        # print("time 2: %g" % (end_time_2-start_time_2))

        start_time_3 = time.time()
        # 结构数据的格式转化
        B, L, E = seq_emb_1.shape
        struc_emb_1 = self.struc_data_format_change(B, L, struc_emb_1, proteins_num_1, device)
        B, L, E = seq_emb_2.shape
        struc_emb_2 = self.struc_data_format_change(B, L, struc_emb_2, proteins_num_2, device)
        end_time_3 = time.time()
        # print("time 3: %g" % (end_time_3-start_time_3))

        if not self.add_structure:
            struc_emb_1 = torch.tensor(-1*10e-9, dtype=torch.float32).repeat(struc_emb_1.shape).to(device)
            struc_emb_2 = torch.tensor(-1*10e-9, dtype=torch.float32).repeat(struc_emb_2.shape).to(device)
        start_time_4 = time.time()
        seq_struc_emb_1 = self.seq_struc_funsion(seq_emb_1, struc_emb_1, proteins_mask_1)
        seq_struc_emb_2 = self.seq_struc_funsion(seq_emb_2, struc_emb_2, proteins_mask_2)
        end_time_4 = time.time()
        # print("time 4: %g" % (end_time_4-start_time_4))

        if not self.add_goterm:
            tmp_goterms_mask_1 = torch.zeros_like(goterms_mask_1)
            tmp_goterms_mask_2 = torch.zeros_like(goterms_mask_2)
            goterms_mask_1, goterms_mask_2 = tmp_goterms_mask_1.to(device), tmp_goterms_mask_2.to(device)

        start_time_5 = time.time()
        goterms_emb_1 = self.goterm_encoder(goterms_1, goterms_mask_1)
        goterms_emb_2 = self.goterm_encoder(goterms_2, goterms_mask_2)
        end_time_5 = time.time()
        # print("time 5: %g" % (end_time_5-start_time_5))

        start_time_6 = time.time()
        pro_emb_1 = self.pro_go_cross_fusion(seq_struc_emb_1, goterms_emb_1, proteins_mask_1, goterms_mask_1)
        pro_emb_2 = self.pro_go_cross_fusion(seq_struc_emb_2, goterms_emb_2, proteins_mask_2, goterms_mask_2)
        end_time_6 = time.time()
        # print("time 6: %g" % (end_time_6-start_time_6))

        pro_emb_1, pro_emb_2 = self.pro_pro_cross_fusion(pro_emb_1, pro_emb_2, proteins_mask_1, proteins_mask_2)

        label = self.regression_layer(pro_emb_1, pro_emb_2, proteins_mask_1, proteins_mask_2)

        return label

    def __call__(self, data):
        seq_feat_1, struc_feat_1, proteins_num_1, goterms_1, goterms_num_1, seq_feat_2, struc_feat_2, proteins_num_2, goterms_2, goterms_num_2, gpu_split = data
        device = seq_feat_1.device

        struc_feat_1 = [struc_feat_1[i] for i in gpu_split]
        struc_feat_1 = torch_geometric.data.Batch.from_data_list(struc_feat_1)
        h_V, h_E, edge_index, seq = (struc_feat_1.node_s.to(device), struc_feat_1.node_v.to(device)), (struc_feat_1.edge_s.to(device), struc_feat_1.edge_v.to(device)), struc_feat_1.edge_index.to(device), struc_feat_1.seq.to(device)
        struc_feat_1 = (h_V, h_E, edge_index, seq)

        struc_feat_2 = [struc_feat_2[i] for i in gpu_split]
        struc_feat_2 = torch_geometric.data.Batch.from_data_list(struc_feat_2)
        h_V, h_E, edge_index, seq = (struc_feat_2.node_s.to(device), struc_feat_2.node_v.to(device)), (struc_feat_2.edge_s.to(device), struc_feat_2.edge_v.to(device)), struc_feat_2.edge_index.to(device), struc_feat_2.seq.to(device)
        struc_feat_2 = (h_V, h_E, edge_index, seq)

        label = self.forward(seq_feat_1, struc_feat_1, proteins_num_1, goterms_1, goterms_num_1, seq_feat_2, struc_feat_2, proteins_num_2, goterms_2, goterms_num_2)
        return label

