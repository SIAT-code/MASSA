from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import math
import numpy as np
import random
import os
import json
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead
from gvp_gnn import StructureEncoder
import gvp.data
from efl import equalized_focal_loss, balanced_equalized_focal_loss

def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    atom_num = torch.tensor(atom_num, dtype=torch.long, device=device)
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    protein_num = torch.tensor(protein_num, dtype=torch.long, device=device)
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)

def pro_mask_tokens(tokens, seq_embs, struc_embs, pros_num):
    pro_max_len = seq_embs[0].shape[0]
    samples_num = len(tokens)
    seq_emb_dim = seq_embs[0].shape[1]
    # h_V, h_E, edge_index, seq = struc_embs
    # h_V_s, h_V_V = h_V  # ([seqs, 6], [seqs, 3, 3])

    # 用1e-3填充sequence的mask token
    seq_mask_emb = torch.tensor([[1e-3]*seq_emb_dim], dtype=torch.float32)
    device = seq_embs[0].device
    seq_mask_emb = seq_mask_emb.to(device)

    # 用1e-3填充h_V(s, V)，用#填充seq的mask token
    h_V_s_mask_emb = torch.tensor([[1e-3]*6], dtype=torch.float32)
    h_V_V_mask_emb = torch.tensor([[1e-3]*3 for _ in range(3)], dtype=torch.float32)
    h_V_s_mask_emb = h_V_s_mask_emb.to(device)
    h_V_V_mask_emb = h_V_V_mask_emb.to(device)

    # h_V, h_E, edge_index, seq = [], [], [], []
    seq_mask_embs, struc_mask_embs, labels = [], [], []
    for i in range(samples_num):
        """
        probability_matrix : tensor, [0.15, ..., 0.15]
        """
        pro_ids = tokens[i].clone()
        pro_ids_without_pad = pro_ids[:pros_num[i]]
        seq_emb = seq_embs[i].clone()
        struc_emb = struc_embs[i].clone()
        h_V_each, h_E_each, edge_index_each, seq_each = (struc_emb.node_s, struc_emb.node_v), (struc_emb.edge_s, struc_emb.edge_v), struc_emb.edge_index, struc_emb.seq
        h_V_s, h_V_V = h_V_each
        h_V_s_embs, h_V_V_embs = h_V_s.clone(), h_V_V.clone()

        probability_matrix = torch.full((len(pro_ids_without_pad), ), 0.15)
        """
        masked_indices : tensor, [False, False, ..., True, ...], 利用伯努利分布把需要mask的索引随机选出来，随机概率小于0.15的作为mask
        labels ：tensor, [-100, id1, -100, ..., idn, -100,..., -100], -100表示非mask的token
        """
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if not masked_indices.any():
            masked_rand_index = random.randint(0, len(pro_ids_without_pad)-1)
            masked_indices[masked_rand_index] = True

        masked_indices_80_mask = torch.bernoulli(torch.full(pro_ids_without_pad.shape, 0.8)).bool() & masked_indices
        masked_indices_10_random_replace = torch.bernoulli(torch.full(pro_ids_without_pad.shape, 0.5)).bool() & masked_indices & ~masked_indices_80_mask
        random_index = torch.randint(pros_num[i], (int(masked_indices_10_random_replace.int().sum()), ), dtype=torch.long)

        pad_index = torch.tensor([False]*(pro_max_len-pros_num[i]), dtype=torch.bool)
        masked_indices = torch.cat((masked_indices, pad_index), dim=0)
        pro_ids[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        if int(h_V_V_embs.sum()) != 0:
            try:
                h_V_s_embs[masked_indices_80_mask] = h_V_s_mask_emb
            except:
                a=1
            h_V_V_embs[masked_indices_80_mask] = h_V_V_mask_emb
            h_V_s_embs[masked_indices_10_random_replace] = h_V_s[random_index]
            h_V_V_embs[masked_indices_10_random_replace] = h_V_V[random_index]

        masked_indices_80_mask = torch.cat((masked_indices_80_mask, pad_index), dim=0)
        seq_emb[masked_indices_80_mask] = seq_mask_emb

        # 10% of the time, we replace masked input tokens with random word
        masked_indices_10_random_replace = torch.cat((masked_indices_10_random_replace, pad_index), dim=0)
        seq_emb[masked_indices_10_random_replace] = seq_embs[i][random_index]

        struc_emb.node_s, struc_emb.node_v = h_V_s_embs, h_V_V_embs
        struc_emb.edge_s, struc_emb.edge_v = h_E_each[0], h_E_each[1]
        struc_emb.edge_index = edge_index_each
        struc_emb.seq = seq_each
        struc_mask_embs.append(struc_emb)
        labels.append(pro_ids.tolist())
        seq_mask_embs.append(seq_emb.tolist())

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    seq_mask_embs = torch.tensor(seq_mask_embs).to(device)
    labels = torch.tensor(labels).to(device)
    return seq_mask_embs, struc_mask_embs, labels

def goterm_mask_tokens(tokens, goterm_embs, goterms_num):
    go_max_len = goterm_embs[0].shape[0]
    samples_num = len(tokens)
    go_emb_dim = goterm_embs[0].shape[1]
    # torch.manual_seed(24)

    go_mask_emb = torch.tensor([[2e-3]* go_emb_dim], dtype=torch.float32)
    device = goterm_embs[0].device
    go_mask_emb = go_mask_emb.to(device)

    go_mask_embs, labels = [], []
    for i in range(samples_num):
        """
        probability_matrix : tensor, [0.15, ..., 0.15]
        """
        go_ids = tokens[i].clone()
        go_ids_without_pad = go_ids[:goterms_num[i]]
        goterm_emb = goterm_embs[i].clone()

        probability_matrix = torch.full((len(go_ids_without_pad), ), 0.15)
        """
        masked_indices : tensor, [False, False, ..., True, ...], 利用伯努利分布把需要mask的索引随机选出来，随机概率小于0.15的作为mask
        labels ：tensor, [-100, id1, -100, ..., idn, -100,..., -100], -100表示非mask的token
        """
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if not masked_indices.any():
            masked_rand_index = random.randint(0, len(go_ids_without_pad)-1)
            masked_indices[masked_rand_index] = True
        
        masked_indices_80_mask = torch.bernoulli(torch.full(go_ids_without_pad.shape, 0.8)).bool() & masked_indices
        masked_indices_10_random_replace = torch.bernoulli(torch.full(go_ids_without_pad.shape, 0.5)).bool() & masked_indices & ~masked_indices_80_mask
        random_index = torch.randint(goterms_num[i], (int(masked_indices_10_random_replace.int().sum()), ), dtype=torch.long)

        pad_index = torch.tensor([False]*(go_max_len-goterms_num[i]), dtype=torch.bool)
        masked_indices = torch.cat((masked_indices, pad_index), dim=0)
        go_ids[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        masked_indices_80_mask = torch.cat((masked_indices_80_mask, pad_index), dim=0)
        goterm_emb[masked_indices_80_mask] = go_mask_emb

        # 10% of the time, we replace masked input tokens with random word
        masked_indices_10_random_replace = torch.cat((masked_indices_10_random_replace, pad_index), dim=0)
        if min(random_index.shape) != 0:
            goterm_emb[masked_indices_10_random_replace] = goterm_embs[i][random_index]

        labels.append(go_ids.tolist())
        go_mask_embs.append(goterm_emb.tolist())

    go_mask_embs = torch.tensor(go_mask_embs).to(device)
    labels = torch.tensor(labels).to(device)
    return go_mask_embs, labels

def remove_no_loss_calculation(logits, targets, cate_num):
    # When targets are - 100, they do not participate in loss calculation
    """
    logits: [batch, seq_len, cate_num]
    targets: [batch, seq_len]
    """
    logits, targets = logits.view(-1, cate_num), targets.view(-1)
    logits = logits[targets!=-100, :]
    targets = targets[targets!=-100]
    return logits, targets

def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def get_mae(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def R2(y_test,y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    SStot=np.sum((y_test-np.mean(y_test))**2)
    SSres=np.sum((y_test-y_pred)**2)
    r2=1-SSres/SStot
    return r2


# def gpu_allocation_sample(n_gpu, sample_num):
#     each_gpu_sample_num = [0] * n_gpu
#     for i in range(sample_num):
#         each_gpu_sample_num[i%n_gpu] += 1
#     gpu_split = []
#     for i in range(n_gpu):
#         if i == 0:
#             gpu_split.append([0, each_gpu_sample_num[i]])
#         else:
#             gpu_split.append([each_gpu_sample_num[i-1], each_gpu_sample_num[i-1]+each_gpu_sample_num[i]])
#             each_gpu_sample_num[i] = each_gpu_sample_num[i-1]+each_gpu_sample_num[i]
#     return gpu_split

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch_size, gradient_accumulation):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation

    def train(self, dataset_tuple, device, dir_input, data_type):
        self.model.train()
        dataset, dataset_structure = dataset_tuple
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size)
        dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)
        # Loss = nn.CrossEntropyLoss()
        loss_total = 0
        self.optimizer.zero_grad()
        current_count = 0
        all_count = len(dataloader)
        spent_time_accumulation = 0
        all_predict_labels, all_real_labels = [], []
        for step, batch in enumerate(zip(dataloader, dataloaderstruc)):
            start_time_batch = time.time()
            start_time_1 = time.time()
            batch1, batch2 = batch
            gpu_split = torch.tensor(list(range(batch1[0].shape[0]))).to(device)
            seq_tokens, go_tokens, pro_ids, goterms, proteins_num, goterms_num, labels = batch1
            max_protein_len_batch = torch.max(proteins_num)
            max_goterms_len_batch = torch.max(goterms_num)
            seq_tokens, go_tokens = seq_tokens[:, :max_protein_len_batch], go_tokens[:, :max_goterms_len_batch]
            seq_tokens, go_tokens, labels = seq_tokens.to(device), go_tokens.to(device), labels.to(device)

            all_seqs_emb = []
            for pro_id in pro_ids:
                seq_emb_path = os.path.join(os.path.join(dir_input, 'sequences_emb_%s' % data_type), pro_id+".npy")
                one_seq_emb = np.load(seq_emb_path, allow_pickle=True)
                proteins_new = np.zeros((max_protein_len_batch, one_seq_emb.shape[1]))
                proteins_new[:one_seq_emb.shape[0], :] = one_seq_emb   
                all_seqs_emb.append(proteins_new.tolist())
            pro_seqs = torch.tensor(all_seqs_emb, dtype=torch.float32)
            pro_seqs = pro_seqs.to(device)

            pro_seqs, goterms = pro_seqs[:, :max_protein_len_batch, :], goterms[:, :max_goterms_len_batch, :]
            pro_seqs, goterms, proteins_num, goterms_num = \
                pro_seqs.to(device), goterms.to(device), proteins_num.to(device), goterms_num.to(device)
            
            batch2 = [batch_each_sample.to(device) for batch_each_sample in batch2]
            end_time_1 = time.time()
            # print("time 1: %g" % (end_time_1-start_time_1))

            start_time_3 = time.time()
            data_pack = (pro_seqs, batch2, gpu_split, goterms, proteins_num, goterms_num)
            predict_labels = self.model(data_pack)
            end_time_3 = time.time()
            # print("time 3: %g" % (end_time_3-start_time_3))

            start_time_5 = time.time()
            
            # loss = Loss(predict_labels, labels)  # CELoss
            loss = equalized_focal_loss(predict_labels, labels)  # EFL loss
            
            loss_total += loss.item()
            loss /= self.gradient_accumulation
            loss.backward()

            # loss = balanced_equalized_focal_loss(predict_labels, labels)  # balanced EFL loss

            predict_scores = torch.argmax(predict_labels, 1)
            all_predict_labels += predict_scores.tolist()
            all_real_labels += labels.tolist()

            end_time_5 = time.time()
            # print("time 5: %g" % (end_time_5-start_time_5))

            start_time_6 = time.time()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            if (step+1) % self.gradient_accumulation == 0 or (step+1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()
            end_time_6 = time.time()
            # print("time 6: %g" % (end_time_6-start_time_6))
            end_time_batch = time.time()
            seconds = end_time_batch-start_time_batch
            spent_time_accumulation += seconds
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_batch = "%02d:%02d:%02d" % (h, m, s)
            m, s = divmod(spent_time_accumulation, 60)
            h, m = divmod(m, 60)
            have_spent_time = "%02d:%02d:%02d" % (h, m, s)

            current_count += 1
            if current_count == all_count:
                print("Finish batch: %d/%d---batch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time))
            else:
                print("Finish batch: %d/%d---tatch time: %s, have spent time: %s" % (current_count, all_count, spend_time_batch, have_spent_time), end='\r')
            
        return loss_total/(step+1), all_predict_labels, all_real_labels

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset_tuple, device, dir_input, data_type):
        self.model.eval()
        dataset, dataset_structure = dataset_tuple
        datasampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.batch_size)
        dataloaderstruc = torch_geometric.loader.DataListLoader(dataset_structure, num_workers=4, batch_size=self.batch_size)
        # Loss = nn.CrossEntropyLoss()
        loss_total = 0
        all_predict_labels, all_real_labels = [], []
        for step, batch in enumerate(zip(dataloader, dataloaderstruc)):
            batch1, batch2 = batch
            # gpu_split = gpu_allocation_sample(n_gpu, batch1[0].shape[0])
            # gpu_split = torch.tensor(gpu_split).to(device)
            gpu_split = torch.tensor(list(range(batch1[0].shape[0]))).to(device)
            seq_tokens, go_tokens, pro_ids, goterms, proteins_num, goterms_num, labels = batch1
            max_protein_len_batch = torch.max(proteins_num)
            max_goterms_len_batch = torch.max(goterms_num)
            seq_tokens, go_tokens = seq_tokens[:, :max_protein_len_batch], go_tokens[:, :max_goterms_len_batch]
            seq_tokens, go_tokens, labels = seq_tokens.to(device), go_tokens.to(device), labels.to(device)

            all_seqs_emb = []
            for pro_id in pro_ids:
                seq_emb_path = os.path.join(os.path.join(dir_input, 'sequences_emb_%s' % data_type), pro_id+".npy")
                one_seq_emb = np.load(seq_emb_path, allow_pickle=True)
                proteins_new = np.zeros((max_protein_len_batch, one_seq_emb.shape[1]))
                proteins_new[:one_seq_emb.shape[0], :] = one_seq_emb   
                all_seqs_emb.append(proteins_new.tolist())
            pro_seqs = torch.tensor(all_seqs_emb, dtype=torch.float32)
            pro_seqs = pro_seqs.to(device)

            pro_seqs, goterms = pro_seqs[:, :max_protein_len_batch, :], goterms[:, :max_goterms_len_batch, :]
            pro_seqs, goterms, proteins_num, goterms_num = \
                pro_seqs.to(device), goterms.to(device), proteins_num.to(device), goterms_num.to(device)
            
            # batch2 = batch2.to(device)
            # h_V, h_E, edge_index, seq = (batch2.node_s, batch2.node_v), (batch2.edge_s, batch2.edge_v), batch2.edge_index, batch2.seq
            batch2 = [batch_each_sample.to(device) for batch_each_sample in batch2]

            data_pack = (pro_seqs, batch2, gpu_split, goterms, proteins_num, goterms_num)
            with torch.no_grad():
                predict_labels = self.model(data_pack)

            # EFL loss 要求 预测的 requires_grad = True
            if not predict_labels.requires_grad:
                predict_labels.requires_grad = True

            # loss = Loss(predict_labels, labels)  # CELoss
            loss = equalized_focal_loss(predict_labels, labels)  # EFL loss
            # loss = balanced_equalized_focal_loss(predict_labels, labels)  # balanced EFL loss

            predict_scores = torch.argmax(predict_labels, 1)
            all_predict_labels += predict_scores.tolist()
            all_real_labels += labels.tolist()

            loss_total += loss.item()

        return loss_total/(step+1), all_predict_labels, all_real_labels

    def save_model(self, model, filename):
        # model_to_save = model
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), filename)


