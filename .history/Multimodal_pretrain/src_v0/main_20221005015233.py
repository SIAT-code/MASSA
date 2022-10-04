import torch
import numpy as np
import random
import pandas as pd
import os
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from asyncio import FastChildWatcher
from model import *
from train_test import *
from gvp_gnn import StructureEncoder
from data_transform import load_tensor_for_pro_ids, transform_data, load_tensor, split_dataset, shuffle_dataset, CATHDataset, ProteinGraphDataset
from visualization_train_dev_loss import plot_train_dev_loss

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    # Parameters of protein sequence encoder
    seq_dim = 768
    seq_hid_dim = 512
    seq_encoder_layer_num = 3
    kernel_size = 7
    seq_dropout = 0.3

    # 蛋白结构编码器的参数
    max_nodes = 1000
    struc_hid_dim = 512
    struc_encoder_layer_num = 2
    node_in_dim = (6, 3)  # node dimensions in input graph, should be (6, 3) if using original features
    node_h_dim = (struc_hid_dim, 16)  # node dimensions to use in GVP-GNN layers
    edge_in_dim = (32, 1)  # edge dimensions in input graph, should be (32, 1) if using original features
    edge_h_dim = (32, 1)  # edge dimensions to embed to before use in GVP-GNN layers
    struc_dropout = 0.4

    # 蛋白序列和结构融合的参数
    protein_dim = 512
    max_pro_seq_len = 1022
    seq_struc_n_heads = 8
    seq_struc_n_layers = 4
    seq_struc_pf_dim = 2048
    seq_struc_dropout = 0.1

    # Goterm编码器的参数
    go_input_dim = 64
    go_dim = 128
    go_n_heads = 4
    go_n_layers = 3
    go_pf_dim = 512
    go_dropout = 0.1

    # 蛋白和Goterm的交互融合
    fusion_pro_n_layers = 3
    fusion_go_n_layers = 3

    # 基本参数
    epochs = 150
    n_gpu = 1
    gpu_start = 1
    optional_mode = False  # Set whether you need to select the GPU ID yourself
    optional_gpus = [0, 1, 3]
    batch_size = 4  # 12, 4
    gradient_accumulation = 8
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    do_train = True

    # 数据文件路径
    data_scale = "1k"  # 1k, 63759
    dir_input = "dataset/sequence_go_structure_%s/" % data_scale
    tokens_input = "dataset/sequence_go_structure_%s/sequence_go.txt" % data_scale
    motif_input = "dataset/sequence_go_structure_%s/motif.txt" % data_scale
    domain_input = "dataset/sequence_go_structure_%s/domain.txt" % data_scale
    region_input = "dataset/sequence_go_structure_%s/region.txt" % data_scale

    if data_scale == "1k":
        pro_cate = 21  # from file "dataset/sequence_go_structure_1k/pro2id.json"
        go_cate = 2028  # from file "dataset/sequence_go_structure_1k/go2id.json"
        motif_cate = 102  # from file "dataset/sequence_go_structure_1k/motif2id.json"
        domain_cate = 564  # from file "dataset/sequence_go_structure_1k/domain2id.json"
        region_cate = 223  # from file "dataset/sequence_go_structure_1k/region2id.json"
    elif data_scale == "63759":
        pro_cate = 20  # from file "dataset/sequence_go_structure_63759/pro2id.json"
        go_cate = 18438  # from file "dataset/sequence_go_structure_63759/go2id.json"
        motif_cate = 336  # from file "dataset/sequence_go_structure_63759/motif2id.json"
        domain_cate = 2065  # from file "dataset/sequence_go_structure_63759/domain2id.json"
        region_cate = 422  # from file "dataset/sequence_go_structure_63759/region2id.json"

    """CPU or GPU"""
    if torch.cuda.is_available():
        if optional_mode:
            device_ids = optional_gpus
            n_gpu = len(device_ids)
            device = torch.device("cuda:{}".format(device_ids[0]))
        else:
            device_ids = []
            device = torch.device("cuda:{}".format(gpu_start))
            for i in range(n_gpu):
                device_ids.append(gpu_start+i)
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    
    assert batch_size >= n_gpu, "Batch size must be greater than the number of GPUs used!!!"

    """ create model, trainer and tester """
    seq_encoder = SeqEncoder(seq_dim, seq_hid_dim, seq_encoder_layer_num, kernel_size, max_pro_seq_len, dropout=seq_dropout)
    struc_encoder = StructureEncoder(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in=False, num_layers=struc_encoder_layer_num, drop_rate=struc_dropout)
    seq_struc_fusion = SeqStrucFusion(seq_struc_n_layers, seq_struc_n_heads, seq_struc_pf_dim, seq_hid_dim, struc_hid_dim, protein_dim, seq_struc_dropout)
    goterm_encoder = GotermEncoder(go_input_dim, go_dim, go_n_heads, go_n_layers, go_pf_dim, go_dropout)
    pro_go_cross_fusion = ProGoCrossFusion(protein_dim, go_dim, pro_cate, go_cate, motif_cate, domain_cate, region_cate, seq_struc_n_heads, go_n_heads, fusion_pro_n_layers, fusion_go_n_layers, seq_struc_pf_dim, go_pf_dim, seq_struc_dropout, go_dropout)
    model = Predictor(seq_encoder, struc_encoder, seq_struc_fusion, goterm_encoder, pro_go_cross_fusion)

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    
    if do_train:
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
        path_model = 'src_v0/output/model-%s/' % current_time
        """Output files."""
        if not os.path.exists(path_model):
            os.system("mkdir -p %s" % path_model)
        result_file = 'results--%s.txt' % current_time
        model_file = 'model--%s.pth' % current_time
        loss_file = 'loss--%s.csv' % current_time
        file_results = os.path.join(path_model, result_file)
        file_model = os.path.join(path_model, model_file)
        file_loss = os.path.join(path_model, loss_file)
        f_results = open(file_results, 'a')

        start_time = time.time()

        start_time_read_data = time.time()
        df_tokens = pd.read_csv(tokens_input, sep=',', header=None)
        seq_tokens, go_tokens = df_tokens[1].values, df_tokens[2].values
        go_tokens  = list(go_tokens)
        seq_tokens_tmp = []
        for seq_token in seq_tokens:
            seq_tokens_tmp.append(seq_token[: max_pro_seq_len])
        seq_tokens = seq_tokens_tmp

        motif_tokens = []
        with open(motif_input) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')[: max_pro_seq_len]
                motif_tokens.append(line)

        domain_tokens = []
        with open(domain_input) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')[: max_pro_seq_len]
                domain_tokens.append(line)

        region_tokens = []
        with open(region_input) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')[: max_pro_seq_len]
                region_tokens.append(line)

        """Load preprocessed data."""
        pro_ids = load_tensor_for_pro_ids(os.path.join(dir_input, 'seqs_id.txt'), torch.FloatTensor)
        goterms = load_tensor(os.path.join(dir_input, 'goterms'), torch.FloatTensor)
        cath = CATHDataset(os.path.join(dir_input, 'structure.jsonl'))

        print("Check whether the length of sequence and structure is the same.")
        unequal_count = 0
        for i in reversed(range(len(seq_tokens))):
            if len(seq_tokens[i]) != len(cath.data[i]['seq']) and (len(seq_tokens[i]) < max_pro_seq_len or len(cath.data[i]['seq']) < max_pro_seq_len):
                unequal_count += 1
                del seq_tokens[i]
                del cath.data[i]
        print("Unequal count: %d, delete %d samples!!!" % (unequal_count, unequal_count))

        end_time_read_data = time.time()
        seconds = end_time_read_data-start_time_read_data
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time_read_data = "%02d:%02d:%02d" % (h, m, s)
        read_time_print = "Loading data spends time: %s" % spend_time_read_data
        print(read_time_print)
        f_results.write(read_time_print)

        """Create a dataset and split it into train/dev/test."""
        start_time_transform_data = time.time()

        dataset = list(zip(seq_tokens, go_tokens, pro_ids, goterms, cath.data, motif_tokens, domain_tokens, region_tokens))
        dataset = shuffle_dataset(dataset, SEED)
        # GPCR: Train dataset num: 10417, Dev dataset num: 2605, max_atom_len: 272, max_protein_len: 1210
        dataset_train, dataset_dev = split_dataset(dataset, 0.8)
        # dataset_train, dataset_dev = dataset_train[:7], dataset_dev[:16]  # debug用的
        dataset_train_pack, dataset_train_structure = transform_data(dataset_train, max_pro_seq_len, dir_input)
        dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
        dataset_dev_pack, dataset_dev_structure = transform_data(dataset_dev, max_pro_seq_len, dir_input)
        dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
        print("Train dataset num: %d, Dev dataset num: %d" % (len(dataset_train_pack), len(dataset_dev_pack)))

        end_time_transform_data = time.time()
        seconds = end_time_transform_data-start_time_transform_data
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time_transform_data = "%02d:%02d:%02d" % (h, m, s)
        transform_data_print = "Transforming and spliting data spend time: %s" % spend_time_transform_data
        print(transform_data_print)
        f_results.write(transform_data_print)

        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        trainer = Trainer(model, lr, weight_decay, batch_size, gradient_accumulation, pro_cate, go_cate, motif_cate, domain_cate, region_cate)
        tester = Tester(model, batch_size, pro_cate, go_cate, motif_cate, domain_cate, region_cate)
        
        results = ('Epoch\tTime\tLoss_train\tLoss_dev')
        with open(file_results, 'w') as f:
            f.write(results + '\n')

        """Start training."""
        print('Training...')
        print(results)
        
        min_loss_dev = float('inf')
        best_epoch = 0

        loss_train_epochs, loss_dev_epochs = [], []
        loss_train_pro_epochs, loss_dev_pro_epochs = [], []
        loss_train_go_epochs, loss_dev_go_epochs = [], []
        loss_train_motif_epochs, loss_dev_motif_epochs = [], []
        loss_train_domain_epochs, loss_dev_domain_epochs = [], []
        loss_train_region_epochs, loss_dev_region_epochs = [], []
        for epoch in range(1, epochs+1):
            start_time_epoch = time.time()
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            start_time_1 = time.time()
            loss_train, loss_train_pro, loss_train_go, loss_train_motif, loss_train_domain, loss_train_region = trainer.train(dataset_train_tuple, device, dir_input, n_gpu)
            end_time_1 = time.time()
            # print("time 1: %g" % (end_time_1-start_time_1))
            start_time_2 = time.time()
            loss_dev, loss_dev_pro, loss_dev_go, loss_dev_motif, loss_dev_domain, loss_dev_region = tester.test(dataset_dev_tuple, device, dir_input, n_gpu)
            end_time_2 = time.time()
            # print("time 2: %g" % (end_time_2-start_time_2))

            loss_train_epochs.append(float("%.3f" % loss_train)), loss_dev_epochs.append(float("%.3f" % loss_dev))
            loss_train_pro_epochs.append(float("%.3f" % loss_train_pro)), loss_dev_pro_epochs.append(float("%.3f" % loss_dev_pro))
            loss_train_go_epochs.append(float("%.3f" % loss_train_go)), loss_dev_go_epochs.append(float("%.3f" % loss_dev_go))
            loss_train_motif_epochs.append(float("%.3f" % loss_train_motif)), loss_dev_motif_epochs.append(float("%.3f" % loss_dev_motif))
            loss_train_domain_epochs.append(float("%.3f" % loss_train_domain)), loss_dev_domain_epochs.append(float("%.3f" % loss_dev_domain))
            loss_train_region_epochs.append(float("%.3f" % loss_train_region)), loss_dev_region_epochs.append(float("%.3f" % loss_dev_region))

            end_time_epoch = time.time()
            seconds = end_time_epoch-start_time_epoch
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)
            loss_train_epoch = "%.3f" % loss_train
            loss_dev_epoch = "%.3f" % loss_dev
            results = [epoch, spend_time_epoch, loss_train_epoch, loss_dev_epoch]
            with open(file_results, 'a') as f:
                f.write('\t'.join(map(str, results)) + '\n')
            if loss_dev < min_loss_dev:
                tester.save_model(model, file_model)
                min_loss_dev = loss_dev
                best_epoch = epoch
            print('\t'.join(map(str, results)))

        end_time = time.time()
        seconds = end_time-start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time = "%02d:%02d:%02d" % (h, m, s)

        dict_loss = {}
        dict_loss['epochs'] = list(range(1, epochs+1))
        dict_loss['loss_train_all'] = loss_train_epochs
        dict_loss['loss_dev_all'] = loss_dev_epochs
        dict_loss['loss_train_pro'] = loss_train_pro_epochs
        dict_loss['loss_dev_pro'] = loss_dev_pro_epochs
        dict_loss['loss_train_go'] = loss_train_go_epochs
        dict_loss['loss_dev_go'] = loss_dev_go_epochs
        dict_loss['loss_train_motif'] = loss_train_motif_epochs
        dict_loss['loss_dev_motif'] = loss_dev_motif_epochs
        dict_loss['loss_train_domain'] = loss_train_domain_epochs
        dict_loss['loss_dev_domain'] = loss_dev_domain_epochs
        dict_loss['loss_train_region'] = loss_train_region_epochs
        dict_loss['loss_dev_region'] = loss_dev_region_epochs

        df_loss = pd.DataFrame(dict_loss)
        df_loss.to_csv(file_loss)

        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, path_model, 'all')
        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_pro_epochs, loss_dev_pro_epochs, path_model, 'protein')
        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_go_epochs, loss_dev_go_epochs, path_model, 'goterm')
        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_motif_epochs, loss_dev_motif_epochs, path_model, 'motif')
        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_domain_epochs, loss_dev_domain_epochs, path_model, 'domain')
        plot_train_dev_loss(list(range(1, epochs+1)), loss_train_region_epochs, loss_dev_region_epochs, path_model, 'region')
        
        final_print = "All epochs spend %s, where the best model is in epoch %d" % (spend_time, best_epoch)
        print(final_print)
        f_results.write(final_print)
        f_results.close()


