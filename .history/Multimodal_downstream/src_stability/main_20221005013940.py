import torch
import numpy as np
import random
import pandas as pd
import os
import time
import copy
import scipy
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import r2_score
from asyncio import FastChildWatcher
from model import *
from train_test import *
from gvp_gnn import StructureEncoder
from data_transform import load_tensor_for_pro_ids, transform_data, load_tensor, split_dataset, shuffle_dataset, CATHDataset, ProteinGraphDataset
from visualization_train_dev_loss import plot_train_dev_metric


def data_read(tokens_input, data_type):
    df_tokens = pd.read_csv(tokens_input, sep=',', header=None)
    seq_tokens, go_tokens, labels = df_tokens[1].values.tolist(), df_tokens[2].values.tolist(), df_tokens[3].values.tolist()
    seq_tokens_tmp = []
    for seq_token in seq_tokens:
        seq_tokens_tmp.append(seq_token[: max_pro_seq_len])
    seq_tokens = seq_tokens_tmp

    """Load preprocessed data."""
    pro_ids = load_tensor_for_pro_ids(os.path.join(dir_input, 'seqs_id_%s.txt' % data_type), torch.FloatTensor)

    goterms = load_tensor(os.path.join(dir_input, 'goterms_%s' % data_type), torch.FloatTensor)

    cath = CATHDataset(os.path.join(dir_input, 'structure_%s.jsonl' % data_type))
    print("Check whether the length of sequence and structure is the same.")
    unequal_count = 0
    for i in reversed(range(len(seq_tokens))):
        if len(seq_tokens[i]) != len(cath.data[i]['seq']) and (len(seq_tokens[i]) < max_pro_seq_len or len(cath.data[i]['seq']) < max_pro_seq_len):
            unequal_count += 1
            del seq_tokens[i]
            del cath.data[i]
    print("Unequal count: %d, delete %d samples!!!" % (unequal_count, unequal_count))
    dataset = list(zip(seq_tokens, go_tokens, pro_ids, goterms, cath.data, labels))

    dataset = shuffle_dataset(dataset, SEED)

    return dataset

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

    # Parameters of protein structure encoder
    max_nodes = 1000
    struc_hid_dim = 512
    struc_encoder_layer_num = 2
    node_in_dim = (6, 3)  # node dimensions in input graph, should be (6, 3) if using original features
    node_h_dim = (struc_hid_dim, 16)  # node dimensions to use in GVP-GNN layers
    edge_in_dim = (32, 1)  # edge dimensions in input graph, should be (32, 1) if using original features
    edge_h_dim = (32, 1)  # edge dimensions to embed to before use in GVP-GNN layers
    struc_dropout = 0.4

    # Parameters of protein sequence and structural fusion
    protein_dim = 512
    max_pro_seq_len = 1022
    seq_struc_n_heads = 8
    seq_struc_n_layers = 4
    seq_struc_pf_dim = 2048
    seq_struc_dropout = 0.1

    # Parameters of protein structure encoder
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
    epochs = 10 # 150
    n_gpu = 1
    gpu_start = 3
    optional_mode = False  # Set whether you need to select the GPU ID yourself
    optional_gpus = [0, 1, 3]
    batch_size = 8
    gradient_accumulation = 8
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    do_train = True
    do_test = True
    add_structure = True
    add_goterm = True
    best_model = None
    train_data_type = 'train'
    dev_data_type = 'valid'
    test_data_type = 'test'

    # 数据文件路径
    data_scale = "1k"  # 1k, 53614
    dir_input = "dataset/pretrain_benchmark/stability/sequence_go_structure_%s/" % data_scale
    train_tokens_input = "dataset/pretrain_benchmark/stability/sequence_go_structure_%s/sequence_go_train.txt" % data_scale
    dev_tokens_input = "dataset/pretrain_benchmark/stability/sequence_go_structure_%s/sequence_go_valid.txt" % data_scale
    test_tokens_input = "dataset/pretrain_benchmark/stability/sequence_go_structure_%s/sequence_go_test.txt" % data_scale

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
    pro_go_cross_fusion = ProGoCrossFusion(protein_dim, go_dim, seq_struc_n_heads, go_n_heads, fusion_pro_n_layers, fusion_go_n_layers, seq_struc_pf_dim, go_pf_dim, seq_struc_dropout, go_dropout)
    regression_layer = RegressionLayer(seq_hid_dim)
    model = Predictor(seq_encoder, struc_encoder, seq_struc_fusion, goterm_encoder, pro_go_cross_fusion, regression_layer, add_structure, add_goterm)

    ## 加载预训练模型参数
    pretrained_dict = torch.load("pretrained/epoch150-model-2022-05-25-12:57:55/model--2022-05-25-12:57:55.pth", map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    # 过滤掉model中不存在的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 用预训练的参数更新model的参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if do_train:
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
        path_model = 'src_stability/output/model-%s/' % current_time
        """Output files."""
        if not os.path.exists(path_model):
            os.system("mkdir -p %s" % path_model)
        result_file = 'results--%s.txt' % current_time
        model_file = 'model--%s.pth' % current_time
        loss_file = 'loss--%s.csv' % current_time
        result_test_file = 'results_test--%s.txt' % current_time
        file_results = os.path.join(path_model, result_file)
        file_model = os.path.join(path_model, model_file)
        file_loss = os.path.join(path_model, loss_file)
        file_test_results = os.path.join(path_model, result_test_file)
        f_results = open(file_results, 'a')

        start_time = time.time()

        start_time_read_data = time.time()
        dataset_train = data_read(train_tokens_input, train_data_type)
        dataset_dev = data_read(dev_tokens_input, dev_data_type)
        dataset_test = data_read(test_tokens_input, test_data_type)

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

        dataset_train_pack, dataset_train_structure = transform_data(dataset_train, max_pro_seq_len, dir_input)
        dataset_train_tuple = (dataset_train_pack, dataset_train_structure)
        dataset_dev_pack, dataset_dev_structure = transform_data(dataset_dev, max_pro_seq_len, dir_input)
        dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure)
        dataset_test_pack, dataset_test_structure = transform_data(dataset_test, max_pro_seq_len, dir_input)
        dataset_test_tuple = (dataset_test_pack, dataset_test_structure)

        print("Train dataset num: %d, Dev dataset num: %d, Test dataset num: %d" % (len(dataset_train_pack), len(dataset_dev_pack), len(dataset_test_pack)))

        end_time_transform_data = time.time()
        seconds = end_time_transform_data-start_time_transform_data
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        spend_time_transform_data = "%02d:%02d:%02d" % (h, m, s)
        transform_data_print = "Transforming and spliting data spend time: %s\n" % spend_time_transform_data
        print(transform_data_print)
        f_results.write(transform_data_print)

        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        trainer = Trainer(model, lr, weight_decay, batch_size, gradient_accumulation)
        tester = Tester(model, batch_size)
        
        results = ('Epoch\tTime\tLoss_train\tLoss_dev\tRMSE_train\tRMSE_dev\tMAE_train\tMAE_dev\tSpearman_train\tSpearman_dev')
        with open(file_results, 'w') as f:
            f.write(results + '\n')

        """Start training."""
        print('Training...')
        print(results)
        
        min_loss_dev = float('inf')
        best_epoch = 0

        loss_train_epochs, loss_dev_epochs = [], []
        rmse_train_epochs, rmse_dev_epochs = [], []
        mae_train_epochs, mae_dev_epochs = [], []
        spear_train_epochs, spear_dev_epochs = [], []
        for epoch in range(1, epochs+1):
            start_time_epoch = time.time()
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            start_time_1 = time.time()
            loss_train, all_predict_labels_train, all_real_labels_train = trainer.train(dataset_train_tuple, device, dir_input, train_data_type)
            end_time_1 = time.time()
            # print("time 1: %g" % (end_time_1-start_time_1))
            start_time_2 = time.time()
            loss_dev, all_predict_labels_dev, all_real_labels_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_data_type)
            end_time_2 = time.time()
            # print("time 2: %g" % (end_time_2-start_time_2))

            rmse_train = get_rmse(all_real_labels_train, all_predict_labels_train)
            rmse_dev = get_rmse(all_real_labels_dev, all_predict_labels_dev)
            mae_train = get_mae(all_real_labels_train, all_predict_labels_train)
            mae_dev = get_mae(all_real_labels_dev, all_predict_labels_dev)
            spear_train, p_train = scipy.stats.spearmanr(all_real_labels_train, all_predict_labels_train)
            spear_dev, p_dev = scipy.stats.spearmanr(all_real_labels_dev, all_predict_labels_dev)

            loss_train_epochs.append(float("%.3f" % loss_train)), loss_dev_epochs.append(float("%.3f" % loss_dev))
            rmse_train_epochs.append(float("%.3f" % rmse_train)), rmse_dev_epochs.append(float("%.3f" % rmse_dev))
            mae_train_epochs.append(float("%.3f" % mae_train)), mae_dev_epochs.append(float("%.3f" % mae_dev))
            spear_train_epochs.append(float("%.3f" % spear_train)), spear_dev_epochs.append(float("%.3f" % spear_dev))

            end_time_epoch = time.time()
            seconds = end_time_epoch-start_time_epoch
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)
            loss_train_epoch = "%.3f" % loss_train; loss_dev_epoch = "%.3f" % loss_dev
            rmse_train_epoch = "%.3f" % rmse_train; rmse_dev_epoch = "%.3f" % rmse_dev
            mae_train_epoch = "%.3f" % mae_train; mae_dev_epoch = "%.3f" % mae_dev
            spear_train_epoch = "%.3f" % spear_train; spear_dev_epoch = "%.3f" % spear_dev
            results = [epoch, spend_time_epoch, loss_train_epoch, loss_dev_epoch, rmse_train_epoch, rmse_dev_epoch, mae_train_epoch, mae_dev_epoch, spear_train_epoch, spear_dev_epoch]
            with open(file_results, 'a') as f:
                f.write('\t'.join(map(str, results)) + '\n')
            if loss_dev < min_loss_dev:
                start_time_test = time.time()
                loss_test, all_predict_labels_test, all_real_labels_test = tester.test(dataset_test_tuple, device, dir_input, test_data_type)
                rmse_test = get_rmse(all_real_labels_test, all_predict_labels_test)
                mae_test = get_mae(all_real_labels_test, all_predict_labels_test)
                spear_test, p_test = scipy.stats.spearmanr(all_real_labels_test, all_predict_labels_test)
                with open(file_test_results, 'a') as f_test_results:
                    f_test_results.write("The best model is in epoch %d. Test results: Loss: %.3f, RMSE: %.3f, MAE: %.3f, SPEARMAN: %.3f\n" % (epoch, loss_test, rmse_test, mae_test, spear_test))

                tester.save_model(model, file_model)
                best_model = copy.deepcopy(model)
                min_loss_dev = loss_dev
                best_epoch = epoch

                end_time_test = time.time()
                seconds = end_time_test-start_time_test
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                spend_time_test = "%02d:%02d:%02d" % (h, m, s)
                test_print = "The best model is in epoch %d. Test spend %s\n" % (epoch, spend_time_test)
                print(test_print)

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
        dict_loss['rmse_train_all'] = rmse_train_epochs
        dict_loss['rmse_dev_all'] = rmse_dev_epochs
        dict_loss['mae_train_all'] = mae_train_epochs
        dict_loss['mae_dev_all'] = mae_dev_epochs
        dict_loss['spear_train_all'] = spear_train_epochs
        dict_loss['spear_dev_all'] = spear_dev_epochs

        df_loss = pd.DataFrame(dict_loss)
        df_loss.to_csv(file_loss)

        plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, path_model, 'loss', 'stability')
        plot_train_dev_metric(list(range(1, epochs+1)), rmse_train_epochs, rmse_dev_epochs, path_model, 'rmse', 'stability')
        plot_train_dev_metric(list(range(1, epochs+1)), mae_train_epochs, mae_dev_epochs, path_model, 'mae', 'stability')
        plot_train_dev_metric(list(range(1, epochs+1)), spear_train_epochs, spear_dev_epochs, path_model, 'spearman', 'stability')
        
        final_print = "All epochs spend %s, where the best model is in epoch %d" % (spend_time, best_epoch)
        print(final_print)
        f_results.write(final_print)
        f_results.close()

    if do_test:
        if not best_model:
            best_model_state_dict = torch.load("src_stability/output/model-2022-06-29-23:58:59/model--2022-06-29-23:58:59.pth")
            model.load_state_dict(best_model_state_dict)
            best_model = model.to(device)

        tester = Tester(best_model, batch_size)

        dataset_test = data_read(test_tokens_input, test_data_type)
        dataset_test_pack, dataset_test_structure = transform_data(dataset_test, max_pro_seq_len, dir_input)
        dataset_test_tuple = (dataset_test_pack, dataset_test_structure)
        print("Test dataset num: %d" % len(dataset_test_pack))
        loss_test, all_predict_labels_test, all_real_labels_test = tester.test(dataset_test_tuple, device, dir_input, test_data_type)

        rmse_test = get_rmse(all_real_labels_test, all_predict_labels_test)
        mae_test = get_mae(all_real_labels_test, all_predict_labels_test)
        spear_test, p_test = scipy.stats.spearmanr(all_real_labels_test, all_predict_labels_test)

        print("Test results: Loss: %.3f, RMSE: %.3f, MAE: %.3f, SPEARMAN: %.3f" % (loss_test, rmse_test, mae_test, spear_test))
