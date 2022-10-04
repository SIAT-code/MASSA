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
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from asyncio import FastChildWatcher
from model import *
from train_test import *
from gvp_gnn import StructureEncoder
from data_transform import *
from visualization_train_dev_loss import plot_train_dev_metric


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

    # Parameters of goterm encoder
    go_input_dim = 64
    go_dim = 128
    go_n_heads = 4
    go_n_layers = 3
    go_pf_dim = 512
    go_dropout = 0.1

    # Parameters of protein sequence and goterm fusion
    fusion_pro_n_layers = 3
    fusion_go_n_layers = 3

    # 化合物小分子编码的参数
    radius = 3
    T = 1
    input_atom_dim = 39
    input_bond_dim = 10
    compound_dim = 128
    compound_n_heads = 8
    compound_dropout = 0.4
    compound_pf_dim = 512

    # 蛋白和化合物的交互融合
    fusion_com_n_layers = 3

   # Parameters of training
    k = 10  # k folds cross validation
    epochs = 150 
    n_gpu = 1
    gpu_start = 7
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
    train_dataset_type = 'all'
    dev_dataset_type = 'all'
    test_dataset_type = 'all'

    # data path
    data_scale = "2950"  # 100, 2950
    dir_input = "dataset/PPI/SKEMPI/sequence_go_structure_%s/" % data_scale

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

    pro_pro_cross_fusion = ProProCrossFusion(protein_dim, seq_struc_n_heads, fusion_pro_n_layers, seq_struc_pf_dim, seq_struc_dropout)
    regression_layer = RegressionLayer(seq_hid_dim)

    start_time = time.time()

    start_time_read_data = time.time()

    dataset_all = data_read(dir_input, train_dataset_type, max_pro_seq_len, SEED)
    dataset_kfolds = split_kfolds_dataset(dataset_all, k=k) 

    end_time_read_data = time.time()
    seconds = end_time_read_data-start_time_read_data
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    spend_time_read_data = "%02d:%02d:%02d" % (h, m, s)
    read_time_print = "Loading data spends time: %s\n" % spend_time_read_data
    print(read_time_print)
    
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 

    for fold_i in range(k): 

        network = Predictor(seq_encoder, struc_encoder, seq_struc_fusion, goterm_encoder, pro_go_cross_fusion, pro_pro_cross_fusion, regression_layer, add_structure, add_goterm)

         ## loading pretrained model
        pretrained_dict = torch.load("pretrained/epoch150-model-2022-05-25-12:57:55/model--2022-05-25-12:57:55.pth", map_location=torch.device('cpu'))
        network_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in network_dict}
        # 用预训练的参数更新model的参数
        network_dict.update(pretrained_dict)
        network.load_state_dict(network_dict)
    
        dataset_dev = dataset_kfolds[fold_i]
        dataset_train = []
        for j in range(k):
            if j == fold_i:
                continue
            dataset_train += dataset_kfolds[j]

        dataset_test = dataset_dev

        if do_train:
            path_model = 'src_SKEMPI/output/cv-%dfolds-model-%s/cv-fold-%d/' % (k, current_time, fold_i + 1) 
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

            f_results.write(read_time_print)
            

            """Create a dataset and split it into train/dev/test."""
            start_time_transform_data = time.time()

            dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2 = transform_data(dataset_train, max_pro_seq_len, dir_input)
            dataset_train_tuple = (dataset_train_pack, dataset_train_structure_1, dataset_train_structure_2)
            dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2 = transform_data(dataset_dev, max_pro_seq_len, dir_input)
            dataset_dev_tuple = (dataset_dev_pack, dataset_dev_structure_1, dataset_dev_structure_2)
            dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = transform_data(dataset_test, max_pro_seq_len, dir_input)
            dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)

            print("Train dataset num: %d, Dev dataset num: %d, Test dataset num: %d\n" % (len(dataset_train_pack), len(dataset_dev_pack), len(dataset_test_pack)))

            end_time_transform_data = time.time()
            seconds = end_time_transform_data-start_time_transform_data
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            spend_time_transform_data = "%02d:%02d:%02d" % (h, m, s)
            transform_data_print = "Transforming and spliting data spend time: %s\n" % spend_time_transform_data
            print(transform_data_print)
            f_results.write(transform_data_print)

            model = nn.DataParallel(network, device_ids=device_ids)
            model = model.to(device)
            trainer = Trainer(model, lr, weight_decay, batch_size, gradient_accumulation)
            tester = Tester(model, batch_size)
            
            results = ('Epoch\tTime\tLoss_train\tLoss_dev\tMSE_train\tMSE_dev\tMAE_train\tMAE_dev\tR_train\tR_dev')
            with open(file_results, 'w') as f:
                f.write(results + '\n')

            """Start training."""
            print('Training...')
            print(results)
            
            min_loss_dev = float('inf')
            min_corr_dev = -float('inf')
            best_epoch = 0

            loss_train_epochs, loss_dev_epochs = [], []
            mse_train_epochs, mse_dev_epochs = [], []
            mae_train_epochs, mae_dev_epochs = [], []
            corr_train_epochs, corr_dev_epochs = [], []
            for epoch in range(1, epochs+1):
                start_time_epoch = time.time()
                if epoch % decay_interval == 0:
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay

                start_time_1 = time.time()
                loss_train, all_predict_labels_train, all_real_labels_train = trainer.train(dataset_train_tuple, device, dir_input, train_dataset_type)
                end_time_1 = time.time()
                # print("time 1: %g" % (end_time_1-start_time_1))
                start_time_2 = time.time()
                loss_dev, all_predict_labels_dev, all_real_labels_dev = tester.test(dataset_dev_tuple, device, dir_input, dev_dataset_type)
                end_time_2 = time.time()
                # print("time 2: %g" % (end_time_2-start_time_2))

                mse_train = get_mse(all_real_labels_train, all_predict_labels_train)
                mae_train = get_mae(all_real_labels_train, all_predict_labels_train)
                corr_train, p_train = scipy.stats.pearsonr(all_real_labels_train, all_predict_labels_train)
                
                mse_dev = get_mse(all_real_labels_dev, all_predict_labels_dev)
                mae_dev = get_mae(all_real_labels_dev, all_predict_labels_dev)
                corr_dev, p_dev = scipy.stats.pearsonr(all_real_labels_dev, all_predict_labels_dev)

                loss_train_epochs.append(float("%.3f" % loss_train)), loss_dev_epochs.append(float("%.3f" % loss_dev))
                mse_train_epochs.append(float("%.3f" % mse_train)), mse_dev_epochs.append(float("%.3f" % mse_dev))
                mae_train_epochs.append(float("%.3f" % mae_train)), mae_dev_epochs.append(float("%.3f" % mae_dev))
                corr_train_epochs.append(float("%.3f" % corr_train)), corr_dev_epochs.append(float("%.3f" % corr_dev))
                
                end_time_epoch = time.time()
                seconds = end_time_epoch-start_time_epoch
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                spend_time_epoch = "%02d:%02d:%02d" % (h, m, s)
                loss_train_epoch = "%.3f" % loss_train; loss_dev_epoch = "%.3f" % loss_dev
                mse_train_epoch = "%.3f" % mse_train; mse_dev_epoch = "%.3f" % mse_dev
                mae_train_epoch = "%.3f" % mae_train; mae_dev_epoch = "%.3f" % mae_dev
                corr_train_epoch = "%.3f" % corr_train; corr_dev_epoch = "%.3f" % corr_dev

                results = [epoch, spend_time_epoch, loss_train_epoch, loss_dev_epoch, mse_train_epoch, mse_dev_epoch, mae_train_epoch, mae_dev_epoch, corr_train_epoch, corr_dev_epoch]
                with open(file_results, 'a') as f:
                    f.write('\t'.join(map(str, results)) + '\n')
                if loss_dev < min_loss_dev:
                    start_time_test = time.time()

                    loss_test, all_predict_labels_test, all_real_labels_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)

                    mse_test = get_mse(all_real_labels_test, all_predict_labels_test)
                    mae_test = get_mae(all_real_labels_test, all_predict_labels_test)
                    corr_test, p_test = scipy.stats.pearsonr(all_real_labels_test, all_predict_labels_test)

                    with open(file_test_results, 'a') as f_test_results:
                        f_test_results.write("The best model is in epoch %d. Test results: Loss: %.3f, MSE: %.3f, MAE: %.3f, R: %3f\n" % (epoch, loss_test, mse_test, mae_test, corr_test))
                    
                    tester.save_model(model, file_model)
                    best_model = copy.deepcopy(model)
                    min_loss_dev = loss_dev
                    min_corr_dev = corr_dev
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
            dict_loss['mse_train_all'] = mse_train_epochs
            dict_loss['mse_dev_all'] = mse_dev_epochs
            dict_loss['mae_train_all'] = mae_train_epochs
            dict_loss['mae_dev_all'] = mae_dev_epochs
            dict_loss['corr_train_all'] = corr_train_epochs
            dict_loss['corr_dev_all'] = corr_dev_epochs

            df_loss = pd.DataFrame(dict_loss)
            df_loss.to_csv(file_loss, index=False)

            plot_train_dev_metric(list(range(1, epochs+1)), loss_train_epochs, loss_dev_epochs, path_model, 'loss', 'skempi')
            plot_train_dev_metric(list(range(1, epochs+1)), mse_train_epochs, mse_dev_epochs, path_model, 'mse', 'skempi')
            plot_train_dev_metric(list(range(1, epochs+1)), mae_train_epochs, mae_dev_epochs, path_model, 'mae', 'skempi')
            plot_train_dev_metric(list(range(1, epochs+1)), corr_train_epochs, corr_dev_epochs, path_model, 'corr', 'skempi')

            final_print = "All epochs spend %s, where the best model is in epoch %d" % (spend_time, best_epoch)
            print(final_print)
            f_results.write(final_print)
            f_results.close()

        if do_test:
            if not best_model:
                best_model_state_dict = torch.load("src_SKEMPI/output/cv-10folds-model-2022-07-11-23:58:56/cv-fold-1/model--2022-07-11-23:58:56.pth", map_location=torch.device('cpu'))
                network.load_state_dict(best_model_state_dict)
                best_model = network.to(device)

            tester = Tester(best_model, batch_size)

            dataset_test = dataset_dev

            dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2 = transform_data(dataset_test, max_pro_seq_len, dir_input)
            dataset_test_tuple = (dataset_test_pack, dataset_test_structure_1, dataset_test_structure_2)

            print("Test dataset num: %d\n" % (len(dataset_test_pack)))

            loss_test, all_predict_labels_test, all_real_labels_test = tester.test(dataset_test_tuple, device, dir_input, test_dataset_type)

            mse_test = get_mse(all_real_labels_test, all_predict_labels_test)
            mae_test = get_mae(all_real_labels_test, all_predict_labels_test)
            corr_test, p_test = scipy.stats.pearsonr(all_real_labels_test, all_predict_labels_test)
            
            print("Test results: Loss: %.3f, MSE: %.3f, MAE: %.3f, R: %.3f\n" % (loss_test, mse_test, mae_test, corr_test))
