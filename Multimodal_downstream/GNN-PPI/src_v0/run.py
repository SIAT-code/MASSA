import os

def run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs):
    os.system("python -u src_v1/gnn_train.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --pre_emb_path={}\
            --split_new={} \
            --split_mode={} \
            --train_valid_index_path={} \
            --use_lr_scheduler={} \
            --save_path={} \
            --graph_only_train={} \
            --batch_size={} \
            --epochs={} \
            ".format(description, ppi_path, pseq_path, vec_path, pre_emb_path,
                    split_new, split_mode, train_valid_index_path,
                    use_lr_scheduler, save_path, graph_only_train, 
                    batch_size, epochs))

if __name__ == "__main__":
    finetune_type = ''  # random_initial, only_sequence, sequence_structure, structure_go, sequence_go, our_model
    dataset_type = 'shs148k'  # shs27k, shs148k, string    
    mode = 'dfs'  # dfs, bfs
#     description = "test_string_bfs"
    description = "test_%s_%s" % (dataset_type, mode)

    # SHS27k
    # ppi_path = "src_v0/data/protein.actions.SHS27k.STRING.txt"
    # pseq_path = "src_v0/data/protein.SHS27k.sequences.dictionary.tsv"
    # vec_path = "src_v0/data/vec5_CTC.txt"
    # pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type
    # SHS148k
    ppi_path = "src_v0/data/protein.actions.SHS148k.STRING.txt"
    pseq_path = "src_v0/data/protein.SHS148k.sequences.dictionary.tsv"
    vec_path = "src_v0/data/vec5_CTC.txt"
    pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type
    # String
    # ppi_path = "src_v0/data/9606.protein.actions.all_connected.txt"
    # pseq_path = "src_v0/data/protein.STRING_all_connected.sequences.dictionary.tsv"
    # vec_path = "src_v0/data/vec5_CTC.txt"
    # pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type

    split_new = "True"
    split_mode = "%s" % mode
    index_path = "src_v0/train_valid_index_json_%s/" % finetune_type
    if not os.path.isdir(index_path):
        os.mkdir(index_path)
    train_valid_index_path = os.path.join(index_path, "%s.%s.fold1.json" % (dataset_type, mode))

    use_lr_scheduler = "True"
    save_path = "./save_model_%s/" % finetune_type
    graph_only_train = "False"

    batch_size = 2048
    epochs = 200

    run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
            split_new, split_mode, train_valid_index_path,
            use_lr_scheduler, save_path, graph_only_train, 
            batch_size, epochs)