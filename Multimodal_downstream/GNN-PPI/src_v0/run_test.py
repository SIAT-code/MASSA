import os


def run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path,
            index_path, gnn_model, test_all):
    os.system("python src_v1/gnn_test.py \
            --description={} \
            --ppi_path={} \
            --pseq_path={} \
            --vec_path={} \
            --pre_emb_path={} \
            --index_path={} \
            --gnn_model={} \
            --test_all={} \
            ".format(description, ppi_path, pseq_path, vec_path, pre_emb_path,
                    index_path, gnn_model, test_all))

if __name__ == "__main__":
    finetune_type = ''  # random_initial, only_sequence, sequence_structure, structure_go, sequence_go, our_model
    dataset_type = 'string'  # shs27k, shs148k, string    
    mode = 'dfs'  # dfs, bfs

    description = "test"
    
    # String
    ppi_path = "src_v0/data/9606.protein.actions.all_connected.txt"
    pseq_path = "src_v0/data/protein.STRING_all_connected.sequences.dictionary.tsv"
    vec_path = "src_v0/data/vec5_CTC.txt"
    pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type
    # SHS27k
    # ppi_path = "src_v0/data/protein.actions.SHS27k.STRING.txt"
    # pseq_path = "src_v0/data/protein.SHS27k.sequences.dictionary.tsv"
    # vec_path = "src_v0/data/vec5_CTC.txt"
    # pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type
    # SHS148k
    # ppi_path = "src_v0/data/protein.actions.SHS148k.STRING.txt"
    # pseq_path = "src_v0/data/protein.SHS148k.sequences.dictionary.tsv"
    # vec_path = "src_v0/data/vec5_CTC.txt"
    # pre_emb_path = 'src_v0/pretrained_emb/shs_%s_all.pickle' % finetune_type

    index_path = "src_v0/train_valid_index_json_%s/%s.%s.fold1.json" % (finetune_type, dataset_type, mode)
#     gnn_model = "save_model_%s/gnn_%s_%s/gnn_model_train.ckpt" % (dataset_type, mode)  
    gnn_model = "save_model_%s/gnn_%s_%s/gnn_model_valid_best.ckpt" % (finetune_type, dataset_type, mode)

    test_all = "True"

    # test test

    run_func(description, ppi_path, pseq_path, vec_path, pre_emb_path, index_path, gnn_model, test_all)
