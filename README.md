# MASSA

Implementation of paper:

Hu, F., Hu, Y., Zhang, W., Huang, H., Pan, Y., & Yin, P. (2023). A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks. Advanced Science, 2301223.
https://doi.org/10.1002/advs.202301223

[![python >3.7.12](https://img.shields.io/badge/python-3.7.12-brightgreen)](https://www.python.org/) 
# Install

[![scipy-1.7.3](https://img.shields.io/badge/scipy-1.7.3-yellowgreen)](https://github.com/scipy/scipy) [![numpy-1.21.5](https://img.shields.io/badge/numpy-1.21.5-red)](https://github.com/numpy/numpy) [![pandas-1.3.0](https://img.shields.io/badge/pandas-1.3.0-lightgrey)](https://github.com/pandas-dev/pandas) [![scikit__learn-0.24.1](https://img.shields.io/badge/scikit__learn-0.24.2-green)](https://github.com/scikit-learn/scikit-learn) [![torch-1.10.1](https://img.shields.io/badge/torch-1.10.1-orange)](https://github.com/pytorch/pytorch)  [![torch_geometric-2.0.3](https://img.shields.io/badge/torch_geometric-2.0.3-green)](https://github.com/pyg-team/pytorch_geometric)

# Data

The data can be downloaded from these links. If you have any question, please contact hz.huang@siat.ac.cn.

Pretrain dataset: https://drive.google.com/file/d/1xHUs0B9VuKviBzj-k-203p4a9vEoo1RW/view?usp=sharing
Downstream dataset: https://drive.google.com/file/d/10yywJNTQ9Z30B_4uyNfQhnXQdhhdjK3W/view?usp=sharing
GNN-PPI data: https://drive.google.com/file/d/1YSXNsTJo-Cdxo08cHLb6ghd6noJJ4y73/view?usp=sharing
GNN-PPI pretrained embedding: https://drive.google.com/file/d/1sq2VQGAMWmWg02hqhyWju2xuiJ-oHbq0/view?usp=sharing
# Checkpoint 

The pre-trained model checkpoint can be downloaded from this link. If you have any question, please contact hz.huang@siat.ac.cn.

https://drive.google.com/file/d/1NVxB00THWxKdTZkLM7T6xdQJM_3TFMVr/view?usp=sharing

# Usage

You can download this repo and run the demo task on your computing machine.

- Pre-train model.
```
cd Multimodal_pretrain/
python src_v0/main.py
```

- Fine-tune on downstream tasks using pre-trained models (downstream tasks: stability, fluorescence, remote homology, secondary structure, pdbbind, kinase).
```
# For example
cd Multimodal_downstream/
python src_stability/main.py
```

- Fine-tune on gnn-ppi using pre-trained embedding.
```
cd Multimodal_downstream/GNN-PPI/
python src_v0/run.py
```

- Guidance for hyperparameter selection.

You can select the hyperparameters of the Performer encoder based on your data and task in:
Hyperparameter|Description                            | Default | Arbitrary range
--------------|---------------------------------------| ------- | ----------------   
seq_dim    |Size of sequence embedding vector  |	768       |	    
seq_hid_dim           |Size of hidden embedding on sequence encoder |	512     |	[128, 256, 512]  
seq_encoder_layer_num         |Number of sequence encoder layers     |	3       |	[3, 4, 5] 
struc_hid_dim         |Size of hidden embedding on structure encoder |	512      |	[128, 256, 512]  
struc_encoder_layer_num         |Number of sequence encoder layers |	2      |	[2, 4, 6] 
go_input_dim         |Size of goterm embedding vector |	64      |
go_dim         |Size of hidden embedding on goterm encoder |	128      |	[128, 256, 512]  
go_n_heads         |Number of attention heads of goterm encoder |	4      |	[4, 8, 16] 
go_n_layers         |Number of goterm encoder layers |	3      |	[3, 4, 5] 

## Citations
If you use our framework in your research, please cite our paper:
```BibTex
@article{hu2023multimodal,
  author={Hu, Fan and Hu, Yishen and Zhang, Weihong and Huang, Huazhen and Pan, Yi and Yin, Peng},
  title={A Multimodal Protein Representation Framework for Quantifying Transferability Across Biochemical Downstream Tasks},
  journal={Advanced Science},
  year={2023},
  pages={2301223},
  doi={10.1002/advs.202301223}
}
```
