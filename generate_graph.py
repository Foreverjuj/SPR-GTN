import numpy as np
from Bio.PDB import PDBParser
import torch
from torch_geometric.data import Batch
from graph_embding import protein_graph
from graph_embding import load_GO_annot
import esm
import argparse
import os



def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


def generate_graph_data(pdb_path, esm1b_model_path, device, output_path='data/processed/train_graph.pt'):
    # 加载GO注释数据
    _, goterms, gonames, _ = load_GO_annot("data/nrPDB-GO_2024.06.24_annot.tsv")

    # 定义氨基酸类型的映射关系
    restype_1to3 = {
        'A': 'ALA',
        'R': 'ARG',
        'N': 'ASN',
        'D': 'ASP',
        'C': 'CYS',
        'Q': 'GLN',
        'E': 'GLU',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'L': 'LEU',
        'K': 'LYS',
        'M': 'MET',
        'F': 'PHE',
        'P': 'PRO',
        'S': 'SER',
        'T': 'THR',
        'W': 'TRP',
        'Y': 'TYR',
        'V': 'VAL',
    }

    restype_3to1 = {v: k for k, v in restype_1to3.items()}

    # 解析PDB文件
    parser = PDBParser()
    struct = parser.get_structure("x", pdb_path)
    model = struct[0]
    chain_id = list(model.child_dict.keys())[0]
    chain = model[chain_id]

    # 定义存储坐标和序列的变量
    Ca_array = []
    sequence = ''
    seq_idx_list = list(chain.child_dict.keys())
    seq_len = seq_idx_list[-1][1] - seq_idx_list[0][1] + 1

    # 提取坐标和序列
    for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
        try:
            Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
        except:
            Ca_array.append([np.nan, np.nan, np.nan])
        try:
            sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
        except:
            sequence += 'X'

    Ca_array = np.array(Ca_array)

    resi_num = Ca_array.shape[0]
    G = np.dot(Ca_array, Ca_array.T)
    H = np.tile(np.diag(G), (resi_num, 1))
    dismap = (H + H.T - 2 * G) ** 0.5

    # 加载ESM模型
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm1b_model_path)
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()

    # 提取ESM嵌入表示
    batch_labels, batch_strs, batch_tokens = batch_converter([('tmp', sequence)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33][0].cpu().numpy().astype(np.float16)
        esm_embed = token_representations[1:len(sequence) + 1]

    row, col = np.where(dismap <= 10)
    edge = [row, col]
    graph = protein_graph(sequence, edge, esm_embed)
    batch = Batch.from_data_list([graph])

    # 保存图数据
    torch.save(batch, output_path)

# 示例调用
pdb_path = 'path/to/your/file.pdb'
esm1b_model_path = 'model_weight/esm1b.pt'
device = 'cuda:0'
generate_graph_data(pdb_path, esm1b_model_path, device)
