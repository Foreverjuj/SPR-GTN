from Bio.PDB.PDBParser import PDBParser
import numpy as np
import glob
import sys, os
import pickle as pkl
import argparse
import torch
from torch_geometric.data import Data, Batch
from graph_embding import protein_graph
from network import GraphTransformer
from graph_embding import load_GO_annot
import esm
import random
import time
import datetime
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import sys
import argparse
import warnings



warnings.filterwarnings("ignore")

def simulate_loading(description, duration):
    with tqdm(total=duration, desc=description, leave=True, ncols=100, bar_format="{l_bar}{bar}") as pbar:
        for _ in range(duration):
            time.sleep(1)
            pbar.update(1)







def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


p = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Simulate training for various tasks.")
parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training.")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter.")
parser.add_argument("--k0", type=int, default=3, help="k0 parameter.")
args = p.parse_args()

_, goterms, gonames, _ = load_GO_annot("data/nrPDB-GO_2024.06.24_annot.tsv")

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

pdb = args.pdb
parser = PDBParser()

struct = parser.get_structure("x", pdb)
model = struct[0]
chain_id = list(model.child_dict.keys())[0]
chain = model[chain_id]
Ca_array = []
sequence = ''
seq_idx_list = list(chain.child_dict.keys())
seq_len = seq_idx_list[-1][1] - seq_idx_list[0][1] + 1

for idx in range(seq_idx_list[0][1], seq_idx_list[-1][1] + 1):
    try:
        Ca_array.append(chain[(' ', idx, ' ')]['CA'].get_coord())
    except:
        Ca_array.append([np.nan, np.nan, np.nan])
    try:
        sequence += restype_3to1[chain[(' ', idx, ' ')].get_resname()]
    except:
        sequence += 'X'

# print(sequence)
Ca_array = np.array(Ca_array)

resi_num = Ca_array.shape[0]
G = np.dot(Ca_array, Ca_array.T)
H = np.tile(np.diag(G), (resi_num, 1))
dismap = (H + H.T - 2 * G) ** 0.5

device = f'cuda:{args.device}'

esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm1b_model)

batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)

esm_model.eval()

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

torch.save(batch,'data/processed/train_graph.pt')

if args.task == 'bp':
    output_dim = 1943
elif args.task == 'mf':
    output_dim = 489
elif args.task == 'cc':
    output_dim = 320

model = GraphTransformer(output_dim).to(device)
if args.only_pdbch:
    model.load_state_dict(torch.load(f'model_weight/model_{args.task}CL.pt', map_location=device))
else:
    model.load_state_dict(torch.load(f'model_weight/model_{args.task}CLaf.pt', map_location=device))
model.eval()
with torch.no_grad():
    y_pred = model(batch.to(device)).cpu().numpy()
func_index = np.where(y_pred > args.prob)[1]
if func_index.shape[0] == 0:
    print(f'Sorry, no functions of {args.task.upper()} can be predicted...')
else:
    print(f'The protein may hold the following functions of {args.task.upper()}:')
    for idx in func_index:
        print(
            f'Possibility: {round(float(y_pred[0][idx]), 2)} ||| Functions: {goterms[args.task][idx]}, {gonames[args.task][idx]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate training for various tasks.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter.")
    parser.add_argument("--k0", type=int, default=3, help="k0 parameter.")

    args = parser.parse_args()
