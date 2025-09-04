import torch
import numpy as np
from torch_geometric.data import Data,Dataset,Batch
import csv
import glob
import warnings
import sys
from joblib import Parallel,delayed,cpu_count
from datetime import datetime
from tqdm import tqdm
from sklearn.exceptions import DataConversionWarning
from sklearn import metrics
from sklearn.utils import resample
from sklearn.metrics import roc_curve,accuracy_score,precision_recall_fscore_support
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser


protein_letters = {
    'A':0,
    'R':1,
    'N':2,
    'D':3,
    'C':4,
    'Q':5,
    'E':6,
    'G':7,
    'H':8,
    'I':9,
    'L':10,
    'K':11,
    'M':12,
    'F':13,
    'P':14,
    'S':15,
    'T':16,
    'W':17,
    'Y':18,
    'V':19,
    '-':20
}

def exchange_seq(seq):
   
    seq_number = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.unit8)
    index = np.array(list(seq),dtype='|S1').view(np.unit8)
    for i in range(seq_number.shape[0]):
        index[index == seq_number[i]] = i

    # treat all unknown characters as gaps
    index[index > 20] = 20
    return index
##

def protein_graph(sequence,edge_index,esm_embed):
   
    seq_code = exchange_seq(sequence)
    seq_code = torch.IntTensor(seq_code) 
  

    edge_index = torch.LongTensor(edge_index)
    data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code)
    return data





def load_predicted_PDB(pdbfile):
   
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()] 

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom') 
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord() 
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]  

def load_FASTA(filename):
  
    infile = open(filename, 'rU')
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):
 

    sort = ['mf', 'bp', 'cc']
    
    prot2annot = {}

    goterms = {ont:[] for ont in sort}
    gonames = {ont:[] for ont in sort}

    with open(filename,mode ='r') as tsvfile:
        reader = csv.reader(tsvfile,delimiter = '\t')


        # MF
        next(reader,None) # 跳过表头信息
        goterms[sort[0]] = next(reader)
        next(reader,None)
        gonames[sort[0]] = next(reader)


        # BP
        next(reader, None)
        goterms[sort[1]] = next(reader)
        next(reader, None)
        gonames[sort[1]] = next(reader)


        # CC
        next(reader, None)
        goterms[sort[2]] = next(reader)
        next(reader, None)
        gonames[sort[2]] = next(reader)



        next(reader, None)
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in sort}
        for row in reader:
            prot, prot_goterms = row[0], row[1:] 
            prot2annot[prot] = {ont: [] for ont in sort} 
            for i in range(3):
                goterm_indices = [goterms[sort[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][sort[i]] = np.zeros(len(goterms[sort[i]]))
                prot2annot[prot][sort[i]][goterm_indices] = 1.0
                counts[sort[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts

def load_EC_annot(filename):


    prot2annot = {}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # mf
        next(reader, None)
        ec_numbers = {'ec': next(reader)}
        next(reader, None)
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1
    return prot2annot, ec_numbers, ec_numbers, counts


def log(*args):
    print(f'[{datetime.now()}]', *args)

def norm_adj(A, symm= True):


    A += np.eye(A.shape[1])
    if symm:
        d = 1.0 / np.sqrt(A.sum(axis = 1))
        D = np.diag(d)
        A = D.dot(A.dot(D))
    else:
        A /= A.sum(axis = 1 )[:, np.newaxis]

    return A

def seq2onehot(seq):

    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))


    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

def PR_metrics(y_true, y_pred):
    precision_list = []
    recall_list = []
    threshold = np.arange(0.01,1.01,0.01)
    for T in threshold:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision_list.append(metrics.precision_score(y_true, np.where(y_pred>=T, 1, 0)))
            recall_list.append(metrics.recall_score(y_true, np.where(y_pred>=T, 1, 0)))  
    return np.array(precision_list), np.array(recall_list)

def fmax(Ytrue, Ypred, nrThresholds):
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(Ytrue, (Ypred >=t).astype(int), average='samples') # f-score

    return np.max(ff)








