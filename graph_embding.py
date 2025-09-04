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
    # 将蛋白质序列中的氨基酸字母转换为对应的整数索引
    seq_number = np.array(list("ARNDCQEGHILKMFPSTWYVX"), dtype='|S1').view(np.unit8)
    index = np.array(list(seq),dtype='|S1').view(np.unit8)
    for i in range(seq_number.shape[0]):
        index[index == seq_number[i]] = i

    # treat all unknown characters as gaps
    index[index > 20] = 20
    return index
##

def protein_graph(sequence,edge_index,esm_embed):
    # 创建表示蛋白质结构的图数据
    seq_code = exchange_seq(sequence)
    seq_code = torch.IntTensor(seq_code) #：将氨基酸序列的索引编码转换为 PyTorch 的整数张量
    # # 将边添加到距离在 8.25 以下更可能的对中

    edge_index = torch.LongTensor(edge_index)
    data = Data(x=torch.from_numpy(esm_embed), edge_index=edge_index, native_x=seq_code)
    return data





def load_predicted_PDB(pdbfile):
    # 从 PDB 文件中加载预测的蛋白质结构数据
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()] # 获取结构对象中的所有残基，并将它们存储在列表 residues 中。

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom') # 从 PDB 文件中读取序列信息
    seqs = [str(r.seq) for r in records] # 遍历序列记录，并将其序列转换为字符串，并将所有序列存储在列表 seqs 中

    distances = np.empty((len(residues), len(residues))) # 用于存储残基之间的 C_alpha 原子距离。该数组的大小为残基数量的平方。
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord() # 获取每个残基的 C_alpha 原子的坐标。
            distances[x, y] = np.linalg.norm(one-two) # 计算并存储两个残基之间的 C_alpha 原子距离。

    return distances, seqs[0]  #返回计算得到的残基之间的 C_alpha 原子距离矩阵和第一个序列的字符串表示。

def load_FASTA(filename):
    # 从 FASTA 文件中加载数据，并返回一个包含蛋白质序列标识和序列本身的列表
    infile = open(filename, 'rU')
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):
    # 下载注释信息

    sort = ['mf', 'bp', 'cc']
    # 初始化字典，用于存储蛋白质的注释信息、GO条目、GO名称
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
            prot, prot_goterms = row[0], row[1:] # 第一个元素（蛋白质标识）赋值给变量 prot，将其余元素（蛋白质的GO条目）赋值给变量 prot_goterms。
            prot2annot[prot] = {ont: [] for ont in sort} # 创建一个字典，用于存储该蛋白质在每个本体类型下的注释信息。
            for i in range(3):
                goterm_indices = [goterms[sort[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][sort[i]] = np.zeros(len(goterms[sort[i]]))
                prot2annot[prot][sort[i]][goterm_indices] = 1.0
                counts[sort[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts

def load_EC_annot(filename):
    # 下载ec信息

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
    # 归一化邻接矩阵

    A += np.eye(A.shape[1])
    if symm:
        d = 1.0 / np.sqrt(A.sum(axis = 1))
        D = np.diag(d)
        A = D.dot(A.dot(D))
    else:
        A /= A.sum(axis = 1 )[:, np.newaxis]

    return A

def seq2onehot(seq):
    # 创建 26 维嵌入
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # 将词汇转换为 one-hot
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
            precision_list.append(metrics.precision_score(y_true, np.where(y_pred>=T, 1, 0))) # 计算预测标签和真实标签间的精确率
            recall_list.append(metrics.recall_score(y_true, np.where(y_pred>=T, 1, 0)))  # 计算预测标签和真实标签间的召回率
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







