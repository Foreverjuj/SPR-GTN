import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F




#adj_matrix = np.loadtxt('adj_matrix.txt', delimiter=',')
# feature_matrix = np.loadtxt('feature_matrix.txt', delimiter=',')

# 输入邻接矩阵和特征矩阵



# adj_matrix = torch.from_numpy(adj_matrix).float()
# feature_matrix = torch.from_numpy(feature_matrix).float()

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, k, k0):
        super(GraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.k0 = k0

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(hidden_dim, hidden_dim)
        self.lambda_operator = nn.Linear(hidden_dim, output_dim)

    def subgraph_extractor(self, i, adj_matrix, k, k0):
        subgraph = torch.eye(self.num_nodes)  # 起始子图
        for _ in range(k):
            subgraph = torch.matmul(adj_matrix, subgraph)  # 逐层扩展子图
        subgraph = subgraph[i]  # 提取与节点i相关的子图

        # eigenvalues = torch.symeig(subgraph, eigenvectors=True)[0]  # 提取特征值
        # eigenvalues = eigenvalues[:k0]  # 选择前k0个非零特征值

        # 提取特征值和特征向量
        eigenvalues, eigenvectors = torch.symeig(subgraph, eigenvectors=True)

        # 按特征值降序排序
        indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices]  # 按排序后的顺序重新排列特征值
        eigenvectors = eigenvectors[:, indices]  # 按排序后的顺序重新排列特征向量

        # 非零特征值的索引
        #nonzero_indices = torch.nonzero(eigenvalues)[:, 0]
        nonzero_indices = torch.nonzero(eigenvalues).squeeze(1)

        # 非零特征值的数量可能小于k0
        k0 = min(k0, len(nonzero_indices))

        # 选择前k0个非零特征值
        #eigenvalues = eigenvalues[:k0]

        # 选择与非零特征值对应的特征向量
        eigenvectors = eigenvectors[:, :k0]

        # 将特征向量调整为正确的形状
        eigenvectors = eigenvectors.T  # 调整形状为 (k0, num_nodes)


        #print("Eigenvalues shape:", eigenvalues.shape)
        #print("Eigenvectors shape:", eigenvectors.shape)
        #print("Eigenvectors content:", eigenvectors)

        return eigenvalues,eigenvectors

    def nonlinear_mapping(self, eigenvalues):
        # print("Eigenvalues shape in nonlinear_mapping:", eigenvalues.shape)
        embedding = torch.tanh(eigenvalues)  # 使用非线性映射
        return embedding.unsqueeze(0)

    def forward(self, adj_matrix, feature_matrix):
        h = self.linear(feature_matrix)
        h = F.relu(h)
        h = self.phi(h)
        #eigenvalues = self.subgraph_extractor(torch.arange(self.num_nodes), adj_matrix, self.k, self.k0)
        eigenvalues, eigenvectors = self.subgraph_extractor(torch.arange(self.num_nodes), adj_matrix, self.k, self.k0)
        print("Eigenvalues shape before nonlinear_mapping:", eigenvalues.shape)

        embeddings = self.nonlinear_mapping(eigenvalues)

        print("Embeddings shape before lambda_operator:", embeddings.shape)


        output = self.lambda_operator(embeddings)

        print("Output shape:", output.shape)

        return output

