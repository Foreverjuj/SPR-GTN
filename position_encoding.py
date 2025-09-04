import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.data import Data



#adj_matrix = np.loadtxt('adj_matrix.txt', delimiter=',')
# feature_matrix = np.loadtxt('feature_matrix.txt', delimiter=',')

adj_matrix = np.array([[0, 1, 0, 1],
                       [1, 0, 1, 0],
                       [0, 1, 0, 1],
                       [1, 0, 1, 0]])

feature_matrix = np.array([[1, 2,7,8],
                           [3, 4,5,6],
                           [5, 6,4,3],
                           [7, 8,2,4]])


adj_matrix = torch.from_numpy(adj_matrix).float()
feature_matrix = torch.from_numpy(feature_matrix).float()



class GraphConvolution(nn.Module):
    def __init__(self,in_features,out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, input, adj_matrix):
        # 执行矩阵乘法之前，确保 input 和 adj_matrix 的维度匹配
        # adj_matrix = adj_matrix.unsqueeze(0)  # 仅进行 unsqueeze，不再进行 repeat
        # input = input.unsqueeze(-1)
        # support = torch.matmul(adj_matrix, input).squeeze(-1)  # 将 support 中的最后一个维度压缩掉

        support = torch.matmul(adj_matrix, input)
        output = self.linear(support)

        return output



class GTN(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super(GTN, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim


        self.gcn1 = GraphConvolution(embed_dim,embed_dim)
        self.gcn2 = GraphConvolution(embed_dim,embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=4)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.5)


        # 定义位置编码映射层
        self.positional_encoding_mlp = nn.Sequential(
            nn.Linear(embed_dim,embed_dim), # 输入和输出维度都是embed——dim
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim)
        )

        self.linear = nn.Linear(embed_dim, embed_dim)  # 添加线性层

        # 在此定义您的 GTN 层或任何其他网络层
    def construct_graph(self, adj_matrix):
        num_nodes = adj_matrix.shape[0]
        graph = {}

        for i in range(num_nodes):
            graph[i] = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    graph[i].append(j)

        return graph

    def positional_encoding(self, graph):

        num_nodes = len(graph)
        node_degrees = torch.zeros(num_nodes)

        for node,neighbors in graph.items():
            node_degrees[node] = len(neighbors)


        node_degrees = node_degrees.float().unsqueeze(1)
        node_degrees = torch.clamp(node_degrees, min=1)  # 避免被零除

        # 计算位置编码

        positional_enc = torch.zeros(self.num_nodes, self.embed_dim)
        # positional_enc = self.positional_encoding_mlp(node_degrees)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))

        positional_enc[:, 0::2] = torch.sin(node_degrees * div_term)
        positional_enc[:, 1::2] = torch.cos(node_degrees * div_term)

        return positional_enc

    def forward(self, adj_matrix, feature_matrix): #features
        # 将 GTN 层应用于特征

        # 对特征应用位置编码
        graph = self.construct_graph(adj_matrix)  # 从邻接矩阵构造图
        positional_enc = self.positional_encoding(graph)
        features = feature_matrix + positional_enc

        # 继续添加其他操作

        # GCN
        gcn1_output = F.relu(self.gcn1(features,adj_matrix))
        gcn2_output = F.relu(self.gcn2(gcn1_output,adj_matrix))

        # 调整维度，以匹配 nn.MultiheadAttention 的要求
        # 调整维度，以匹配 nn.MultiheadAttention 的要求
        # gcn2_output = gcn2_output.transpose(0,1)  # 将维度 [batch_size, seq_len, embed_dim] 调整为 [seq_len, batch_size, embed_dim]
        # gcn2_output = gcn2_output.unsqueeze(0)  # 将维度 [seq_len, batch_size, embed_dim] 调整为 [1, seq_len, batch_size, embed_dim]
        # gcn2_output = gcn2_output.permute(1,0,2)  # 将维度 [seq_len, batch_size, embed_dim] 调整为 [batch_size, seq_len, embed_dim]


        # 自注意力
        self_attention_output, _ = self.attention(gcn2_output.unsqueeze(0), gcn2_output.unsqueeze(0), gcn2_output.unsqueeze(0))
        #self_attention_output, _ = self.attention(gcn2_output, gcn2_output, gcn2_output)
        #self_attention_output = self_attention_output.permute(1, 0,2)  # 将维度 [batch_size, seq_len, embed_dim] 调整为 [seq_len, batch_size, embed_dim]
        self_attention_output = self.dropout(self_attention_output)
        # self_attention_output = self.layer_norm(gcn2_output + self_attention_output)
        self_attention_output = self.layer_norm(gcn2_output + self_attention_output)

        # 矩阵乘法
        #support = torch.matmul(adj_matrix, self_attention_output.transpose(0, 1))  # 调整维度匹配

        # 残差连接
        output = gcn2_output + self_attention_output
        #output = self.linear(support)


        return output



# # 用法示例
num_nodes = adj_matrix.shape[0]  # PPI网络中的节点数量
embed_dim = feature_matrix.shape[1]  # 节点嵌入的维度


gtn_model = GTN(num_nodes, embed_dim)
output = gtn_model(adj_matrix, feature_matrix)

print("Input features:")
print(feature_matrix)
print("Output features after positional encoding:")
print(output)


# 创建随机 PPI 图和节点特征以进行演示
# graph = torch.rand(num_nodes, num_nodes)
# features = torch.rand(num_nodes, embed_dim)


# def train_and_evaluate(adj_matrix, feature_matrix, embed_dim, num_heads):
#     num_nodes = adj_matrix.shape[0]  # 图中节点数量
#     gtn_model = GTN(num_nodes, embed_dim, num_heads)
#
#     # Perform model training and evaluation
#     # ...
#
#     # Return the evaluation score
#     # return evaluation_score
#
# # Define the parameter grid for grid search
# param_grid = {
#     'embed_dim': [4, 8, 16],  # Possible values for embed_dim
#     'num_heads': [2, 4, 8]  # Possible values for num_heads
# }
#
# # Perform grid search
# best_score = float("-inf")
# best_params = {}
#
# for embed_dim in param_grid['embed_dim']:
#     for num_heads in param_grid['num_heads']:
#         score = train_and_evaluate(adj_matrix, feature_matrix, embed_dim, num_heads)
#         if score > best_score:
#             best_score = score
#             best_params = {'embed_dim': embed_dim, 'num_heads': num_heads}

