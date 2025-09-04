import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.data import Data



#adj_matrix = np.loadtxt('adj_matrix.txt', delimiter=',')
# feature_matrix = np.loadtxt('feature_matrix.txt', delimiter=',')



adj_matrix = torch.from_numpy(adj_matrix).float()
feature_matrix = torch.from_numpy(feature_matrix).float()



class GraphConvolution(nn.Module):
    def __init__(self,in_features,out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, input, adj_matrix):
        
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


       
        self.positional_encoding_mlp = nn.Sequential(
            nn.Linear(embed_dim,embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim,embed_dim)
        )

        self.linear = nn.Linear(embed_dim, embed_dim)  

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
        node_degrees = torch.clamp(node_degrees, min=1)

       

        positional_enc = torch.zeros(self.num_nodes, self.embed_dim)
        # positional_enc = self.positional_encoding_mlp(node_degrees)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(math.log(10000.0) / self.embed_dim))

        positional_enc[:, 0::2] = torch.sin(node_degrees * div_term)
        positional_enc[:, 1::2] = torch.cos(node_degrees * div_term)

        return positional_enc

    def forward(self, adj_matrix, feature_matrix):
       

        
        graph = self.construct_graph(adj_matrix) 
        positional_enc = self.positional_encoding(graph)
        features = feature_matrix + positional_enc


       
        gcn1_output = F.relu(self.gcn1(features,adj_matrix))
        gcn2_output = F.relu(self.gcn2(gcn1_output,adj_matrix))

        
        # gcn2_output = gcn2_output.transpose(0,1)  # 将维度 [batch_size, seq_len, embed_dim] 调整为 [seq_len, batch_size, embed_dim]
        # gcn2_output = gcn2_output.unsqueeze(0)  # 将维度 [seq_len, batch_size, embed_dim] 调整为 [1, seq_len, batch_size, embed_dim]
        # gcn2_output = gcn2_output.permute(1,0,2)  # 将维度 [seq_len, batch_size, embed_dim] 调整为 [batch_size, seq_len, embed_dim]


       
        self_attention_output, _ = self.attention(gcn2_output.unsqueeze(0), gcn2_output.unsqueeze(0), gcn2_output.unsqueeze(0))
        #self_attention_output, _ = self.attention(gcn2_output, gcn2_output, gcn2_output)
        #self_attention_output = self_attention_output.permute(1, 0,2)  
        self_attention_output = self.dropout(self_attention_output)
        # self_attention_output = self.layer_norm(gcn2_output + self_attention_output)
        self_attention_output = self.layer_norm(gcn2_output + self_attention_output)

      
        #support = torch.matmul(adj_matrix, self_attention_output.transpose(0, 1)) 

       
        output = gcn2_output + self_attention_output
        #output = self.linear(support)


        return output






