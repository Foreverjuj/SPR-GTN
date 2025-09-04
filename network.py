import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision.transforms import RandomAffine

class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, k, k0, alpha):
        super(GraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.k0 = k0
        self.alpha = alpha

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.phi = nn.Linear(hidden_dim, hidden_dim)
        self.lambda_operator = nn.Linear(hidden_dim, output_dim)

  

        self.pe_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.se_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.transform = RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)) 

        self.dropout = nn.Dropout(0.2) 

    def subgraph_extractor(self, i, adj_matrix, k, k0):
        subgraph = torch.eye(self.num_nodes) 
        for _ in range(k):
            subgraph = torch.matmul(adj_matrix, subgraph)  
        subgraph = subgraph[i]  

    
       
        eigenvalues, eigenvectors = torch.symeig(subgraph, eigenvectors=True)

     
        indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices] 
        eigenvectors = eigenvectors[:, indices]  
     
        # nonzero_indices = torch.nonzero(eigenvalues)[:, 0]
        nonzero_indices = torch.nonzero(eigenvalues).squeeze(1)

      
        k0 = min(k0, len(nonzero_indices))

        
        # eigenvalues = eigenvalues[:k0]

        
        eigenvectors = eigenvectors[:, :k0]

      
        eigenvectors = eigenvectors.T  # 调整形状为 (k0, num_nodes)

        # print("Eigenvalues shape:", eigenvalues.shape)
        # print("Eigenvectors shape:", eigenvectors.shape)
        # print("Eigenvectors content:", eigenvectors)

        return eigenvalues, eigenvectors

    def nonlinear_mapping(self, eigenvalues):
        embedding = torch.tanh(eigenvalues)
        return embedding.unsqueeze(0)

    def forward(self, adj_matrix, feature_matrix):
        h = self.linear(feature_matrix)
        h = F.relu(h)
        h = self.phi(h)

        #pe_h = self.pe_gat(h, adj_matrix)  
        #pe_h = F.relu(pe_h)
        #pe_h = self.dropout(pe_h)  
        pe_h = self.pe_mlp(h)

        eigenvalues, eigenvectors = self.subgraph_extractor(torch.arange(self.num_nodes), adj_matrix, self.k, self.k0)

        embeddings = self.nonlinear_mapping(eigenvalues)
        #se_h = self.se_gat(h, adj_matrix)  
        #se_h = F.relu(se_h)
        #se_h = self.dropout(se_h) 
        se_h = self.se_mlp(h)

        combined_output = self.alpha * pe_h + (1 - self.alpha) * se_h

        output = self.lambda_operator(combined_output)

        return output

    def data_augmentation(self, x):
        x = self.transform(x)
        return x

    def regularization(self, x):
        x = self.dropout(x)

        return x
