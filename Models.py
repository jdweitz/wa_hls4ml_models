import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch # Batch is implicitly used by DataLoader




class FPGA_GNN(nn.Module):
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=2, mlp_hidden_dim=64, dropout_rate=0.2):
        """
        GraphSAGE based GNN for FPGA resource and latency prediction.

        Args:
            node_feature_dim (int): Dimensionality of node features.
            num_targets (int): Number of target values to predict (6 in your case).
            hidden_dim (int): Hidden dimension for SAGEConv layers.
            num_gnn_layers (int): Number of SAGEConv layers.
            mlp_hidden_dim (int): Hidden dimension for the MLP head.
            dropout_rate (float): Dropout rate.
        """
        super(FPGA_GNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        # Initial SAGEConv layer
        self.convs.append(SAGEConv(node_feature_dim, hidden_dim, aggr='sum'))
        
        # Additional SAGEConv layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='sum'))

        # MLP head for graph-level regression
        # The input to the MLP will be the pooled graph embedding + the strategy feature
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, mlp_hidden_dim), # +1 for the strategy feature
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )
        
        # Optional: Batch normalization for GNN layers (can sometimes help)
        # self.bns = nn.ModuleList()
        # for _ in range(num_gnn_layers):
        #     self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        """
        Forward pass of the GNN.

        Args:
            data (torch_geometric.data.Batch or torch_geometric.data.Data):
                A batch of graph data containing:
                - data.x: Node features [num_nodes_in_batch, node_feature_dim]
                - data.edge_index: Edge connectivity [2, num_edges_in_batch]
                - data.batch: Batch assignment vector [num_nodes_in_batch]
                - data.strategy: Global strategy feature [batch_size, 1]
        
        Returns:
            torch.Tensor: Predicted target values [batch_size, num_targets]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embeddings through SAGEConv layers
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            # x = self.bns[i](x) # If using BatchNorm
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global pooling (sum aggregation as discussed)
        # Alternatives: global_mean_pool, global_max_pool
        graph_embedding = global_add_pool(x, batch) # [batch_size, hidden_dim]
        
        # Concatenate global strategy feature
        # data.strategy should have shape [batch_size, 1]
        strategy_feature = data.strategy
        if strategy_feature.dim() == 1: # Ensure it's [batch_size, 1]
            strategy_feature = strategy_feature.unsqueeze(1)
            
        # Verify batch sizes match before concatenation
        if strategy_feature.shape[0] != graph_embedding.shape[0]:
             # This can happen if the last batch is smaller and drop_last=False in DataLoader
             # For FPGAGraphDataset, strategy is [1,1] for single Data, and collated to [batch_size,1,1]
             # then accessed via batch.strategy, resulting in [batch_size, 1]
             # However, your FPGAGraphDataset's get method shapes strategy to [1,1],
             # so when batched it will be [batch_size, 1, 1]. Need to squeeze.
            if strategy_feature.dim() == 3 and strategy_feature.shape[2] == 1:
                strategy_feature = strategy_feature.squeeze(-1) # [batch_size, 1]
            
            # If after squeeze, it's still not matching, there might be an issue.
            # This check is mostly a safeguard.
            if strategy_feature.shape[0] != graph_embedding.shape[0]:
                raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) and strategy feature ({strategy_feature.shape[0]})")

        combined_embedding = torch.cat([graph_embedding, strategy_feature], dim=1) # [batch_size, hidden_dim + 1]
        
        # MLP for prediction
        output = self.mlp(combined_embedding) # [batch_size, num_targets]
        
        return output
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class FPGA_GNN_GATv2(nn.Module):
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=3, 
                 num_attention_heads=4, mlp_hidden_dim=64, dropout_rate=0.2, 
                 edge_dim=None, concat_heads=True, residual_connections=True):
        """
        GATv2-based GNN for FPGA resource and latency prediction.

        Args:
            node_feature_dim (int): Dimensionality of node features.
            num_targets (int): Number of target values to predict (6 in your case).
            hidden_dim (int): Hidden dimension for GATv2Conv layers.
            num_gnn_layers (int): Number of GATv2Conv layers.
            num_attention_heads (int): Number of attention heads for GATv2Conv.
            mlp_hidden_dim (int): Hidden dimension for the MLP head.
            dropout_rate (float): Dropout rate.
            edge_dim (int, optional): Edge feature dimension if using edge features.
            concat_heads (bool): Whether to concatenate attention heads or average them.
            residual_connections (bool): Whether to use residual connections between layers.
        """
        super(FPGA_GNN_GATv2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.residual_connections = residual_connections
        
        # Calculate dimensions based on concatenation strategy
        if concat_heads:
            gat_out_dim = hidden_dim * num_attention_heads
        else:
            gat_out_dim = hidden_dim
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()  # Layer normalization
        
        # Initial GATv2Conv layer
        self.convs.append(
            GATv2Conv(node_feature_dim, hidden_dim, 
                      heads=num_attention_heads, 
                      dropout=dropout_rate,
                      edge_dim=edge_dim,
                      concat=concat_heads)
        )
        self.norms.append(nn.LayerNorm(gat_out_dim))
        
        # Additional GATv2Conv layers
        for i in range(num_gnn_layers - 1):
            # For intermediate layers, input dim depends on concat strategy
            in_dim = gat_out_dim
            
            # Keep all layers with same configuration for simplicity
            self.convs.append(
                GATv2Conv(in_dim, hidden_dim, 
                          heads=num_attention_heads,
                          dropout=dropout_rate,
                          edge_dim=edge_dim,
                          concat=concat_heads)
            )
            self.norms.append(nn.LayerNorm(gat_out_dim))
        
        # Projection layers for residual connections if needed
        self.residual_projs = nn.ModuleList()
        if residual_connections:
            # First residual projection
            if node_feature_dim != gat_out_dim:
                self.residual_projs.append(nn.Linear(node_feature_dim, gat_out_dim))
            else:
                self.residual_projs.append(nn.Identity())
            
            # Remaining residual projections - all stay at gat_out_dim
            for i in range(num_gnn_layers - 1):
                self.residual_projs.append(nn.Identity())
        
        # Final projection to reduce dimension before pooling
        self.final_projection = nn.Linear(gat_out_dim, hidden_dim)
        
        # Global pooling - using multiple pooling strategies
        self.pool_weight = nn.Parameter(torch.ones(3) / 3)  # Learnable pooling weights
        
        # MLP head for graph-level regression
        # Input: pooled features (hidden_dim) + strategy feature (1)
        mlp_input_dim = hidden_dim + 1  # +1 for strategy
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )

    def forward(self, data):
        """
        Forward pass of the GATv2-based GNN.

        Args:
            data (torch_geometric.data.Batch or torch_geometric.data.Data):
                A batch of graph data containing:
                - data.x: Node features [num_nodes_in_batch, node_feature_dim]
                - data.edge_index: Edge connectivity [2, num_edges_in_batch]
                - data.batch: Batch assignment vector [num_nodes_in_batch]
                - data.strategy: Global strategy feature [batch_size, 1]
                - data.edge_attr (optional): Edge features [num_edges_in_batch, edge_dim]
        
        Returns:
            torch.Tensor: Predicted target values [batch_size, num_targets]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Store attention weights for potential visualization
        attention_weights = []
        
        # Node embeddings through GATv2Conv layers
        for i in range(self.num_gnn_layers):
            identity = x
            
            # Apply GATv2Conv
            if edge_attr is not None:
                x, (edge_index_out, alpha) = self.convs[i](x, edge_index, edge_attr, 
                                                           return_attention_weights=True)
            else:
                x, (edge_index_out, alpha) = self.convs[i](x, edge_index, 
                                                           return_attention_weights=True)
            
            # Store attention weights (optional, for analysis)
            attention_weights.append(alpha)
            
            # Apply normalization
            x = self.norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)  # ELU often works better with attention mechanisms
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection
            if self.residual_connections:
                proj_identity = self.residual_projs[i](identity)
                x = x + proj_identity
        
        # Project to hidden_dim before pooling
        x = self.final_projection(x)
        
        # Multi-strategy global pooling
        # Combine different pooling strategies with learnable weights
        pool_weights = F.softmax(self.pool_weight, dim=0)
        
        graph_embedding_add = global_add_pool(x, batch)
        graph_embedding_mean = global_mean_pool(x, batch)
        graph_embedding_max = global_max_pool(x, batch)
        
        # Weighted combination of pooling strategies
        graph_embedding = (pool_weights[0] * graph_embedding_add + 
                          pool_weights[1] * graph_embedding_mean + 
                          pool_weights[2] * graph_embedding_max)
        
        # Concatenate global strategy feature
        strategy_feature = data.strategy
        if strategy_feature.dim() == 1:
            strategy_feature = strategy_feature.unsqueeze(1)
        
        # Handle potential dimension mismatches
        if strategy_feature.dim() == 3 and strategy_feature.shape[2] == 1:
            strategy_feature = strategy_feature.squeeze(-1)
        
        # Verify batch sizes match
        if strategy_feature.shape[0] != graph_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) "
                           f"and strategy feature ({strategy_feature.shape[0]})")
        
        combined_embedding = torch.cat([graph_embedding, strategy_feature], dim=1)
        
        # MLP for prediction
        output = self.mlp(combined_embedding)
        
        return output


class FPGA_GNN_GATv2_Enhanced(nn.Module):
    """
    Enhanced version with additional features like edge features and skip connections across all layers.
    """
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=3, 
                 num_attention_heads=4, mlp_hidden_dim=64, dropout_rate=0.2,
                 use_edge_features=True, edge_feature_dim=1):
        super(FPGA_GNN_GATv2_Enhanced, self).__init__()
        
        self.use_edge_features = use_edge_features
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        
        # Edge feature embedding (if using edge features)
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
            edge_dim = hidden_dim // 4
        else:
            edge_dim = None
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Track dimensions for skip connections
        current_dim = node_feature_dim
        skip_dims = [current_dim]
        
        for i in range(num_gnn_layers):
            # GATv2Conv layer
            self.convs.append(
                GATv2Conv(current_dim, hidden_dim, 
                          heads=num_attention_heads,
                          dropout=dropout_rate,
                          edge_dim=edge_dim,
                          concat=True)
            )
            
            # Update current dimension
            current_dim = hidden_dim * num_attention_heads
            skip_dims.append(current_dim)
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(current_dim))
            
            # Skip connection projection
            if i > 0:  # Skip connections from all previous layers
                skip_dim_total = sum(skip_dims[:i+1])
                self.skip_connections.append(
                    nn.Linear(skip_dim_total + current_dim, current_dim)
                )
        
        # Final projection to standard hidden dimension
        self.final_projection = nn.Linear(current_dim, hidden_dim)
        
        # Global attention pooling layer
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, mlp_hidden_dim),  # +1 for strategy
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )
    
    def create_edge_features(self, edge_index, x):
        """
        Create edge features based on node features.
        For now, using the absolute difference in node feature magnitudes.
        """
        row, col = edge_index
        # Compute L2 norm of node features
        node_norms = torch.norm(x, p=2, dim=1)
        # Edge feature is the absolute difference in norms
        edge_features = torch.abs(node_norms[row] - node_norms[col]).unsqueeze(1)
        return edge_features
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Create or process edge features
        if self.use_edge_features:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = self.edge_encoder(data.edge_attr)
            else:
                # Create edge features from node features
                edge_features = self.create_edge_features(edge_index, x)
                edge_attr = self.edge_encoder(edge_features)
        else:
            edge_attr = None
        
        # Store all layer outputs for skip connections
        layer_outputs = [x]
        
        # Forward through GATv2 layers
        for i in range(self.num_gnn_layers):
            # Apply GATv2Conv
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Skip connections from all previous layers
            if i > 0:
                skip_features = torch.cat(layer_outputs + [x], dim=1)
                x = self.skip_connections[i-1](skip_features)
                x = F.elu(x)
            
            layer_outputs.append(x)
        
        # Project to standard hidden dimension
        x = self.final_projection(x)
        
        # Global attention pooling
        attention_scores = self.global_attention(x)
        attention_scores = F.softmax(attention_scores, dim=0)
        
        # Weighted sum based on attention
        graph_embedding = global_add_pool(x * attention_scores, batch)
        
        # Concatenate strategy feature
        strategy_feature = data.strategy
        if strategy_feature.dim() == 1:
            strategy_feature = strategy_feature.unsqueeze(1)
        if strategy_feature.dim() == 3 and strategy_feature.shape[2] == 1:
            strategy_feature = strategy_feature.squeeze(-1)
        
        combined_embedding = torch.cat([graph_embedding, strategy_feature], dim=1)
        
        # MLP prediction
        output = self.mlp(combined_embedding)
        
        return output