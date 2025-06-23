import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, TopKPooling
from torch_geometric.utils import remove_self_loops, to_undirected


class GraphEncoder(nn.Module):
    def __init__(self, num_atom_types=118, electronegativity=1, embedding_dim=16, in_edge_attr_dim=4,
                 latent_feat_dim=24, latent_edge_attr_dim=12, out_nodefeat_dim=64):
        super(GraphEncoder, self).__init__()

        # Node and Edge initializations
        self.atom_embedding = nn.Embedding(num_embeddings=num_atom_types, embedding_dim=embedding_dim)
        self.node_dense = nn.Linear(embedding_dim + electronegativity, latent_feat_dim)  # Project features to latent embedding
        self.edge_dense = nn.Linear(in_edge_attr_dim, latent_edge_attr_dim)  # Assuming 4 features for edge

        # GraphSAGE layers
        #self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        #self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv1 = GATv2Conv(in_channels=latent_feat_dim, out_channels=latent_feat_dim, edge_dim=latent_edge_attr_dim)
        self.conv2 = GATv2Conv(in_channels=latent_feat_dim, out_channels=36, edge_dim=latent_edge_attr_dim)
        # Readout and final layer
        self.readout_layer = global_mean_pool
        self.final_layer = nn.Linear(36, out_nodefeat_dim)  # Adjust output size if needed

    def forward(self, data):
        # Extract graph data: x = node features, edge_index = Shape (2, num_edges), and still data.edge_attributes
        # edge_index = [[source_indices], [target_indices]]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # Duplicate edge features
        #edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        # Debugging print
        #print(f"Graph input shape: {x.shape}")  # Should be (num_nodes, 119) --> x.shape[0] is what I'm interested in
        # #// 118 vector for atomic number one-hot encoding + electronegativity

        # Separate the one-hot atomic number and electronegativity feature
        atom_numb_one_hot = x[:, :-1]  # Take the first 118 features (one-hot encoding)
        electronegativity = x[:, -1].unsqueeze(1)  # Take the last feature and keep it as a column vector

        # Convert one-hot encoding to indices (argmax) and use the embedding layer
        atom_indices = torch.argmax(atom_numb_one_hot, dim=1)  # Get atomic number index from one-hot
        atom_numb = self.atom_embedding(atom_indices.long())  # Convert indices to embeddings

        # Concatenate the learned embedding with electronegativity scalar
        x = torch.cat((atom_numb, electronegativity), dim=1)  # Final node feature representation
        # mapping into latent space of 2 features: atomic number & electronegativity
        #print(f"Graph input shape after embedding: {x.shape}")
        x = self.node_dense(x)
        edge_attr = self.edge_dense(edge_attr)

        # Pass through GraphSAGE layers (message passing)
        #x = self.conv1(x, edge_index)  # Aggregate information from neighbors
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)  # Non-linearity

        #x = self.conv2(x, edge_index)  # Aggregate again
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)

        # Readout (global pooling)
        out = self.readout_layer(x, data.batch)

        # Final layer
        out = self.final_layer(out)
        return out


class NN(nn.Module):
    def __init__(self, gnn_model, n_l1, n_l2, n_l3, dropout_rate):
        super(NN, self).__init__()
        self.gnn_model = gnn_model

        # Fully connected layers for external features (X1: alloys, X2: miller indices)
        self.alloys_dense = nn.Linear(61, 16)  # Adjust input dimension to match X1 size
        #self.miller_dense = nn.Linear(4, 3)  # Adjust input dimension to match X2 size

        # Fully connected layers after concatenation
        self.dense_1 = nn.Linear(16 + 4 + 64, n_l1)  # Concatenate X1, X2, and GNN output
        self.relu_1 = nn.SiLU()
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dense_2 = nn.Linear(n_l1, n_l2)
        self.relu_2 = nn.SiLU()
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dense_3 = nn.Linear(n_l2, n_l3)
        self.relu_3 = nn.SiLU()
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(n_l3, 1)  # Output layer for regression

    def forward(self, input_cat, input_mill, graph_input):
        # Pass through the external feature layers
        cat_features = self.alloys_dense(input_cat)
        #miller_features = self.miller_dense(input_mill)

        # Get GNN features
        gnn_output = self.gnn_model(graph_input)

        concatenated_features = torch.cat((cat_features, input_mill, gnn_output), dim=1)

        # Pass through fully connected layers
        x = self.dense_1(concatenated_features)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.relu_3(x)
        x = self.dropout_3(x)

        return self.output_layer(x)


class Hybrid(nn.Module):
    def __init__(self, gnn_model, n_l1=648, n_l2=430, n_l3=131, dropout_rate=0.3):
        super(Hybrid, self).__init__()
        self.nn_model = NN(gnn_model=gnn_model, n_l1=n_l1, n_l2=n_l2, n_l3=n_l3, dropout_rate=dropout_rate)

    def forward(self, input_cat, input_mill, graph_input):
        return self.nn_model(input_cat, input_mill, graph_input)

