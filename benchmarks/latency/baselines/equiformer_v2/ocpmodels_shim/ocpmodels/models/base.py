import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def generate_graph(self, data):
        edge_index = radius_graph(
            data.pos,
            r=self.max_radius,
            batch=data.batch,
            max_num_neighbors=self.max_neighbors,
        )
        j, i = edge_index
        edge_distance_vec = data.pos[i] - data.pos[j]
        edge_distance = edge_distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(
            edge_index.shape[1], 3, device=data.pos.device, dtype=data.pos.dtype
        )
        num_nodes_per_graph = data.natoms
        neighbors = torch.zeros_like(num_nodes_per_graph)
        return edge_index, edge_distance, edge_distance_vec, cell_offsets, None, neighbors
