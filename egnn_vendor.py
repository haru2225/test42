"""Vendored from mila-iqia/diffusion_for_multi_scale_molecular_dynamics,
local checkout based on commit 2b27142dfc815efd6507a022e117714a9ab55ecd
(including its Cartesian edge-distance modifications),
src/diffusion_for_multi_scale_molecular_dynamics/models/egnn.py and
egnn_utils.py, MIT licensed (see licenses/). E_GCL/EGNN are themselves based
on https://github.com/vgsatorras/egnn (E(n) Equivariant Graph Neural
Networks, Satorras et al., https://arxiv.org/abs/2102.09844).

Adaptations: omitted attention/tanh/normalize options and atom classification,
replaced the AXL return type with a tensor pair, and made coordinate updates
out-of-place. There is no dependency on the source project package.
"""
from typing import Callable, Tuple

import torch
from torch import nn


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sum elements of data grouped by segment_ids into num_segments buckets."""
    result_shape = (num_segments, data.size(1))
    result = torch.zeros(result_shape).to(data)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Average elements of data grouped by segment_ids into num_segments buckets."""
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = torch.zeros(result_shape).to(data)
    count = torch.zeros(result_shape).to(data)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer (Satorras et al., eqs. 3-6)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        coords_agg: str = "mean",
        message_agg: str = "mean",
    ):
        super().__init__()
        self.residual = residual
        self.epsilon = 1e-8
        self.coords_agg_fn = unsorted_segment_sum if coords_agg == "sum" else unsorted_segment_mean
        self.msg_agg_fn = unsorted_segment_sum if message_agg == "sum" else unsorted_segment_mean

        message_input_size = input_size * 2 + 1
        self.message_mlp = nn.Sequential(nn.Linear(message_input_size, message_hidden_dimensions_size), act_fn)
        for _ in range(message_n_hidden_dimensions):
            self.message_mlp.append(nn.Linear(message_hidden_dimensions_size, message_hidden_dimensions_size))
            self.message_mlp.append(act_fn)

        node_input_size = input_size + message_hidden_dimensions_size
        self.node_mlp = nn.Sequential(nn.Linear(node_input_size, node_hidden_dimensions_size), act_fn)
        for _ in range(node_n_hidden_dimensions):
            self.node_mlp.append(nn.Linear(node_hidden_dimensions_size, node_hidden_dimensions_size))
            self.node_mlp.append(act_fn)
        self.node_mlp.append(nn.Linear(node_hidden_dimensions_size, output_size))

        coordinate_input_size = message_hidden_dimensions_size
        self.coord_mlp = nn.Sequential(nn.Linear(coordinate_input_size, coordinate_hidden_dimensions_size))
        self.coord_mlp.append(act_fn)
        for _ in range(coordinate_n_hidden_dimensions):
            self.coord_mlp.append(nn.Linear(coordinate_hidden_dimensions_size, coordinate_hidden_dimensions_size))
            self.coord_mlp.append(act_fn)
        self.coord_mlp.append(nn.Linear(coordinate_hidden_dimensions_size, 1, bias=False))

    def message_model(self, source: torch.Tensor, target: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        out = torch.cat([source, target, radial], dim=1)
        return self.message_mlp(out)

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        row = edge_index[:, 0].long()
        agg = self.msg_agg_fn(messages, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord: torch.Tensor, edge_index: torch.Tensor, coord_diff: torch.Tensor,
                     messages: torch.Tensor) -> torch.Tensor:
        row = edge_index[:, 0].long()
        trans = coord_diff * self.coord_mlp(messages)
        agg = self.coords_agg_fn(trans, row, num_segments=coord.size(0))
        return coord + agg

    def coord2radial(self, edge_index: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        return radial, coord_diff

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """edge_index: [n_edges, 3] = (src, dst, cartesian_distance_A)."""
        row, col = edge_index[:, 0].long(), edge_index[:, 1].long()
        _, coord_diff = self.coord2radial(edge_index, coord)
        radial = edge_index[:, 2].unsqueeze(1)  # real cartesian distance, Angstrom
        messages = self.message_model(h[row], h[col], radial)
        coord = self.coord_model(coord, edge_index, coord_diff, messages)
        h = self.node_model(h, edge_index, messages)
        return h, coord


class EGNN(nn.Module):
    """EGNN model: stacks E_GCL layers, embedding node features in and out."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        coords_agg: str = "mean",
        message_agg: str = "mean",
        n_layers: int = 4,
    ):
        super().__init__()
        self.embedding_in = nn.Linear(input_size, node_hidden_dimensions_size)
        self.graph_layers = nn.ModuleList([
            E_GCL(
                input_size=node_hidden_dimensions_size,
                output_size=node_hidden_dimensions_size,
                message_n_hidden_dimensions=message_n_hidden_dimensions,
                message_hidden_dimensions_size=message_hidden_dimensions_size,
                node_n_hidden_dimensions=node_n_hidden_dimensions,
                node_hidden_dimensions_size=node_hidden_dimensions_size,
                coordinate_n_hidden_dimensions=coordinate_n_hidden_dimensions,
                coordinate_hidden_dimensions_size=coordinate_hidden_dimensions_size,
                act_fn=act_fn,
                residual=residual,
                coords_agg=coords_agg,
                message_agg=message_agg,
            )
            for _ in range(n_layers)
        ])

    def forward(self, h: torch.Tensor, edges: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding_in(h)
        for graph_layer in self.graph_layers:
            h, x = graph_layer(h, edges, x)
        return h, x
