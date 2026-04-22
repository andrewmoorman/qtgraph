"""Shared pytest fixtures for qtgraph tests.

Toy fixtures cover structural cases with concrete, hand-verifiable
expected values. Spatial fixtures use SpatialGridGraph for realistic
and scalable test cases.

Directed toy fixtures use the convention: edges flow left-to-right
and top-to-bottom. Undirected fixtures include both directions.

Expected chunk assignments for each toy fixture are documented in
the fixture docstrings.
"""
import torch
import pytest
from torch_geometric.data import Data, HeteroData

from generators import SpatialGridGraph, SpatialGraphCase


# Toy fixtures (structural cases)

@pytest.fixture(params=["directed", "undirected"])
def minimal_overlap(request) -> dict:
    """4-node chain; one overlap chunk between two core chunks.

    Graph: 0 → 1 → 2 → 3 (directed), labels [0, 0, 1, 1], k=1.

    Directed expected chunks:
        core 0:        {0, 1}
        overlap {0,1}: {2}       (node 2 is 1 hop from core 0's boundary)
        core 1:        {3}
        intra-chunk edges:  core 0: {0→1}
        inter-chunk edges:  (core 0, overlap {0,1}): {1→2}
                            (overlap {0,1}, core 1): {2→3}

    Undirected expected chunks:
        core 0:        {0}
        overlap {0,1}: {1, 2}   (node 1 also reachable from core 1's boundary)
        core 1:        {3}
        intra-chunk edges:  overlap {0,1}: {1↔2}
        inter-chunk edges:  (core 0, overlap {0,1}): {0↔1}
                            (overlap {0,1}, core 1): {2↔3}
    """
    directed = request.param == "directed"
    base = torch.tensor([[0, 1, 2], [1, 2, 3]])
    if directed:
        edge_index = base
    else:
        edge_index = torch.cat([base, base.flip(0)], dim=1)
    data = Data(
        x=torch.arange(4).float().unsqueeze(1),
        edge_index=edge_index,
    )
    return {"data": data, "labels": torch.tensor([0, 0, 1, 1]), "k": 1}


@pytest.fixture
def three_way_overlap() -> dict:
    """4-node graph; one node in a 3-way overlap chunk at a corner.

    Graph: 0 → 3, 1 → 3 (directed), labels [0, 1, 2, 2], k=1.

    Node 3 is in core chunk 2 and is reachable from both core chunk 0's
    and core chunk 1's k-hop expansion.

    Expected chunks:
        core 0:          {0}
        core 1:          {1}
        core 2:          {2}
        overlap {0,1,2}: {3}
        inter-chunk edges: (core 0, overlap {0,1,2}): {0→3}
                           (core 1, overlap {0,1,2}): {1→3}
    """
    data = Data(
        x=torch.arange(4).float().unsqueeze(1),
        edge_index=torch.tensor([[0, 1], [3, 3]]),
    )
    return {"data": data, "labels": torch.tensor([0, 1, 2, 2]), "k": 1}


@pytest.fixture
def k_zero() -> dict:
    """4-node chain; k=0 produces no overlap chunks.

    Graph: 0 → 1 → 2 → 3 (directed), labels [0, 0, 1, 1], k=0.

    Expected chunks:
        core 0: {0, 1}
        core 1: {2, 3}
        intra-chunk edges: core 0: {0→1},  core 1: {2→3}
        inter-chunk edges: (core 0, core 1): {1→2}
    """
    data = Data(
        x=torch.arange(4).float().unsqueeze(1),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    return {"data": data, "labels": torch.tensor([0, 0, 1, 1]), "k": 0}


@pytest.fixture(params=["directed", "undirected"])
def multi_hop(request) -> dict:
    """5-node chain; k=2 pulls nodes 2 hops from the chunk boundary.

    Graph: 0 → 1 → 2 → 3 → 4 (directed), labels [0, 0, 1, 1, 1], k=2.

    Directed expected chunks:
        core 0:        {0, 1}
        overlap {0,1}: {2, 3}    (2 hops from node 1)
        core 1:        {4}

    Undirected expected chunks:
        core 0:        {0}
        overlap {0,1}: {1, 2, 3} (node 1 reachable from core 1's
                                  2-hop expansion via 3→2→1)
        core 1:        {4}
    """
    directed = request.param == "directed"
    base = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_index = base if directed else torch.cat(
        [base, base.flip(0)], dim=1
    )
    data = Data(
        x=torch.arange(5).float().unsqueeze(1),
        edge_index=edge_index,
    )
    return {
        "data": data,
        "labels": torch.tensor([0, 0, 1, 1, 1]),
        "k": 2,
    }


@pytest.fixture
def overlap_eliminates_core() -> dict:
    """3-node graph; k=1 empties core chunk 1 entirely.

    Graph: 0 → 1, 0 → 2 (directed), labels [0, 1, 1], k=1.

    Both nodes in core chunk 1 are reachable from core chunk 0's
    boundary, so they are moved into the overlap chunk.

    Expected chunks:
        core 0:        {0}
        overlap {0,1}: {1, 2}
        core 1:        {}          (empty — should produce a warning)
        inter-chunk edges: (core 0, overlap {0,1}): {0→1, 0→2}
    """
    data = Data(
        x=torch.arange(3).float().unsqueeze(1),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
    )
    return {"data": data, "labels": torch.tensor([0, 1, 1]), "k": 1}


@pytest.fixture
def single_partition() -> dict:
    """4-node chain; all nodes in one partition — no inter-chunk edges.

    Graph: 0 → 1 → 2 → 3 (directed), labels [0, 0, 0, 0], k=1.

    Expected chunks:
        core 0: {0, 1, 2, 3}
        no overlap chunks
        no inter-chunk edges
    """
    data = Data(
        x=torch.arange(4).float().unsqueeze(1),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    return {"data": data, "labels": torch.tensor([0, 0, 0, 0]), "k": 1}


@pytest.fixture
def no_node_features() -> dict:
    """Minimal overlap graph without any node features.

    Same structure as directed minimal_overlap but with no x attribute.
    Tests that partitioning handles graphs without node features.
    """
    data = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        num_nodes=4,
    )
    return {"data": data, "labels": torch.tensor([0, 0, 1, 1]), "k": 1}


@pytest.fixture
def with_edge_features() -> dict:
    """Minimal overlap graph with edge features.

    Same structure as directed minimal_overlap but with an edge_attr
    tensor. Tests that edge attributes are permuted correctly alongside
    edge_index.
    """
    data = Data(
        x=torch.arange(4).float().unsqueeze(1),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.arange(3).float().unsqueeze(1),
    )
    return {"data": data, "labels": torch.tensor([0, 0, 1, 1]), "k": 1}


@pytest.fixture
def minimal_overlap_hetero() -> dict:
    """Minimal hetero graph; one overlap chunk between two core chunks.

    Node types: 'A' (source), 'B' (target).
    Edge type: ('A', 'to', 'B').

    Graph (directed):
        A nodes: 0, 1  — labels [0, 0]
        B nodes: 0, 1  — labels [1, 1]
        Edges: A0→B0, A1→B0, A1→B1
        k=1

    A node 1 is a boundary node (edges to B nodes in core chunk 1).
    B node 0 is reached by core chunk 0's k-hop expansion.

    Expected chunks:
        A core 0:      {A0, A1}
        B core 1:      {B1}
        overlap {0,1}: {B0}
    """
    data = HeteroData()
    data['A'].x = torch.arange(2).float().unsqueeze(1)
    data['B'].x = torch.arange(2).float().unsqueeze(1)
    data['A', 'to', 'B'].edge_index = torch.tensor(
        [[0, 1, 1], [0, 0, 1]]
    )
    return {
        "data": data,
        "labels": {"A": torch.tensor([0, 0]), "B": torch.tensor([1, 1])},
        "k": 1,
    }


# Spatial fixtures — realistic and scalable

@pytest.fixture
def make_spatial_graph():
    """Factory fixture returning SpatialGridGraph for custom configs."""
    return SpatialGridGraph


@pytest.fixture(params=[
    {"grid_size": 4, "tile_size": 2},
    {"grid_size": 8, "tile_size": 2},
    {"grid_size": 8, "tile_size": 4},
])
def small_spatial_graph(request) -> SpatialGraphCase:
    """Small spatial grid graphs for unit testing at multiple scales.

    Produces 2×2 to 4×4 tile configurations, each with a 4-connected
    directed grid edge structure.
    """
    return SpatialGridGraph(**request.param).build()


@pytest.fixture(params=[32, 64, 128, 256])
def benchmark_spatial_graph(request) -> SpatialGraphCase:
    """Spatial graphs at increasing scales for benchmark tests.

    Grid sizes 32–256 with a fixed tile_size of 16, producing
    4–256 partitions.
    """
    return SpatialGridGraph(
        grid_size=request.param,
        tile_size=16,
    ).build()
