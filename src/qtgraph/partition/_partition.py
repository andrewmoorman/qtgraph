from dataclasses import dataclass
from torch import Tensor


NodeType = str
EdgeType = tuple[str, str, str]


@dataclass
class _Partition:
    """Index into the permuted node and edge arrays for a homogeneous
    graph.

    All tensor shapes assume P core chunks, O overlap chunks, and M
    inter-chunk edge blocks. Core chunks are indexed 0..P-1; overlap
    chunks are indexed P..P+O-1 in the unified chunk index space used
    by inter_chunk_pairs.

    Attributes
    ----------
    core_node_indptr : Tensor
        Shape (P+1,). CSR pointers into the permuted node array for
        each core chunk.
    core_edge_indptr : Tensor
        Shape (P+1,). CSR pointers into the permuted edge array for
        intra-chunk edges of each core chunk.
    overlap_node_indptr : Tensor
        Shape (O+1,). CSR pointers into the permuted node array for
        each overlap chunk.
    overlap_edge_indptr : Tensor
        Shape (O+1,). CSR pointers for intra-chunk edges of each
        overlap chunk.
    overlap_origins_indptr : Tensor
        Shape (O+1,). CSR pointers into overlap_origins_values for
        each overlap chunk's set of originating core chunk indices.
    overlap_origins_values : Tensor
        Shape (total_origins,). Flat array of core chunk indices,
        partitioned by overlap_origins_indptr.
    inter_edge_indptr : Tensor
        Shape (M+1,). CSR pointers into the permuted edge array for
        each inter-chunk edge block.
    inter_chunk_pairs : Tensor
        Shape (M, 2). Unified chunk indices of the two chunks
        connected by each inter-chunk edge block.
    """
    core_node_indptr:       Tensor
    core_edge_indptr:       Tensor
    overlap_node_indptr:    Tensor
    overlap_edge_indptr:    Tensor
    overlap_origins_indptr: Tensor
    overlap_origins_values: Tensor
    inter_edge_indptr:      Tensor
    inter_chunk_pairs:      Tensor


@dataclass
class _HeteroPartition:
    """Index into the permuted node and edge arrays for a heterogeneous
    graph.

    Node and edge ranges are stored per type. Overlap chunk origins and
    inter-chunk chunk pair indices are shared across types since overlap
    chunks are defined by core chunk membership, not node type.

    Attributes
    ----------
    core_node_indptr : dict[NodeType, Tensor]
        Shape (P+1,) per node type. CSR pointers into the permuted node
        array for each core chunk.
    core_edge_indptr : dict[EdgeType, Tensor]
        Shape (P+1,) per edge type. CSR pointers for intra-chunk edges
        of each core chunk.
    overlap_node_indptr : dict[NodeType, Tensor]
        Shape (O+1,) per node type. CSR pointers for each overlap chunk.
    overlap_edge_indptr : dict[EdgeType, Tensor]
        Shape (O+1,) per edge type. CSR pointers for intra-chunk edges
        of each overlap chunk.
    overlap_origins_indptr : Tensor
        Shape (O+1,). Shared across types. CSR pointers into
        overlap_origins_values for each overlap chunk.
    overlap_origins_values : Tensor
        Shape (total_origins,). Shared across types. Flat array of core
        chunk indices per overlap chunk.
    inter_edge_indptr : dict[EdgeType, Tensor]
        Shape (M+1,) per edge type. CSR pointers for inter-chunk edge
        blocks.
    inter_chunk_pairs : Tensor
        Shape (M, 2). Shared across types. Unified chunk indices of the
        two chunks connected by each inter-chunk edge block.
    """
    core_node_indptr:       dict[NodeType, Tensor]
    core_edge_indptr:       dict[EdgeType, Tensor]
    overlap_node_indptr:    dict[NodeType, Tensor]
    overlap_edge_indptr:    dict[EdgeType, Tensor]
    overlap_origins_indptr: Tensor
    overlap_origins_values: Tensor
    inter_edge_indptr:      dict[EdgeType, Tensor]
    inter_chunk_pairs:      Tensor
