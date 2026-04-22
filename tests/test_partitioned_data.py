"""Tests for PartitionedDataBuilder and PartitionedData."""
import torch
import pytest
from torch_geometric.data import Data

from qtgraph import partition, PartitionedData, PartitionedDataBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_id_set(edge_index: torch.Tensor, x: torch.Tensor) -> set:
    """Set of (src_id, dst_id) int pairs using single-column x as node IDs."""
    src = x[edge_index[0]].squeeze(1).long().tolist()
    dst = x[edge_index[1]].squeeze(1).long().tolist()
    return set(zip(src, dst))


def _node_id_set(x: torch.Tensor) -> set:
    """Set of node IDs from single-column x tensor."""
    return set(x.squeeze(1).long().tolist())


def _all_edge_ranges(pd: PartitionedData) -> list[tuple[int, int]]:
    """All (lo, hi) index ranges from every intra- and inter-chunk block."""
    p = pd._partition
    n_core = len(pd)
    n_overlap = p.overlap_node_indptr.shape[0] - 1
    n_inter = p.inter_chunk_pairs.shape[0]

    ranges = []
    for i in range(n_core):
        ranges.append((
            p.core_edge_indptr[i].item(),
            p.core_edge_indptr[i + 1].item(),
        ))
    for o in range(n_overlap):
        ranges.append((
            p.overlap_edge_indptr[o].item(),
            p.overlap_edge_indptr[o + 1].item(),
        ))
    for m in range(n_inter):
        ranges.append((
            p.inter_edge_indptr[m].item(),
            p.inter_edge_indptr[m + 1].item(),
        ))
    return ranges


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------

class TestCountInvariants:
    def test_node_count_preserved(self, minimal_overlap):
        c = minimal_overlap
        n_before = c["data"].num_nodes
        pd = partition(c["data"], c["labels"], c["k"])
        assert pd._data.num_nodes == n_before

    def test_edge_count_preserved(self, minimal_overlap):
        c = minimal_overlap
        e_before = c["data"].edge_index.shape[1]
        pd = partition(c["data"], c["labels"], c["k"])
        assert pd._data.edge_index.shape[1] == e_before

    def test_num_core_chunks_equals_num_labels(self, minimal_overlap):
        c = minimal_overlap
        n_partitions = c["labels"].unique().shape[0]
        pd = partition(c["data"], c["labels"], c["k"])
        assert len(pd) == n_partitions

    def test_k_zero_no_overlap_chunks(self, k_zero):
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        assert p.overlap_node_indptr.shape[0] == 1  # O=0 → (1,)

    def test_single_partition_len(self, single_partition):
        c = single_partition
        pd = partition(c["data"], c["labels"], c["k"])
        assert len(pd) == 1

    def test_spatial_node_count_preserved(self, small_spatial_graph):
        g = small_spatial_graph
        n_before = g.data.num_nodes
        pd = partition(g.data, g.labels, k=1)
        assert pd._data.num_nodes == n_before

    def test_spatial_edge_count_preserved(self, small_spatial_graph):
        g = small_spatial_graph
        e_before = g.data.edge_index.shape[1]
        pd = partition(g.data, g.labels, k=1)
        assert pd._data.edge_index.shape[1] == e_before

    def test_spatial_num_core_chunks(self, small_spatial_graph):
        g = small_spatial_graph
        pd = partition(g.data, g.labels, k=1)
        assert len(pd) == g.n_tiles


# ---------------------------------------------------------------------------
# CSR index structure
# ---------------------------------------------------------------------------

class TestCSRIndex:
    def test_core_node_indptr_shape(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        assert p.core_node_indptr.shape[0] == len(pd) + 1

    def test_overlap_node_indptr_shape(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        n_overlap = p.overlap_node_indptr.shape[0] - 1
        assert p.overlap_origins_indptr.shape[0] == n_overlap + 1

    def test_inter_edge_indptr_shape(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        n_inter = p.inter_chunk_pairs.shape[0]
        assert p.inter_edge_indptr.shape[0] == n_inter + 1

    def test_core_node_indptr_nondecreasing(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        diffs = pd._partition.core_node_indptr.diff()
        assert (diffs >= 0).all()

    def test_overlap_node_indptr_nondecreasing(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        diffs = pd._partition.overlap_node_indptr.diff()
        assert (diffs >= 0).all()

    def test_core_edge_indptr_nondecreasing(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        diffs = pd._partition.core_edge_indptr.diff()
        assert (diffs >= 0).all()

    def test_overlap_edge_indptr_nondecreasing(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        diffs = pd._partition.overlap_edge_indptr.diff()
        assert (diffs >= 0).all()

    def test_inter_edge_indptr_nondecreasing(self, k_zero):
        # k_zero guarantees at least one inter-chunk edge block
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        diffs = pd._partition.inter_edge_indptr.diff()
        assert (diffs >= 0).all()

    def test_node_ranges_cover_all_nodes_no_overlap(self, minimal_overlap):
        """Every node index 0..N-1 appears in exactly one chunk."""
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        N = pd._data.num_nodes
        n_core = len(pd)
        n_overlap = p.overlap_node_indptr.shape[0] - 1

        indices = []
        for i in range(n_core):
            lo = p.core_node_indptr[i].item()
            hi = p.core_node_indptr[i + 1].item()
            indices.extend(range(lo, hi))
        for o in range(n_overlap):
            lo = p.overlap_node_indptr[o].item()
            hi = p.overlap_node_indptr[o + 1].item()
            indices.extend(range(lo, hi))

        assert len(indices) == N
        assert len(set(indices)) == N

    def test_edge_ranges_cover_all_edges_no_overlap(self, minimal_overlap):
        """Every edge index 0..E-1 appears in exactly one block."""
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        E = pd._data.edge_index.shape[1]
        ranges = sorted(
            (lo, hi) for lo, hi in _all_edge_ranges(pd) if lo < hi
        )
        covered = 0
        for lo, hi in ranges:
            assert lo == covered
            covered = hi
        assert covered == E

    def test_k_zero_edge_ranges_cover_all(self, k_zero):
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        E = pd._data.edge_index.shape[1]
        ranges = sorted(
            (lo, hi) for lo, hi in _all_edge_ranges(pd) if lo < hi
        )
        covered = 0
        for lo, hi in ranges:
            assert lo == covered
            covered = hi
        assert covered == E

    def test_single_partition_no_inter_edges(self, single_partition):
        c = single_partition
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        assert p.inter_chunk_pairs.shape[0] == 0
        assert p.inter_edge_indptr.shape[0] == 1


# ---------------------------------------------------------------------------
# Edge reconstruction
# ---------------------------------------------------------------------------

class TestEdgeReconstruction:
    def test_exact_edge_set_preserved(self, minimal_overlap):
        """Permuted edge set matches original edge set by node ID."""
        c = minimal_overlap
        original_edges = _edge_id_set(
            c["data"].edge_index, c["data"].x
        )
        pd = partition(c["data"], c["labels"], c["k"])
        permuted_edges = _edge_id_set(pd._data.edge_index, pd._data.x)
        assert permuted_edges == original_edges

    def test_exact_edge_set_preserved_k_zero(self, k_zero):
        c = k_zero
        original_edges = _edge_id_set(
            c["data"].edge_index, c["data"].x
        )
        pd = partition(c["data"], c["labels"], c["k"])
        permuted_edges = _edge_id_set(pd._data.edge_index, pd._data.x)
        assert permuted_edges == original_edges

    def test_exact_edge_set_preserved_multi_hop(self, multi_hop):
        c = multi_hop
        original_edges = _edge_id_set(
            c["data"].edge_index, c["data"].x
        )
        pd = partition(c["data"], c["labels"], c["k"])
        permuted_edges = _edge_id_set(pd._data.edge_index, pd._data.x)
        assert permuted_edges == original_edges

    def test_spatial_edge_count_preserved(self, small_spatial_graph):
        g = small_spatial_graph
        e_before = g.data.edge_index.shape[1]
        pd = partition(g.data, g.labels, k=1)
        assert pd._data.edge_index.shape[1] == e_before


# ---------------------------------------------------------------------------
# Overlap structure
# ---------------------------------------------------------------------------

class TestOverlapStructure:
    def test_minimal_overlap_has_one_overlap_chunk(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        n_overlap = pd._partition.overlap_node_indptr.shape[0] - 1
        assert n_overlap == 1

    def test_minimal_overlap_directed_chunk_size(self, minimal_overlap):
        """Directed: only node 2 enters the overlap chunk."""
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 3:
            pytest.skip("directed only")
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_size = (
            p.overlap_node_indptr[1] - p.overlap_node_indptr[0]
        ).item()
        assert overlap_size == 1

    def test_minimal_overlap_undirected_chunk_size(self, minimal_overlap):
        """Undirected: nodes 1 and 2 both enter the overlap chunk."""
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 6:
            pytest.skip("undirected only")
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_size = (
            p.overlap_node_indptr[1] - p.overlap_node_indptr[0]
        ).item()
        assert overlap_size == 2

    def test_multi_hop_directed_overlap_size(self, multi_hop):
        """Directed k=2: nodes 2 and 3 enter the overlap chunk."""
        c = multi_hop
        if c["data"].edge_index.shape[1] != 4:
            pytest.skip("directed only")
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_size = (
            p.overlap_node_indptr[1] - p.overlap_node_indptr[0]
        ).item()
        assert overlap_size == 2

    def test_multi_hop_undirected_overlap_size(self, multi_hop):
        """Undirected k=2: nodes 1, 2, and 3 enter the overlap chunk."""
        c = multi_hop
        if c["data"].edge_index.shape[1] != 8:
            pytest.skip("undirected only")
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        overlap_size = (
            p.overlap_node_indptr[1] - p.overlap_node_indptr[0]
        ).item()
        assert overlap_size == 3

    def test_three_way_overlap_origins(self, three_way_overlap):
        c = three_way_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        n_overlap = p.overlap_node_indptr.shape[0] - 1
        assert n_overlap == 1
        assert set(p.overlap_origins_values.tolist()) == {0, 1, 2}
        assert p.overlap_origins_values.shape[0] == 3

    def test_k_zero_no_overlap_chunks(self, k_zero):
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        assert p.overlap_node_indptr.shape[0] == 1
        assert p.overlap_origins_values.shape[0] == 0

    def test_overlap_eliminates_core_warns(self, overlap_eliminates_core):
        c = overlap_eliminates_core
        with pytest.warns(UserWarning):
            partition(c["data"], c["labels"], c["k"])

    def test_overlap_eliminates_core_empty_chunk(
        self, overlap_eliminates_core
    ):
        c = overlap_eliminates_core
        with pytest.warns(UserWarning):
            pd = partition(c["data"], c["labels"], c["k"])
        p = pd._partition
        core1_size = (
            p.core_node_indptr[2] - p.core_node_indptr[1]
        ).item()
        assert core1_size == 0


# ---------------------------------------------------------------------------
# Feature permutation
# ---------------------------------------------------------------------------

class TestFeaturePermutation:
    def test_node_feature_set_preserved(self, minimal_overlap):
        """Every original node ID is present exactly once after permutation."""
        c = minimal_overlap
        original_ids = _node_id_set(c["data"].x)
        pd = partition(c["data"], c["labels"], c["k"])
        permuted_ids = _node_id_set(pd._data.x)
        assert permuted_ids == original_ids

    def test_edge_feature_set_preserved(self, with_edge_features):
        """Edge attr values are preserved as a multiset after permutation."""
        c = with_edge_features
        original_attr = sorted(
            c["data"].edge_attr.squeeze(1).tolist()
        )
        pd = partition(c["data"], c["labels"], c["k"])
        permuted_attr = sorted(
            pd._data.edge_attr.squeeze(1).tolist()
        )
        assert permuted_attr == original_attr

    def test_no_node_features_ok(self, no_node_features):
        c = no_node_features
        pd = partition(c["data"], c["labels"], c["k"])
        assert pd._data.num_nodes == 4


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_getitem_returns_data(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        assert isinstance(pd[0], Data)

    def test_chunk0_exact_node_ids(self, minimal_overlap):
        """chunk 0 retrieval returns exactly nodes {0, 1, 2} for both
        directed (core {0,1} + overlap {2}) and undirected
        (core {0} + overlap {1,2})."""
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[0].x) == {0, 1, 2}

    def test_chunk1_exact_node_ids_directed(self, minimal_overlap):
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 3:
            pytest.skip("directed only")
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[1].x) == {2, 3}

    def test_chunk1_exact_node_ids_undirected(self, minimal_overlap):
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 6:
            pytest.skip("undirected only")
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[1].x) == {1, 2, 3}

    def test_all_chunks_covers_all_nodes(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[[0, 1]].x) == {0, 1, 2, 3}

    def test_chunk0_exact_edges_directed(self, minimal_overlap):
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 3:
            pytest.skip("directed only")
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        assert _edge_id_set(sub.edge_index, sub.x) == {(0, 1), (1, 2)}

    def test_chunk0_exact_edges_undirected(self, minimal_overlap):
        c = minimal_overlap
        if c["data"].edge_index.shape[1] != 6:
            pytest.skip("undirected only")
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        assert _edge_id_set(sub.edge_index, sub.x) == {
            (0, 1), (1, 0), (1, 2), (2, 1),
        }

    def test_all_chunks_exact_edge_set(self, minimal_overlap):
        """Retrieving all chunks returns the exact original edge set."""
        c = minimal_overlap
        original_edges = _edge_id_set(c["data"].edge_index, c["data"].x)
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[[0, 1]]
        assert _edge_id_set(sub.edge_index, sub.x) == original_edges

    def test_edge_indices_within_bounds(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        assert int(sub.edge_index.max()) < sub.num_nodes

    def test_edge_indices_within_bounds_multi(self, minimal_overlap):
        c = minimal_overlap
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[[0, 1]]
        assert int(sub.edge_index.max()) < sub.num_nodes

    def test_k_zero_chunk0_exact_node_ids(self, k_zero):
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[0].x) == {0, 1}

    def test_k_zero_chunk0_exact_edges(self, k_zero):
        c = k_zero
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        assert _edge_id_set(sub.edge_index, sub.x) == {(0, 1)}

    def test_single_partition_retrieval_all_nodes(self, single_partition):
        c = single_partition
        pd = partition(c["data"], c["labels"], c["k"])
        assert _node_id_set(pd[0].x) == {0, 1, 2, 3}

    def test_single_partition_retrieval_all_edges(self, single_partition):
        c = single_partition
        pd = partition(c["data"], c["labels"], c["k"])
        sub = pd[0]
        assert _edge_id_set(sub.edge_index, sub.x) == {
            (0, 1), (1, 2), (2, 3)
        }

    def test_spatial_single_chunk_edge_bounds(self, small_spatial_graph):
        g = small_spatial_graph
        pd = partition(g.data, g.labels, k=1)
        sub = pd[0]
        assert int(sub.edge_index.max()) < sub.num_nodes

    def test_len(self, minimal_overlap):
        c = minimal_overlap
        n_partitions = c["labels"].unique().shape[0]
        pd = partition(c["data"], c["labels"], c["k"])
        assert len(pd) == n_partitions
