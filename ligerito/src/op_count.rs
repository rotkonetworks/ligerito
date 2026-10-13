//! Operation counters (feature `op-count`): field multiplications and
//! inversions, row hashes, and Merkle leaf/node hashes, for auditing what a
//! verifier costs (e.g. an on-chain budget).

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(crate) static ROW_HASHES: AtomicU64 = AtomicU64::new(0);

/// A snapshot of every counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Costs {
    pub field: binary_fields::op_count::FieldOps,
    pub row_hashes: u64,
    pub merkle_leaf_hashes: u64,
    pub merkle_node_hashes: u64,
}

pub fn snapshot() -> Costs {
    let (leaf, node) = merkle_tree::op_count::snapshot();
    Costs {
        field: binary_fields::op_count::snapshot(),
        row_hashes: ROW_HASHES.load(Relaxed),
        merkle_leaf_hashes: leaf,
        merkle_node_hashes: node,
    }
}

pub fn reset() {
    binary_fields::op_count::reset();
    merkle_tree::op_count::reset();
    ROW_HASHES.store(0, Relaxed);
}

#[inline(always)]
pub(crate) fn bump_row_hash() {
    ROW_HASHES.fetch_add(1, Relaxed);
}
