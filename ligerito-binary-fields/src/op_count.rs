//! Global field-operation counters (feature `op-count`).
//!
//! Relaxed atomics, so counts are exact but the feature costs a little speed;
//! intended for measuring verifier budgets, not production builds.

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(crate) static MUL16: AtomicU64 = AtomicU64::new(0);
pub(crate) static MUL32: AtomicU64 = AtomicU64::new(0);
pub(crate) static MUL128: AtomicU64 = AtomicU64::new(0);
pub(crate) static INV32: AtomicU64 = AtomicU64::new(0);
pub(crate) static INV128: AtomicU64 = AtomicU64::new(0);

/// A snapshot of the counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldOps {
    pub mul16: u64,
    pub mul32: u64,
    pub mul128: u64,
    pub inv32: u64,
    pub inv128: u64,
}

pub fn snapshot() -> FieldOps {
    FieldOps {
        mul16: MUL16.load(Relaxed),
        mul32: MUL32.load(Relaxed),
        mul128: MUL128.load(Relaxed),
        inv32: INV32.load(Relaxed),
        inv128: INV128.load(Relaxed),
    }
}

pub fn reset() {
    for c in [&MUL16, &MUL32, &MUL128, &INV32, &INV128] {
        c.store(0, Relaxed);
    }
}

#[inline(always)]
pub(crate) fn bump(c: &AtomicU64) {
    c.fetch_add(1, Relaxed);
}

#[inline(always)]
pub(crate) fn bump_mul(bits: u32) {
    match bits {
        16 => bump(&MUL16),
        32 => bump(&MUL32),
        _ => {}
    }
}

#[inline(always)]
pub(crate) fn bump_inv(bits: u32) {
    if bits == 32 {
        bump(&INV32);
    }
}
