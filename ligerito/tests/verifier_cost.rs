//! What one `verify` costs, by operation (run with `--features op-count`).
//!
//! Prints a table for the shipped configs and fails if verifier cost stops
//! being sublinear in the committed size (a sign it lost its factored basis
//! and is materialising length-m₀ vectors again).

use ligerito::{op_count, prove_sha256, verify_sha256};
use ligerito_binary_fields::{BinaryElem128 as U, BinaryElem32 as T};

#[test]
fn verifier_costs() {
    let mut mul128 = Vec::new();
    println!("config  mul128      mul32     inv128  row-hash  merkle-leaf  merkle-node");
    for log in [12u32, 20, 24] {
        let (pc, vc) = if log == 12 {
            (
                ligerito::hardcoded_config_12(std::marker::PhantomData::<T>, std::marker::PhantomData::<U>),
                ligerito::hardcoded_config_12_verifier(),
            )
        } else {
            (
                ligerito::prover_config_for_log_size::<T, U>(log),
                ligerito::verifier_config_for_log_size(log),
            )
        };
        let poly: Vec<T> = (0..1u32 << log).map(|i| T::from(i.wrapping_mul(2654435761))).collect();
        let proof = prove_sha256(&pc, &poly).unwrap();
        op_count::reset();
        assert!(verify_sha256(&vc, &proof).unwrap());
        let c = op_count::snapshot();
        println!(
            "2^{log:<5} {:<11} {:<9} {:<7} {:<9} {:<12} {}",
            c.field.mul128, c.field.mul32, c.field.inv128, c.row_hashes, c.merkle_leaf_hashes, c.merkle_node_hashes
        );
        assert_eq!(c.field.inv128, 0, "verifier gained an inversion");
        mul128.push(c.field.mul128);
    }
    // 2^20 -> 2^24 is 16x the data; the verifier must grow far slower
    assert!(mul128[2] < 4 * mul128[1], "verifier mul128 not sublinear: {mul128:?}");
}
