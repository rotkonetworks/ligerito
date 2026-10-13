//! Every shipped prover config validates and derives exactly the matching
//! hand-written / autosized verifier config.

use ligerito::{autosizer, configs::*, ProverConfig, VerifierConfig};
use ligerito_binary_fields::{BinaryElem128 as U, BinaryElem32 as T};
use std::marker::PhantomData;

/// Sizes whose encoded matrix (2^(n+2) elements) a 32-bit target cannot address.
fn fits_target(v: &VerifierConfig) -> bool {
    v.poly_log_size() + 2 < usize::BITS as usize
}

fn same(name: &str, p: &ProverConfig<T, U>, v: &VerifierConfig) {
    if !fits_target(v) {
        assert!(p.validate().is_err(), "{name}: oversized config must be rejected");
        return;
    }
    p.validate().unwrap_or_else(|e| panic!("{name}: validate failed: {e:?}"));
    let d = p.verifier_config();
    assert!(v.is_well_formed(), "{name}: verifier config malformed");
    assert_eq!(
        (d.recursive_steps, d.initial_dim, &d.log_dims, d.initial_k, &d.ks, d.num_queries),
        (v.recursive_steps, v.initial_dim, &v.log_dims, v.initial_k, &v.ks, v.num_queries),
        "{name}: derived verifier config differs"
    );
}

#[test]
fn hardcoded_configs_consistent() {
    let (t, u) = (PhantomData::<T>, PhantomData::<U>);
    same("12", &hardcoded_config_12(t, u), &hardcoded_config_12_verifier());
    same("16", &hardcoded_config_16(t, u), &hardcoded_config_16_verifier());
    same("20", &hardcoded_config_20(t, u), &hardcoded_config_20_verifier());
    same("20_k8", &hardcoded_config_20_k8(t, u), &hardcoded_config_20_k8_verifier());
    same("20_k10", &hardcoded_config_20_k10(t, u), &hardcoded_config_20_k10_verifier());
    same("24", &hardcoded_config_24(t, u), &hardcoded_config_24_verifier());
    same("26", &hardcoded_config_26(t, u), &hardcoded_config_26_verifier());
    same("28", &hardcoded_config_28(t, u), &hardcoded_config_28_verifier());
    same("30", &hardcoded_config_30(t, u), &hardcoded_config_30_verifier());
}

#[test]
fn autosized_configs_consistent() {
    for log in autosizer::MIN_LOG_SIZE..=autosizer::MAX_LOG_SIZE {
        same(
            &format!("auto {log}"),
            &autosizer::prover_config_for_log_size::<T, U>(log),
            &autosizer::verifier_config_for_log_size(log),
        );
    }
}

#[test]
fn autosized_roundtrip() {
    for log in [20u32, 22] {
        let pc = autosizer::prover_config_for_log_size::<T, U>(log);
        let vc = autosizer::verifier_config_for_log_size(log);
        let poly: Vec<T> = (0..1u32 << log).map(|i| T::from(i.wrapping_mul(2654435761))).collect();
        let proof = ligerito::prove_sha256(&pc, &poly).unwrap();
        assert!(ligerito::verify_sha256(&vc, &proof).unwrap(), "log {log}");
    }
}
