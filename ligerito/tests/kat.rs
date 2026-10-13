//! Known-answer tests: transcript challenges, a Merkle root and a digest of a
//! complete deterministic proof, pinned in `test-vectors/ligerito_kat.txt`.
//!
//! Any change to these values is a proof-format change. To regenerate after
//! an intentional one: `LIGERITO_REGEN_KAT=1 cargo test --test kat`.

use ligerito::{
    hardcoded_config_12, hardcoded_config_12_verifier, prove_sha256, utils::hash_row,
    verify_sha256, FiatShamir, Transcript,
};
use ligerito_binary_fields::{BinaryElem128 as U, BinaryElem32 as T, BinaryFieldElement};
use sha2::{Digest, Sha256};
use std::{fmt::Write, marker::PhantomData};

const KAT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-vectors/ligerito_kat.txt");

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn elem_bytes<F: BinaryFieldElement>(x: &F) -> Vec<u8> {
    // field elements are plain little-endian integers in memory
    unsafe { core::slice::from_raw_parts(x as *const F as *const u8, core::mem::size_of::<F>()) }
        .to_vec()
}

fn transcript_lines(name: &str, mut fs: impl Transcript, out: &mut String) {
    fs.absorb_bytes(b"kat", b"ligerito known-answer test");
    fs.absorb_elem(U::from(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128));
    fs.absorb_elems(&[T::from(1u32), T::from(0xdead_beefu32)]);
    let c: Vec<U> = (0..3).map(|_| fs.get_challenge()).collect();
    let t: T = fs.get_challenge();
    let q = fs.get_distinct_queries(1024, 8);
    let cs: Vec<String> = c.iter().map(|x| hex(&elem_bytes(x))).collect();
    writeln!(out, "transcript.{name}.challenges_u128 = {}", cs.join(",")).unwrap();
    writeln!(out, "transcript.{name}.challenge_u32 = {}", hex(&elem_bytes(&t))).unwrap();
    writeln!(out, "transcript.{name}.queries = {q:?}").unwrap();
}

fn compute() -> String {
    let mut out = String::new();
    transcript_lines("sha256", FiatShamir::new_sha256(0), &mut out);
    #[cfg(feature = "transcript-merlin")]
    transcript_lines("merlin", FiatShamir::new_merlin(), &mut out);
    #[cfg(feature = "transcript-blake2b")]
    transcript_lines("blake2b", FiatShamir::new_blake2b(), &mut out);

    let rows: Vec<Vec<T>> = (0..64u32)
        .map(|i| (0..16u32).map(|j| T::from(i.wrapping_mul(2654435761) ^ j)).collect())
        .collect();
    let leaves: Vec<_> = rows.iter().map(|r| hash_row(r)).collect();
    let root = ligerito_merkle::build_merkle_tree(&leaves).get_root();
    writeln!(out, "merkle.root_64_rows = {}", hex(&root.root.unwrap())).unwrap();

    let pc = hardcoded_config_12(PhantomData::<T>, PhantomData::<U>);
    let poly: Vec<T> = (0..1u32 << 12).map(|i| T::from(i.wrapping_mul(0x9E37_79B9) ^ 0x5a5a)).collect();
    let proof = prove_sha256(&pc, &poly).unwrap();
    assert!(verify_sha256(&hardcoded_config_12_verifier(), &proof).unwrap());
    let mut h = Sha256::new();
    h.update(proof.initial_ligero_cm.root.root.unwrap());
    for c in &proof.recursive_commitments {
        h.update(c.root.root.unwrap());
    }
    for &(a, b, c) in &proof.sumcheck_transcript.transcript {
        for x in [a, b, c] {
            h.update(elem_bytes(&x));
        }
    }
    for x in &proof.final_ligero_proof.yr {
        h.update(elem_bytes(x));
    }
    for row in &proof.final_ligero_proof.opened_rows {
        for x in row {
            h.update(elem_bytes(x));
        }
    }
    writeln!(out, "proof.config12_sha256.digest = {}", hex(&h.finalize())).unwrap();
    writeln!(out, "proof.config12_sha256.size_bytes = {}", proof.size_of()).unwrap();
    out
}

#[test]
fn known_answers() {
    let got = compute();
    if std::env::var("LIGERITO_REGEN_KAT").is_ok() {
        std::fs::write(KAT_PATH, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(KAT_PATH).expect("missing KAT file; run with LIGERITO_REGEN_KAT=1");
    for line in got.lines() {
        let key = line.split(" = ").next().unwrap();
        // lines for transcripts not compiled into this build are skipped
        let expected = want.lines().find(|l| l.starts_with(&format!("{key} = ")));
        assert_eq!(Some(line), expected, "known-answer mismatch for {key}");
    }
}

/// Every transcript must produce full-width GF(2^128) challenges.
#[test]
fn challenges_use_all_128_bits() {
    fn check(name: &str, mut fs: impl Transcript) {
        let (mut or_hi, mut or_lo) = (0u64, 0u64);
        for _ in 0..64 {
            let c: U = fs.get_challenge();
            let b = elem_bytes(&c);
            or_lo |= u64::from_le_bytes(b[..8].try_into().unwrap());
            or_hi |= u64::from_le_bytes(b[8..].try_into().unwrap());
        }
        assert_eq!((or_lo, or_hi), (u64::MAX, u64::MAX), "{name}: challenge bits stuck at zero");
    }
    check("sha256", FiatShamir::new_sha256(0));
    #[cfg(feature = "transcript-merlin")]
    check("merlin", FiatShamir::new_merlin());
    #[cfg(feature = "transcript-blake2b")]
    check("blake2b", FiatShamir::new_blake2b());
}
