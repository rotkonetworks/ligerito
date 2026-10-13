//! Known-answer vectors from an independent implementation.
//!
//! `test-vectors/bolt_rs_vectors.json` was produced by bcc-research/bolt-rs
//! (itself cross-verified against the authors' Julia reference): GF(2^32) and
//! GF(2^128) products and non-systematic Reed-Solomon codewords. Both
//! libraries use the same moduli and RS basis, so the values must agree.

use ligerito_binary_fields::{BinaryElem128, BinaryElem32, BinaryFieldElement};
use ligerito_reed_solomon::{encode_non_systematic, reed_solomon};

const VECTORS: &str = include_str!("../../test-vectors/bolt_rs_vectors.json");

/// Hex strings of the JSON value under `key`, searched after `section`
/// (flat arrays, nested arrays of triples, or a single string).
fn hex(section: &str, key: &str) -> Vec<u128> {
    let base = VECTORS.find(&format!("\"{section}\"")).expect("section");
    let rest = &VECTORS[base..];
    let at = rest.find(&format!("\"{key}\":")).expect("key") + key.len() + 3;
    let val = rest[at..].trim_start();
    let end = if val.starts_with('[') {
        let mut depth = 0;
        val.char_indices()
            .find(|&(_, c)| {
                depth += (c == '[') as i32 - (c == ']') as i32;
                depth == 0
            })
            .unwrap()
            .0
    } else {
        val[1..].find('"').unwrap() + 1
    };
    val[..=end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(|h| u128::from_str_radix(h, 16).unwrap())
        .collect()
}

fn g32(v: u128) -> BinaryElem32 {
    BinaryElem32::from(v as u32)
}
fn g128(v: u128) -> BinaryElem128 {
    BinaryElem128::from(v)
}

#[test]
fn gf32_mul_matches_reference() {
    let v = hex("gf32_mul", "gf32_mul");
    assert_eq!(v.len(), 64 * 3);
    for t in v.chunks_exact(3) {
        assert_eq!(g32(t[0]).mul(&g32(t[1])), g32(t[2]));
    }
}

#[test]
fn gf128_mul_matches_reference() {
    let v = hex("gf128_mul", "gf128_mul");
    assert_eq!(v.len(), 64 * 3);
    for t in v.chunks_exact(3) {
        assert_eq!(g128(t[0]).mul(&g128(t[1])), g128(t[2]));
    }
}

#[test]
fn rs_gf32_matches_reference() {
    let (msg, cw) = (hex("rs32_m16", "message"), hex("rs32_m16", "codeword"));
    assert_eq!((msg.len(), cw.len()), (16, 64));
    let rs = reed_solomon::<BinaryElem32>(16, 64);
    let mut data: Vec<BinaryElem32> = msg.iter().map(|&x| g32(x)).collect();
    data.resize(64, BinaryElem32::zero());
    encode_non_systematic(&rs, &mut data);
    assert_eq!(data, cw.iter().map(|&x| g32(x)).collect::<Vec<_>>());
}

#[test]
fn rs_gf128_matches_reference() {
    let (msg, cw) = (hex("rs128_m8", "message"), hex("rs128_m8", "codeword"));
    assert_eq!((msg.len(), cw.len()), (8, 32));
    let rs = reed_solomon::<BinaryElem128>(8, 32);
    let mut data: Vec<BinaryElem128> = msg.iter().map(|&x| g128(x)).collect();
    data.resize(32, BinaryElem128::zero());
    encode_non_systematic(&rs, &mut data);
    assert_eq!(data, cw.iter().map(|&x| g128(x)).collect::<Vec<_>>());
}

/// Our GF(2^32) → GF(2^128) embedding sends x to a root γ of the GF(2^32)
/// modulus; the reference uses root β. Both are field embeddings iff they
/// are Frobenius conjugates (β = γ^(2^i)); check that, so the two
/// implementations provably agree up to the choice of root.
#[test]
fn embedding_root_is_conjugate_of_reference() {
    let beta = g128(hex("embed_root_beta32", "embed_root_beta32")[0]);
    let gamma = BinaryElem128::from(BinaryElem32::from(2u32)); // image of x
    let mut c = gamma;
    let mut found = None;
    for i in 0..32 {
        if c == beta {
            found = Some(i);
            break;
        }
        c = c.mul(&c);
    }
    assert!(found.is_some(), "reference root is not a conjugate of ours");
}
