//! SIMD paths must agree with plain shift-and-add reference arithmetic.
//! Run with `RUSTFLAGS="-C target-cpu=native"` to exercise the
//! PCLMULQDQ / VPCLMULQDQ code.

use ligerito_binary_fields::{BinaryElem128, BinaryElem32, BinaryFieldElement};

fn ref_mul(mut a: u128, mut b: u128, bits: u32, irr_low: u128) -> u128 {
    let top = 1u128 << (bits - 1);
    let mask = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
    let mut r = 0u128;
    while b != 0 {
        if b & 1 == 1 {
            r ^= a;
        }
        b >>= 1;
        let carry = a & top != 0;
        a = (a << 1) & mask;
        if carry {
            a ^= irr_low;
        }
    }
    r
}

fn rng() -> impl FnMut() -> u128 {
    let mut x = 0x1234_5678_9abc_def1u64;
    move || {
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        ((next() as u128) << 64) | next() as u128
    }
}

#[test]
fn field_mul_matches_reference() {
    let mut r = rng();
    for _ in 0..2000 {
        let (a, b) = (r(), r());
        let got = BinaryElem128::from(a).mul(&BinaryElem128::from(b));
        assert_eq!(got, BinaryElem128::from(ref_mul(a, b, 128, 0x87)));

        let (a, b) = (a as u32, b as u32);
        let irr32 = 0b11001 | (1 << 7) | (1 << 9) | (1 << 15);
        let got = BinaryElem32::from(a).mul(&BinaryElem32::from(b));
        assert_eq!(got, BinaryElem32::from(ref_mul(a as u128, b as u128, 32, irr32) as u32));
    }
}

#[cfg(all(feature = "hardware-accel", target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[test]
fn fft_butterfly_tiers_match_definition() {
    use ligerito_binary_fields::simd;
    type F = BinaryElem32;
    let mut r = rng();
    let has_avx2 = is_x86_feature_detected!("vpclmulqdq") && is_x86_feature_detected!("avx2");
    let has_avx512 = has_avx2 && is_x86_feature_detected!("avx512f");
    let mut tiers: Vec<(&str, fn(&mut [F], &mut [F], F))> = vec![
        ("dispatch", simd::fft_butterfly_gf32),
        ("sse", simd::fft_butterfly_gf32_sse),
        ("scalar", simd::fft_butterfly_gf32_scalar),
    ];
    if has_avx2 {
        tiers.push(("avx2", simd::fft_butterfly_gf32_avx2_only));
    }
    if has_avx512 {
        tiers.push(("avx512", simd::fft_butterfly_gf32_avx512_only));
    }
    for n in [1usize, 2, 3, 4, 7, 8, 9, 16, 33, 64, 257] {
        let u: Vec<F> = (0..n).map(|_| F::from(r() as u32)).collect();
        let w: Vec<F> = (0..n).map(|_| F::from(r() as u32)).collect();
        let l = F::from(r() as u32);
        let eu: Vec<F> = u.iter().zip(&w).map(|(a, b)| a.add(&l.mul(b))).collect();
        let ew: Vec<F> = w.iter().zip(&eu).map(|(b, a)| b.add(a)).collect();
        for (name, f) in &tiers {
            let (mut a, mut b) = (u.clone(), w.clone());
            f(&mut a, &mut b, l);
            assert!(a == eu && b == ew, "tier {name} wrong at n={n}");
        }
    }
}
