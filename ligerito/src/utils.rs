//! Utility functions for Ligerito - FINAL FIXED VERSION

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use binary_fields::BinaryFieldElement;

/// Evaluate Lagrange basis at given points
pub fn evaluate_lagrange_basis<F: BinaryFieldElement>(rs: &[F]) -> Vec<F> {
    if rs.is_empty() {
        return vec![F::one()];
    }

    let one = F::one();
    let mut current_layer = vec![one.add(&rs[0]), rs[0]];
    let mut len = 2;

    for i in 1..rs.len() {
        let mut next_layer = Vec::with_capacity(2 * len);
        let ri_plus_one = one.add(&rs[i]);

        for j in 0..len {
            next_layer.push(current_layer[j].mul(&ri_plus_one));
            next_layer.push(current_layer[j].mul(&rs[i]));
        }

        current_layer = next_layer;
        len *= 2;
    }

    // debug check
    debug_assert!(
        !current_layer.iter().all(|&x| x == F::zero()),
        "Lagrange basis should not be all zeros"
    );

    current_layer
}

/// Evaluate s_k at v_k values (for sumcheck)
/// Returns evaluation of all s_k polynomials at v_k points
pub fn eval_sk_at_vks<F: BinaryFieldElement>(n: usize) -> Vec<F> {
    assert!(n.is_power_of_two());
    let num_subspaces = n.trailing_zeros() as usize;

    let mut sks_vks = vec![F::zero(); num_subspaces + 1];
    sks_vks[0] = F::one(); // s_0(v_0) = 1

    // Initialize with powers of 2: 2^1, 2^2, ..., 2^num_subspaces
    let mut layer: Vec<F> = (1..=num_subspaces)
        .map(|i| F::from_bits(1u64 << i))
        .collect();

    let mut cur_len = num_subspaces;

    for i in 0..num_subspaces {
        for j in 0..cur_len {
            let sk_at_vk = if j == 0 {
                // s_{i+1}(v_{i+1}) computation
                let val = layer[0].mul(&layer[0]).add(&sks_vks[i].mul(&layer[0]));
                sks_vks[i + 1] = val;
                val
            } else {
                layer[j].mul(&layer[j]).add(&sks_vks[i].mul(&layer[j]))
            };

            if j > 0 {
                layer[j - 1] = sk_at_vk;
            }
        }
        cur_len -= 1;
    }

    sks_vks
}

/// Row `qf` of the non-systematic Reed-Solomon generator matrix, scaled.
///
/// Writes `basis[i] = scale · X_i(qf)` where `X_i(x) = Π_{j : bit j of i} s_j(x)`
/// is the unnormalised novel polynomial basis and `s_j` the subspace vanishing
/// polynomials. With `qf = from_bits(q)` this is exactly
/// `encode_non_systematic(e_i)[q]`, so for a message `v`:
/// `Enc(v)[q] = Σ_i v[i] · basis[i]` (with `scale = 1`).
///
/// `sks_x` is scratch (length ≥ log₂ |basis|) and receives `s_j(qf)`;
/// `sks_vks` comes from `eval_sk_at_vks(|basis|)`.
pub fn evaluate_scaled_basis_inplace<F: BinaryFieldElement, U: BinaryFieldElement>(
    sks_x: &mut [F],
    basis: &mut [U],
    sks_vks: &[F],
    qf: F,
    scale: U,
) where
    U: From<F>,
{
    let n = basis.len();
    assert!(n.is_power_of_two(), "basis length must be a power of two");
    let k = n.trailing_zeros() as usize;
    assert!(sks_x.len() >= k && sks_vks.len() >= k);

    if k > 0 {
        sks_x[0] = qf;
        for j in 1..k {
            let s_prev = sks_x[j - 1];
            sks_x[j] = s_prev.mul(&s_prev).add(&sks_vks[j - 1].mul(&s_prev));
        }
    }

    // tensor product: bit j of the index selects the factor s_j(qf)
    basis[0] = scale;
    for j in 0..k {
        let half = 1 << j;
        let s = U::from(sks_x[j]);
        for i in 0..half {
            basis[half + i] = basis[i].mul(&s);
        }
    }
}

/// Check if a number is a power of 2
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Encode non-systematic Reed-Solomon (prover only)
#[cfg(feature = "prover")]
pub fn encode_non_systematic<F: BinaryFieldElement + 'static>(
    rs: &reed_solomon::ReedSolomon<F>,
    data: &mut [F],
) {
    // Non-systematic encoding (no original message preservation)
    reed_solomon::encode_in_place(rs, data);
}

/// Multilinear polynomial partial evaluation
pub fn partial_eval_multilinear<F: BinaryFieldElement>(poly: &mut Vec<F>, evals: &[F]) {
    let mut n = poly.len();

    for &e in evals {
        n /= 2;

        for i in 0..n {
            let p0 = poly[2 * i];
            let p1 = poly[2 * i + 1];
            poly[i] = p0.add(&e.mul(&p1.add(&p0)));
        }
    }

    poly.truncate(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use binary_fields::{BinaryElem128, BinaryElem16, BinaryElem32};

    #[test]
    fn test_lagrange_basis() {
        let rs = vec![
            BinaryElem16::from_bits(0x1234),
            BinaryElem16::from_bits(0x5678),
            BinaryElem16::from_bits(0x9ABC),
        ];

        let basis = evaluate_lagrange_basis(&rs);
        assert_eq!(basis.len(), 8); // 2^3
    }

    #[test]
    fn test_lagrange_basis_all_ones() {
        // Test with all ones
        let rs = vec![
            BinaryElem32::one(),
            BinaryElem32::one(),
            BinaryElem32::one(),
            BinaryElem32::one(),
        ];

        let basis = evaluate_lagrange_basis(&rs);
        assert_eq!(basis.len(), 16); // 2^4

        // When all rs[i] = 1, then 1 + rs[i] = 0 in binary fields
        // So most entries should be zero
        let non_zero_count = basis.iter().filter(|&&x| x != BinaryElem32::zero()).count();
        println!("Non-zero entries: {}/{}", non_zero_count, basis.len());
    }

    #[test]
    fn test_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(1023));
    }

    #[test]
    fn test_sk_evaluation() {
        // Test for n = 16
        let sks_vks = eval_sk_at_vks::<BinaryElem32>(16);
        assert_eq!(sks_vks.len(), 5); // log2(16) + 1
        assert_eq!(sks_vks[0], BinaryElem32::one()); // s_0(v_0) = 1

        // Test for n = 8
        let sks_vks = eval_sk_at_vks::<BinaryElem16>(8);
        assert_eq!(sks_vks.len(), 4); // log2(8) + 1
        assert_eq!(sks_vks[0], BinaryElem16::one());
    }

    #[test]
    fn test_partial_eval() {
        let mut poly = vec![
            BinaryElem32::from_bits(1),
            BinaryElem32::from_bits(2),
            BinaryElem32::from_bits(3),
            BinaryElem32::from_bits(4),
            BinaryElem32::from_bits(5),
            BinaryElem32::from_bits(6),
            BinaryElem32::from_bits(7),
            BinaryElem32::from_bits(8),
        ];

        let original_len = poly.len();
        let evals = vec![BinaryElem32::from_bits(2)];

        partial_eval_multilinear(&mut poly, &evals);

        // Should halve the size
        assert_eq!(poly.len(), original_len / 2);
    }

    /// The basis row must be the generator row of `encode_non_systematic`
    /// at every domain point, including those past the message length.
    #[cfg(feature = "prover")]
    #[test]
    fn test_scaled_basis_is_generator_row() {
        fn check<F: BinaryFieldElement + 'static>(m: usize) {
            let n = 4 * m;
            let rs = reed_solomon::reed_solomon::<F>(m, n);
            let sks_vks = eval_sk_at_vks::<F>(m);
            let cols: Vec<Vec<F>> = (0..m)
                .map(|i| {
                    let mut d = vec![F::zero(); n];
                    d[i] = F::one();
                    reed_solomon::encode_non_systematic(&rs, &mut d);
                    d
                })
                .collect();
            let mut sks_x = vec![F::zero(); sks_vks.len()];
            let mut basis = vec![F::zero(); m];
            for q in 0..n {
                let qf = F::from_bits(q as u64);
                evaluate_scaled_basis_inplace(&mut sks_x, &mut basis, &sks_vks, qf, F::one());
                for i in 0..m {
                    assert_eq!(basis[i], cols[i][q], "m={m} q={q} i={i}");
                }
            }
        }
        check::<BinaryElem32>(16);
        check::<BinaryElem32>(64);
        check::<BinaryElem128>(32);
    }
}

/// Hash a row for Merkle tree commitment
/// Used by both prover and verifier
pub fn hash_row<F: BinaryFieldElement>(row: &[F]) -> merkle_tree::Hash {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();

    // Hash row length for domain separation
    hasher.update((row.len() as u32).to_le_bytes());

    // Hash each element
    let elem_size = core::mem::size_of::<F>();
    for elem in row.iter() {
        let bytes =
            unsafe { core::slice::from_raw_parts(elem as *const F as *const u8, elem_size) };
        hasher.update(bytes);
    }

    hasher.finalize().into()
}
