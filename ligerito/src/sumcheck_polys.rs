#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::utils::evaluate_scaled_basis_inplace;
use binary_fields::BinaryFieldElement;

/// `⟨row, eq(challenges, ·)⟩`, binding the lsb of the column index first.
///
/// Equals `partial_eval_multilinear(row, challenges)[0]`, which is how the
/// prover folds the committed polynomial (see `ligero::poly2mat`).
pub fn tensorized_dot_product<T, U>(row: &[T], challenges: &[U]) -> U
where
    T: BinaryFieldElement,
    U: BinaryFieldElement + From<T>,
{
    assert_eq!(row.len(), 1 << challenges.len(), "row length must be 2^k");

    let mut current: Vec<U> = row.iter().map(|&x| U::from(x)).collect();
    for &r in challenges {
        let half = current.len() / 2;
        for i in 0..half {
            let (lo, hi) = (current[2 * i], current[2 * i + 1]);
            current[i] = lo.add(&r.mul(&hi.add(&lo)));
        }
        current.truncate(half);
    }
    current[0]
}

/// Round message of the product sumcheck `Σ_x f(x)·g(x)` on the lsb variable:
/// `(g(0), g(1), coefficient of X²)` of the univariate round polynomial.
pub fn product_round_coeffs<F: BinaryFieldElement>(f: &[F], g: &[F]) -> (F, F, F) {
    debug_assert_eq!(f.len(), g.len());
    let (mut s0, mut s1, mut s2) = (F::zero(), F::zero(), F::zero());
    for (fp, gp) in f.chunks_exact(2).zip(g.chunks_exact(2)) {
        s0 = s0.add(&fp[0].mul(&gp[0]));
        s1 = s1.add(&fp[1].mul(&gp[1]));
        s2 = s2.add(&fp[1].add(&fp[0]).mul(&gp[1].add(&gp[0])));
    }
    (s0, s1, s2)
}

/// Evaluate a round polynomial given as `(g(0), g(1), X² coefficient)` at `r`.
pub fn eval_round_poly<F: BinaryFieldElement>(coeffs: (F, F, F), r: F) -> F {
    let (s0, s1, s2) = coeffs;
    // g(X) = s0 + (s0 + s1 + s2)·X + s2·X²   (char 2)
    let lin = s0.add(&s1).add(&s2);
    s0.add(&lin.mul(&r)).add(&s2.mul(&r.mul(&r)))
}

/// Bind the lsb variable of a multilinear table to `r`.
pub fn fold_lsb<F: BinaryFieldElement>(poly: &[F], r: F) -> Vec<F> {
    poly.chunks_exact(2)
        .map(|p| p[0].add(&r.mul(&p[1].add(&p[0]))))
        .collect()
}

/// `Σ_i a[i]·b[i]`
pub fn inner_product<F: BinaryFieldElement>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .fold(F::zero(), |acc, (x, y)| acc.add(&x.mul(y)))
}

/// precompute powers of alpha to avoid repeated multiplications
pub fn precompute_alpha_powers<F: BinaryFieldElement>(alpha: F, n: usize) -> Vec<F> {
    let mut alpha_pows = vec![F::zero(); n];
    if n > 0 {
        alpha_pows[0] = F::one();
        for i in 1..n {
            alpha_pows[i] = alpha_pows[i - 1].mul(&alpha);
        }
    }
    alpha_pows
}

/// Batch the Ligero row checks at `sorted_queries` into one inner-product claim
/// (paper section 6.2).
///
/// Returns `(basis, enforced_sum)` with `basis = Σ_k α^k · G[q_k]` (generator
/// rows of the length-`2^n` message code) and
/// `enforced_sum = Σ_k α^k · ⟨row_k, eq(v_challenges)⟩`. For an honest
/// commitment, `⟨partial_eval(v, v_challenges), basis⟩ = enforced_sum`.
pub fn induce_sumcheck_poly<T, U>(
    n: usize,
    sks_vks: &[T],
    opened_rows: &[Vec<T>],
    v_challenges: &[U],
    sorted_queries: &[usize],
    alpha: U,
) -> (Vec<U>, U)
where
    T: BinaryFieldElement,
    U: BinaryFieldElement + From<T>,
{
    let mut basis_poly = vec![U::zero(); 1 << n];
    let mut enforced_sum = U::zero();
    let alpha_pows = precompute_alpha_powers(alpha, opened_rows.len());

    // reuse allocations across iterations
    let mut local_sks_x = vec![T::zero(); sks_vks.len()];
    let mut local_basis = vec![U::zero(); 1 << n];

    for (i, (row, &query)) in opened_rows.iter().zip(sorted_queries.iter()).enumerate() {
        let dot = tensorized_dot_product(row, v_challenges);
        let contribution = dot.mul(&alpha_pows[i]);
        enforced_sum = enforced_sum.add(&contribution);

        let qf = T::from_bits(query as u64);

        // compute scaled basis (clears buffers internally)
        evaluate_scaled_basis_inplace(
            &mut local_sks_x,
            &mut local_basis,
            sks_vks,
            qf,
            alpha_pows[i],
        );

        for (j, &val) in local_basis.iter().enumerate() {
            basis_poly[j] = basis_poly[j].add(&val);
        }
    }

    (basis_poly, enforced_sum)
}

/// parallel version using thread-local accumulators (julia-style chunked parallelism)
/// divides work into contiguous chunks, one per thread, to avoid locking overhead
#[cfg(feature = "parallel")]
pub fn induce_sumcheck_poly_parallel<T, U>(
    n: usize,
    sks_vks: &[T],
    opened_rows: &[Vec<T>],
    v_challenges: &[U],
    sorted_queries: &[usize],
    alpha: U,
) -> (Vec<U>, U)
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    use rayon::prelude::*;
    use std::sync::Arc;

    assert_eq!(opened_rows.len(), sorted_queries.len());

    let alpha_pows = precompute_alpha_powers(alpha, opened_rows.len());
    let basis_size = 1 << n;
    let n_rows = opened_rows.len();
    let n_threads = rayon::current_num_threads();

    // wrap shared data in Arc for safe sharing across threads
    let sks_vks = Arc::new(sks_vks);

    // compute chunk size
    let chunk_size = n_rows.div_ceil(n_threads);

    // process chunks in parallel, each thread produces its own basis and sum
    let results: Vec<(Vec<U>, U)> = (0..n_threads)
        .into_par_iter()
        .map(|thread_id| {
            let start_idx = thread_id * chunk_size;
            let end_idx = (start_idx + chunk_size).min(n_rows);

            if start_idx >= n_rows {
                return (vec![U::zero(); basis_size], U::zero());
            }

            let mut thread_basis = vec![U::zero(); basis_size];
            let mut thread_sum = U::zero();

            // reusable buffers for this thread
            let mut local_sks_x = vec![T::zero(); sks_vks.len()];
            let mut local_basis = vec![U::zero(); basis_size];

            for i in start_idx..end_idx {
                let row = &opened_rows[i];
                let query = sorted_queries[i];
                let alpha_pow = alpha_pows[i];

                // compute dot product
                let dot = tensorized_dot_product(row, v_challenges);
                let contribution = dot.mul(&alpha_pow);
                thread_sum = thread_sum.add(&contribution);

                let qf = T::from_bits(query as u64);

                // compute scaled basis (clears buffers internally)
                evaluate_scaled_basis_inplace(
                    &mut local_sks_x,
                    &mut local_basis,
                    &sks_vks,
                    qf,
                    alpha_pow,
                );

                // accumulate into thread-local basis
                for (j, &val) in local_basis.iter().enumerate() {
                    thread_basis[j] = thread_basis[j].add(&val);
                }
            }

            (thread_basis, thread_sum)
        })
        .collect();

    // combine results from all threads
    let mut basis_poly = vec![U::zero(); basis_size];
    let mut enforced_sum = U::zero();

    for (thread_basis, thread_sum) in results {
        for (j, val) in thread_basis.into_iter().enumerate() {
            basis_poly[j] = basis_poly[j].add(&val);
        }
        enforced_sum = enforced_sum.add(&thread_sum);
    }

    (basis_poly, enforced_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::eval_sk_at_vks;
    use ligerito_binary_fields::{BinaryElem128, BinaryElem32};

    #[test]
    fn test_alpha_powers() {
        let alpha = BinaryElem128::from(5);
        let powers = precompute_alpha_powers(alpha, 4);

        assert_eq!(powers[0], BinaryElem128::one());
        assert_eq!(powers[1], alpha);
        assert_eq!(powers[2], alpha.mul(&alpha));
        assert_eq!(powers[3], alpha.mul(&alpha).mul(&alpha));
    }

    #[test]
    fn test_tensorized_dot_matches_partial_eval() {
        let row: Vec<BinaryElem32> = (0..16).map(|i| BinaryElem32::from(i * 7 + 1)).collect();
        let rs: Vec<BinaryElem128> = (0..4).map(|i| BinaryElem128::from(0x1234 + i)).collect();
        let mut lifted: Vec<BinaryElem128> = row.iter().map(|&x| x.into()).collect();
        crate::utils::partial_eval_multilinear(&mut lifted, &rs);
        assert_eq!(tensorized_dot_product(&row, &rs), lifted[0]);
    }

    /// The Ligero identity the verifier relies on: for an honest commitment of
    /// `v` laid out as an `m × n` matrix, every encoded row `q ∈ [0, 4m)`
    /// satisfies `⟨row_q, eq(ρ)⟩ = Enc(partial_eval(v, ρ))[q]`, so the induced
    /// basis gives `⟨partial_eval(v, ρ), basis⟩ = enforced_sum`.
    #[cfg(feature = "prover")]
    #[test]
    fn test_induced_basis_matches_committed_rows() {
        fn check<T>(m: usize, n: usize)
        where
            T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
            BinaryElem128: From<T>,
        {
            type U = BinaryElem128;
            let log_m = m.trailing_zeros() as usize;
            let log_n = n.trailing_zeros() as usize;
            let v: Vec<T> = (0..m * n)
                .map(|i| T::from_bits((i as u64).wrapping_mul(0x9E37_79B9) ^ 0x5A5A))
                .collect();
            let rs = reed_solomon::reed_solomon::<T>(m, 4 * m);
            let w = crate::ligero::ligero_commit(&v, m, n, &rs);

            let rho: Vec<U> = (0..log_n).map(|i| U::from_bits(0xABCD_0000 + i as u64)).collect();
            let mut y: Vec<U> = v.iter().map(|&x| U::from(x)).collect();
            crate::utils::partial_eval_multilinear(&mut y, &rho);

            // all queries, including q >= m
            let queries: Vec<usize> = (0..4 * m).collect();
            let rows: Vec<Vec<T>> = queries.iter().map(|&q| w.mat[q].clone()).collect();
            let sks_vks: Vec<T> = eval_sk_at_vks(m);
            let alpha = U::from_bits(0x1357_9BDF);
            let (basis, sum) = induce_sumcheck_poly(log_m, &sks_vks, &rows, &rho, &queries, alpha);
            assert_eq!(inner_product(&y, &basis), sum);

            // and per row, with no batching
            let mut sks_x = vec![T::zero(); sks_vks.len()];
            let mut g = vec![U::zero(); m];
            for (&q, row) in queries.iter().zip(&rows) {
                crate::utils::evaluate_scaled_basis_inplace(
                    &mut sks_x,
                    &mut g,
                    &sks_vks,
                    T::from_bits(q as u64),
                    U::one(),
                );
                assert_eq!(tensorized_dot_product(row, &rho), inner_product(&y, &g), "q={q}");
            }
        }
        check::<BinaryElem32>(64, 16);
        check::<BinaryElem128>(32, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_vs_sequential() {
        // Test that parallel and sequential versions produce identical results
        let n = 12; // 2^12 = 4096 elements
        let sks_vks: Vec<BinaryElem32> = eval_sk_at_vks(1 << n);

        // Create realistic test data
        let num_queries = 148;
        let v_challenges = vec![
            BinaryElem128::from(0x1234567890abcdef),
            BinaryElem128::from(0xfedcba0987654321),
        ];

        let queries: Vec<usize> = (0..num_queries).map(|i| (i * 113) % (1 << n)).collect();
        let opened_rows: Vec<Vec<BinaryElem32>> = (0..num_queries)
            .map(|i| {
                (0..4)
                    .map(|j| BinaryElem32::from((i * j + 1) as u32))
                    .collect()
            })
            .collect();

        let alpha = BinaryElem128::from(0x9ABC);

        // Run sequential version
        let (seq_basis, seq_sum) =
            induce_sumcheck_poly(n, &sks_vks, &opened_rows, &v_challenges, &queries, alpha);

        // Run parallel version
        let (par_basis, par_sum) = induce_sumcheck_poly_parallel(
            n,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            alpha,
        );

        // Compare enforced sums
        assert_eq!(
            par_sum, seq_sum,
            "Parallel and sequential enforced sums differ"
        );

        // Compare basis polynomials element by element
        for (i, (&par_val, &seq_val)) in par_basis.iter().zip(seq_basis.iter()).enumerate() {
            if par_val != seq_val {
                println!(
                    "Mismatch at index {}: parallel={:?}, sequential={:?}",
                    i, par_val, seq_val
                );
            }
        }

        assert_eq!(
            par_basis, seq_basis,
            "Parallel and sequential basis polynomials differ"
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_sumcheck_parallel_consistency() {
        let n = 2; // 2^2 = 4 elements
        let sks_vks: Vec<BinaryElem32> = eval_sk_at_vks(1 << n);

        // 1 challenge -> Lagrange basis length = 2^1 = 2
        let v_challenges = vec![BinaryElem128::from(0xABCD)];

        let queries = vec![0, 1, 3];
        // each row must have length 2 to match Lagrange basis
        let opened_rows = vec![
            vec![BinaryElem32::from(7), BinaryElem32::from(9)],
            vec![BinaryElem32::from(11), BinaryElem32::from(13)],
            vec![BinaryElem32::from(15), BinaryElem32::from(17)],
        ];

        let alpha = BinaryElem128::from(0x1337);

        // Sequential version
        let (basis_seq, sum_seq) =
            induce_sumcheck_poly(n, &sks_vks, &opened_rows, &v_challenges, &queries, alpha);

        // Parallel version
        let (basis_par, sum_par) = induce_sumcheck_poly_parallel(
            n,
            &sks_vks,
            &opened_rows,
            &v_challenges,
            &queries,
            alpha,
        );

        // Results should be identical
        assert_eq!(
            sum_seq, sum_par,
            "Sequential and parallel sums should match"
        );
        assert_eq!(
            basis_seq, basis_par,
            "Sequential and parallel basis polynomials should match"
        );
    }

    #[test]
    fn test_empty_inputs() {
        let n = 2;
        let sks_vks: Vec<BinaryElem32> = eval_sk_at_vks(1 << n);
        let v_challenges = vec![BinaryElem128::from(1)];
        let queries: Vec<usize> = vec![];
        let opened_rows: Vec<Vec<BinaryElem32>> = vec![];
        let alpha = BinaryElem128::from(42);

        let (basis_poly, enforced_sum) =
            induce_sumcheck_poly(n, &sks_vks, &opened_rows, &v_challenges, &queries, alpha);

        // With no inputs, everything should be zero
        assert_eq!(enforced_sum, BinaryElem128::zero());
        assert!(basis_poly.iter().all(|&x| x == BinaryElem128::zero()));
    }
}
