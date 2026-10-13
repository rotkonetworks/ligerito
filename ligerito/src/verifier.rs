#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::{
    eval_proof::{absorb_claims, EvalClaim},
    sumcheck_polys::{eval_round_poly, precompute_alpha_powers, tensorized_dot_product},
    transcript::{FiatShamir, Transcript},
    utils::{eval_sk_at_vks, hash_row},
    FinalizedLigeritoProof, VerifierConfig,
};
use binary_fields::BinaryFieldElement;
use merkle_tree::{BatchedMerkleProof, Hash, MerkleRoot};

const LOG_INV_RATE: usize = 2;

/// `s_j(x)` for `j < k` (see `utils::evaluate_scaled_basis_inplace`).
fn subspace_evals<F: BinaryFieldElement>(x: F, k: usize, sks_vks: &[F]) -> Vec<F> {
    let mut s = Vec::with_capacity(k);
    if k > 0 {
        s.push(x);
        for j in 1..k {
            let p = s[j - 1];
            s.push(p.mul(&p).add(&sks_vks[j - 1].mul(&p)));
        }
    }
    s
}

/// `⟨y, ⊗_j (1, a_j)⟩` with `a_0` on the lsb of the index.
fn tensor_inner<U: BinaryFieldElement>(y: &[U], a: &[U]) -> U {
    debug_assert_eq!(y.len(), 1 << a.len());
    let mut cur: Vec<U> = y.chunks_exact(2).map(|p| p[0].add(&a[0].mul(&p[1]))).collect();
    for &aj in &a[1..] {
        cur = cur.chunks_exact(2).map(|p| p[0].add(&aj.mul(&p[1]))).collect();
    }
    cur[0]
}

/// The verifier's running batched basis `B`, kept factored instead of as a
/// length-`m` vector: a sum of scaled generator rows `c · ⊗_{j≥off}(1, a_j)`
/// plus a sparse part from evaluation claims.
///
/// Binding the lsb variable of `c · (1, a_off) ⊗ rest` to `r` gives
/// `c · (1 + r·(a_off + 1)) · rest`, so each fold is O(#terms) and the final
/// `⟨yr, B⟩` is O(#terms · |yr|), independent of the initial message length.
struct FactoredBasis<U> {
    rows: Vec<(U, Vec<U>, usize)>,
    sparse: Vec<(usize, U)>,
}

impl<U: BinaryFieldElement> FactoredBasis<U> {
    /// Add `scale · Σ_k α^k · G[q_k]` for a code with message length `2^log_m`
    /// over field `F`, and return `Σ_k α^k · ⟨row_k, eq(rs)⟩` (the claim it
    /// enforces on the next message).
    fn add_queries<F>(
        &mut self,
        log_m: usize,
        rows: &[Vec<F>],
        queries: &[usize],
        rs: &[U],
        alpha: U,
        scale: U,
    ) -> U
    where
        F: BinaryFieldElement,
        U: From<F>,
    {
        let sks_vks: Vec<F> = eval_sk_at_vks(1 << log_m);
        let alpha_pows = precompute_alpha_powers(alpha, queries.len());
        let mut sum = U::zero();
        for ((row, &q), &ak) in rows.iter().zip(queries).zip(&alpha_pows) {
            sum = sum.add(&ak.mul(&tensorized_dot_product(row, rs)));
            let a = subspace_evals(F::from_bits(q as u64), log_m, &sks_vks)
                .into_iter()
                .map(U::from)
                .collect();
            self.rows.push((scale.mul(&ak), a, 0));
        }
        sum
    }

    fn fold(&mut self, r: U) {
        let one = U::one();
        for (c, a, off) in &mut self.rows {
            *c = c.mul(&one.add(&r.mul(&a[*off].add(&one))));
            *off += 1;
        }
        for (i, w) in &mut self.sparse {
            *w = w.mul(&if *i & 1 == 1 { r } else { one.add(&r) });
            *i >>= 1;
        }
    }

    fn scale(&mut self, beta: U) {
        for (c, _, _) in &mut self.rows {
            *c = c.mul(&beta);
        }
        for (_, w) in &mut self.sparse {
            *w = w.mul(&beta);
        }
    }

    fn inner(&self, y: &[U]) -> U {
        let rows = self
            .rows
            .iter()
            .fold(U::zero(), |acc, (c, a, off)| acc.add(&c.mul(&tensor_inner(y, &a[*off..]))));
        self.sparse
            .iter()
            .fold(rows, |acc, &(i, w)| acc.add(&w.mul(&y[i])))
    }
}

/// Verify a Ligerito proof (default Merlin transcript)
pub fn verify<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    #[cfg(feature = "transcript-merlin")]
    let fs = FiatShamir::new_merlin();

    #[cfg(not(feature = "transcript-merlin"))]
    let fs = FiatShamir::new_sha256(0);

    verify_with_transcript(config, proof, fs)
}

/// Queries for one round, checked for shape and Merkle membership.
///
/// Returns `None` if the opened rows do not have the expected count and
/// width or do not open `root` at the transcript-chosen positions.
fn open_round<F: BinaryFieldElement>(
    fs: &mut impl Transcript,
    config: &VerifierConfig,
    root: &MerkleRoot,
    merkle_proof: &BatchedMerkleProof,
    opened_rows: &[Vec<F>],
    log_rows: usize,
    row_len: usize,
) -> Option<Vec<usize>> {
    let rows = 1usize << log_rows;
    let queries = fs.get_distinct_queries(rows, config.num_queries);
    if queries.len() != config.num_queries.min(rows)
        || opened_rows.len() != queries.len()
        || opened_rows.iter().any(|r| r.len() != row_len)
    {
        return None;
    }
    let hashed: Vec<Hash> = opened_rows.iter().map(|row| hash_row(row)).collect();
    ligerito_merkle::verify(root, merkle_proof, log_rows, &hashed, &queries).then_some(queries)
}

/// The Ligerito verifier; mirrors `prover::prove_inner` step for step.
///
/// Checks, in order: config and proof shapes; Merkle openings of every round;
/// each sumcheck round against the running claim; and at the end both
/// `claim == ⟨yr, B⟩` (B = folded, glued batched generator rows, computed
/// here from the opened rows) and, per final query, the Ligero row check
/// `⟨row_q, eq(ρ)⟩ == Enc(yr)[q]`.
fn verify_inner<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    claims: &[EvalClaim<T>],
    fs: &mut impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    if !config.is_well_formed() {
        return Err(crate::LigeritoError::InvalidProof);
    }
    let steps = config.recursive_steps;
    let last = steps - 1;
    let n_total = config.poly_log_size();
    let expected_eval_rounds = if claims.is_empty() { 0 } else { config.initial_k };
    if proof.recursive_commitments.len() != steps
        || proof.recursive_proofs.len() != steps - 1
        || proof.sumcheck_transcript.transcript.len() != config.ks.iter().sum::<usize>()
        || proof.eval_rounds.len() != expected_eval_rounds
        || proof.final_ligero_proof.yr.len() != 1 << config.log_dims[last]
        || claims.iter().any(|c| c.index >> n_total != 0)
    {
        return Ok(false);
    }

    config.absorb_into(fs);
    fs.absorb_root(&proof.initial_ligero_cm.root);
    absorb_claims(fs, claims);

    // Initial challenges ρ_0, from the eval sumcheck when there are claims.
    let mut sum = U::zero();
    let mut eval_alphas = Vec::new();
    let mut rho_0 = Vec::with_capacity(config.initial_k);
    if claims.is_empty() {
        for _ in 0..config.initial_k {
            rho_0.push(fs.get_challenge::<U>());
        }
    } else {
        eval_alphas = (0..claims.len()).map(|_| fs.get_challenge::<U>()).collect();
        for (c, &a) in claims.iter().zip(&eval_alphas) {
            sum = sum.add(&a.mul(&U::from(c.value)));
        }
        for round in &proof.eval_rounds {
            if round.s0.add(&round.s1) != sum {
                return Ok(false);
            }
            fs.absorb_elem(round.s0);
            fs.absorb_elem(round.s1);
            fs.absorb_elem(round.s2);
            let r = fs.get_challenge::<U>();
            sum = eval_round_poly((round.s0, round.s1, round.s2), r);
            rho_0.push(r);
        }
    }

    fs.absorb_root(&proof.recursive_commitments[0].root);

    let initial = &proof.initial_ligero_proof;
    let Some(queries) = open_round(
        fs,
        config,
        &proof.initial_ligero_cm.root,
        &initial.merkle_proof,
        &initial.opened_rows,
        config.initial_dim + LOG_INV_RATE,
        1 << config.initial_k,
    ) else {
        return Ok(false);
    };
    let alpha = fs.get_challenge::<U>();
    let mut basis = FactoredBasis {
        rows: Vec::with_capacity(queries.len() * steps),
        sparse: Vec::new(),
    };
    let s_0 = basis.add_queries::<T>(
        config.initial_dim,
        &initial.opened_rows,
        &queries,
        &rho_0,
        alpha,
        U::one(),
    );
    fs.absorb_elem(s_0);

    if claims.is_empty() {
        sum = s_0;
    } else {
        // Q(ρ_0, ·): claim k sits at z_k >> initial_k with weight
        // α_k · eq(z_k mod 2^initial_k, ρ_0); glue b_0 in with β
        let beta = fs.get_challenge::<U>();
        basis.scale(beta);
        for (c, &a) in claims.iter().zip(&eval_alphas) {
            let mut w = a;
            for (j, &r) in rho_0.iter().enumerate() {
                w = w.mul(&if (c.index >> j) & 1 == 1 { r } else { U::one().add(&r) });
            }
            basis.sparse.push((c.index >> config.initial_k, w));
        }
        sum = sum.add(&beta.mul(&s_0));
    }

    let mut rounds = proof.sumcheck_transcript.transcript.iter();

    for i in 0..steps {
        let mut rs = Vec::with_capacity(config.ks[i]);
        for _ in 0..config.ks[i] {
            let Some(&coeffs) = rounds.next() else {
                return Ok(false);
            };
            if coeffs.0.add(&coeffs.1) != sum {
                return Ok(false);
            }
            fs.absorb_elem(coeffs.0);
            fs.absorb_elem(coeffs.1);
            fs.absorb_elem(coeffs.2);
            let r = fs.get_challenge::<U>();
            sum = eval_round_poly(coeffs, r);
            basis.fold(r);
            rs.push(r);
        }

        let root = &proof.recursive_commitments[i].root;
        let log_rows = config.log_dims[i] + LOG_INV_RATE;
        if i == last {
            let fin = &proof.final_ligero_proof;
            fs.absorb_elems(&fin.yr);
            let Some(queries) = open_round(
                fs,
                config,
                root,
                &fin.merkle_proof,
                &fin.opened_rows,
                log_rows,
                1 << config.ks[i],
            ) else {
                return Ok(false);
            };

            // the sumcheck chain must end in the claimed folded message
            if basis.inner(&fin.yr) != sum {
                return Ok(false);
            }

            // every opened row of the final commitment must agree with yr:
            // ⟨row_q, eq(rs)⟩ = Enc(yr)[q] = ⟨yr, ⊗_j (1, s_j(ω_q))⟩
            let sks_vks: Vec<U> = eval_sk_at_vks(1 << config.log_dims[i]);
            for (&q, row) in queries.iter().zip(&fin.opened_rows) {
                let a = subspace_evals(U::from_bits(q as u64), config.log_dims[i], &sks_vks);
                if tensorized_dot_product(row, &rs) != tensor_inner(&fin.yr, &a) {
                    return Ok(false);
                }
            }
            return Ok(true);
        }

        fs.absorb_root(&proof.recursive_commitments[i + 1].root);
        let ligero_proof = &proof.recursive_proofs[i];
        let Some(queries) = open_round(
            fs,
            config,
            root,
            &ligero_proof.merkle_proof,
            &ligero_proof.opened_rows,
            log_rows,
            1 << config.ks[i],
        ) else {
            return Ok(false);
        };
        let alpha = fs.get_challenge::<U>();
        // β is drawn after s is absorbed, so add the rows unscaled and fix up
        let start = basis.rows.len();
        let s = basis.add_queries::<U>(
            config.log_dims[i],
            &ligero_proof.opened_rows,
            &queries,
            &rs,
            alpha,
            U::one(),
        );
        fs.absorb_elem(s);
        let beta = fs.get_challenge::<U>();
        sum = sum.add(&beta.mul(&s));
        for (c, _, _) in &mut basis.rows[start..] {
            *c = c.mul(&beta);
        }
    }

    unreachable!("the final round returns")
}

/// Verify with custom transcript implementation
pub fn verify_with_transcript<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    mut fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify_inner(config, proof, &[], &mut fs)
}

/// Verify a proof from `prove_with_evaluations`: the commitment opens to a
/// polynomial with `P(z_k) = v_k` for every claim.
pub fn verify_with_evaluations<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    claims: &[EvalClaim<T>],
    mut fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    if claims.is_empty() {
        return Err(crate::LigeritoError::InvalidProof);
    }
    verify_inner(config, proof, claims, &mut fs)
}

/// SHA256-based verification for compatibility
pub fn verify_sha256<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(1234);
    verify_with_transcript(config, proof, fs)
}

/// Same as `verify` (kept for API compatibility).
pub fn verify_debug<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify(config, proof)
}

/// Same as `verify_with_transcript` (kept for API compatibility).
pub fn verify_complete_with_transcript<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify_with_transcript(config, proof, fs)
}

/// complete verifier with default transcript
#[cfg(feature = "transcript-merlin")]
pub fn verify_complete<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_merlin();
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with default transcript (SHA256 when Merlin not available)
#[cfg(not(feature = "transcript-merlin"))]
pub fn verify_complete<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(0);
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with SHA256 transcript (Julia-compatible)
pub fn verify_complete_sha256<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(1234);
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with BLAKE2b transcript (optimized for Substrate runtimes)
///
/// Uses `sp_io::hashing::blake2_256` in no_std for efficient host function calls.
/// Proofs must be generated with `prove_blake2b()` to be compatible.
#[cfg(feature = "transcript-blake2b")]
pub fn verify_blake2b<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_blake2b();
    verify_with_transcript(config, proof, fs)
}

/// complete verifier with BLAKE2b transcript
#[cfg(feature = "transcript-blake2b")]
pub fn verify_complete_blake2b<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_blake2b();
    verify_complete_with_transcript(config, proof, fs)
}

// ============================================================================
// bytes-based verification (no_std, no atomics, no SCALE codec)
// ============================================================================

/// Verify a proof from serialized bytes using postcard (no_std serde format).
///
/// This function handles deserialization internally, avoiding the need for
/// parity-scale-codec (which requires Arc/atomics). Suitable for on-chain
/// verification on targets like riscv32em (PolkaVM/JAM).
///
/// The proof bytes must be serialized with `postcard::to_allocvec()` by the prover.
/// Public outputs bytes are absorbed into the transcript before verification.
///
/// Returns Ok(true) if the proof is valid, Ok(false) if invalid, Err on deserialize failure.
#[cfg(feature = "postcard-serde")]
pub fn verify_bytes<T, U>(
    config: &VerifierConfig,
    proof_bytes: &[u8],
    public_outputs_bytes: &[u8],
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync + serde::de::DeserializeOwned,
    U: BinaryFieldElement + Send + Sync + From<T> + serde::de::DeserializeOwned,
{
    let proof: FinalizedLigeritoProof<T, U> =
        postcard::from_bytes(proof_bytes).map_err(|_| crate::LigeritoError::InvalidProof)?;

    let mut fs = FiatShamir::new_sha256(0);
    fs.absorb_bytes(b"public_outputs", public_outputs_bytes);

    verify_with_transcript(config, &proof, fs)
}

/// Serialize a proof to bytes using postcard (for prover side).
///
/// The resulting bytes can be verified with `verify_bytes()`.
#[cfg(feature = "postcard-serde")]
pub fn proof_to_bytes<T, U>(
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<alloc::vec::Vec<u8>>
where
    T: BinaryFieldElement + Send + Sync + serde::Serialize,
    U: BinaryFieldElement + Send + Sync + serde::Serialize,
{
    postcard::to_allocvec(proof).map_err(|_| crate::LigeritoError::InvalidProof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::{hardcoded_config_12, hardcoded_config_12_verifier};
    use crate::prover::prove;
    use ligerito_binary_fields::{BinaryElem128, BinaryElem32};
    use std::marker::PhantomData;

    #[test]
    fn test_verify_simple_proof() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with simple polynomial
        let poly = vec![BinaryElem32::one(); 1 << 12];

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify(&verifier_config, &proof).expect("Verification failed");

        assert!(result, "Verification should succeed for valid proof");
    }

    #[test]
    fn test_verify_zero_polynomial() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with zero polynomial
        let poly = vec![BinaryElem32::zero(); 1 << 12];

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify(&verifier_config, &proof).expect("Verification failed");

        assert!(result, "Verification should succeed for zero polynomial");
    }

    #[test]
    fn test_verify_with_sha256() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with patterned polynomial
        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::one();
        poly[1] = BinaryElem32::from(2);

        let proof = crate::prover::prove_sha256(&prover_config, &poly)
            .expect("SHA256 proof generation failed");
        let result = verify_sha256(&verifier_config, &proof).expect("SHA256 verification failed");

        assert!(result, "SHA256 verification should succeed");
    }

    #[test]
    fn test_debug_verification() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Use a non-constant polynomial to avoid degenerate case
        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32)) // Some pattern to ensure diversity
            .collect();

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify_debug(&verifier_config, &proof).expect("Debug verification failed");

        assert!(result, "Debug verification should succeed");
    }

    #[test]
    #[cfg(feature = "transcript-blake2b")]
    fn test_blake2b_transcript_compatibility() {
        use crate::prover::prove_blake2b;

        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with patterned polynomial
        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32))
            .collect();

        // Prove with Blake2b
        let proof = prove_blake2b(&prover_config, &poly).expect("Blake2b proof generation failed");

        // Verify with Blake2b
        let result = verify_blake2b(&verifier_config, &proof).expect("Blake2b verification failed");

        assert!(
            result,
            "Blake2b verification should succeed for valid proof"
        );

        // Also test complete verifier
        let proof2 = prove_blake2b(&prover_config, &poly).expect("Blake2b proof generation failed");
        let result2 = verify_complete_blake2b(&verifier_config, &proof2)
            .expect("Blake2b complete verification failed");

        assert!(result2, "Blake2b complete verification should succeed");
    }

    #[test]
    #[cfg(feature = "transcript-blake2b")]
    fn test_sha256_transcript_compatibility() {
        // This test verifies that SHA256 proofs cannot be verified with Blake2b
        // (and vice versa) - they are NOT compatible
        use crate::prover::prove_sha256;

        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32))
            .collect();

        // Prove with SHA256
        let sha_proof =
            prove_sha256(&prover_config, &poly).expect("SHA256 proof generation failed");

        // Trying to verify SHA256 proof with Blake2b should fail
        let result = verify_blake2b(&verifier_config, &sha_proof)
            .expect("Verification call should not panic");

        assert!(
            !result,
            "SHA256 proof should NOT verify with Blake2b transcript"
        );
    }

    /// Commit uniformly random (non-codeword) matrices and follow the
    /// transcript exactly, patching every round message so `g(0) + g(1)`
    /// matches the running claim. This forgery verified before the fix.
    #[test]
    fn test_random_matrix_forgery_rejected() {
        use crate::data_structures::{finalize, RecursiveLigeroWitness};
        use crate::sumcheck_polys::{fold_lsb, induce_sumcheck_poly, product_round_coeffs};
        use crate::{
            FinalLigeroProof, LigeritoProof, RecursiveLigeroCommitment, RecursiveLigeroProof,
            SumcheckTranscript,
        };
        type T = BinaryElem32;
        type U = BinaryElem128;

        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        fn garbage<F: BinaryFieldElement + bytemuck::Pod + Send + Sync>(
            rows: usize,
            cols: usize,
            rnd: &mut impl FnMut() -> u64,
        ) -> RecursiveLigeroWitness<F> {
            let data: Vec<F> = (0..rows * cols).map(|_| F::from_bits(rnd())).collect();
            RecursiveLigeroWitness::from_columns(data, cols)
        }

        let pc = hardcoded_config_12(PhantomData::<T>, PhantomData::<U>);
        let vc = hardcoded_config_12_verifier();
        let mut fs = FiatShamir::new_sha256(0);

        vc.absorb_into(&mut fs);
        let w0 = garbage::<T>(pc.initial_dims.0 * 4, pc.initial_dims.1, &mut rnd);
        let cm0 = RecursiveLigeroCommitment { root: w0.tree.get_root() };
        fs.absorb_root(&cm0.root);
        absorb_claims::<T>(&mut fs, &[]);
        let rho0: Vec<U> = (0..vc.initial_k).map(|_| fs.get_challenge()).collect();

        let w1 = garbage::<U>(pc.dims[0].0 * 4, pc.dims[0].1, &mut rnd);
        let cm1 = RecursiveLigeroCommitment { root: w1.tree.get_root() };
        fs.absorb_root(&cm1.root);

        let queries = fs.get_distinct_queries(w0.num_rows(), vc.num_queries);
        let alpha = fs.get_challenge::<U>();
        let rows: Vec<Vec<T>> = queries.iter().map(|&q| w0.row(q)).collect();
        let sks: Vec<T> = eval_sk_at_vks(1 << vc.initial_dim);
        let (mut basis, mut sum) =
            induce_sumcheck_poly(vc.initial_dim, &sks, &rows, &rho0, &queries, alpha);
        fs.absorb_elem(sum);

        // the forger's message: anything of the right length
        let mut v: Vec<U> = (0..1 << vc.initial_dim).map(|_| U::from_bits(rnd())).collect();
        let mut transcript = vec![];
        for _ in 0..vc.ks[0] {
            let (s0, _, s2) = product_round_coeffs(&v, &basis);
            let c = (s0, s0.add(&sum), s2); // force g(0) + g(1) = sum
            fs.absorb_elem(c.0);
            fs.absorb_elem(c.1);
            fs.absorb_elem(c.2);
            let r = fs.get_challenge::<U>();
            sum = eval_round_poly(c, r);
            v = fold_lsb(&v, r);
            basis = fold_lsb(&basis, r);
            transcript.push(c);
        }
        fs.absorb_elems(&v);
        let fq = fs.get_distinct_queries(w1.num_rows(), vc.num_queries);

        let mut proof = LigeritoProof::<T, U>::new();
        proof.initial_ligero_cm = Some(cm0);
        proof.initial_ligero_proof = Some(RecursiveLigeroProof {
            opened_rows: rows,
            merkle_proof: w0.tree.prove(&queries),
        });
        proof.recursive_commitments.push(cm1);
        proof.final_ligero_proof = Some(FinalLigeroProof {
            yr: v,
            opened_rows: fq.iter().map(|&q| w1.row(q)).collect(),
            merkle_proof: w1.tree.prove(&fq),
        });
        proof.sumcheck_transcript = Some(SumcheckTranscript { transcript });
        let proof = finalize(proof).unwrap();

        assert!(!verify_with_transcript(&vc, &proof, FiatShamir::new_sha256(0)).unwrap());
    }
}
