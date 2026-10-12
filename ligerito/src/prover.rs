use crate::{
    data_structures::finalize,
    eval_proof::{absorb_claims, compute_batched_eq, EvalClaim, EvalSumcheckRound},
    ligero::ligero_commit,
    sumcheck_polys::{fold_lsb, product_round_coeffs},
    transcript::{FiatShamir, Transcript},
    utils::{eval_sk_at_vks, partial_eval_multilinear},
    FinalLigeroProof, FinalizedLigeritoProof, LigeritoProof, ProverConfig,
    RecursiveLigeroCommitment, RecursiveLigeroProof, SumcheckTranscript,
};
use binary_fields::BinaryFieldElement;

#[cfg(feature = "parallel")]
use crate::sumcheck_polys::induce_sumcheck_poly_parallel as induce_sumcheck_poly_auto;
#[cfg(not(feature = "parallel"))]
use crate::sumcheck_polys::induce_sumcheck_poly as induce_sumcheck_poly_auto;

/// One sumcheck round on `Σ_x f(x)·g(x)`: absorb the round message, draw the
/// challenge, bind the lsb variable of both tables.
fn product_sumcheck_round<U: BinaryFieldElement>(
    f: &mut Vec<U>,
    g: &mut Vec<U>,
    fs: &mut impl Transcript,
) -> ((U, U, U), U) {
    let coeffs = product_round_coeffs(f, g);
    fs.absorb_elem(coeffs.0);
    fs.absorb_elem(coeffs.1);
    fs.absorb_elem(coeffs.2);
    let r = fs.get_challenge::<U>();
    *f = fold_lsb(f, r);
    *g = fold_lsb(g, r);
    (coeffs, r)
}

/// The Ligerito prover.
///
/// Message vectors `v_0 = poly, v_1, …` are committed as row-contiguous
/// matrices (`ligero::poly2mat`), so binding the column variables of `v_t`
/// to `ρ` is `partial_eval_multilinear(v_t, ρ)` (lsb-first). Each round:
///
/// 1. queries `Q` on the previous commitment reduce the Ligero row checks to
///    `⟨v_{t+1}, b⟩ = s` (`induce_sumcheck_poly`), glued into the running claim
///    `⟨v_{t+1}, B⟩ = sum` with a random `β`;
/// 2. a product sumcheck on `⟨v_{t+1}, B⟩` produces the challenges `ρ_{t+1}`
///    that fold `v_{t+1}` into `v_{t+2}`, the next committed message.
///
/// The last folded message is sent in the clear as `yr`; the verifier checks
/// it against the final commitment's opened rows and against the claim.
///
/// With evaluation claims `P(z_k) = v_k`, the first `initial_k` challenges come
/// from a product sumcheck on `Σ_x P(x)·Q(x)` (`Q = Σ α_k eq(z_k, ·)`) instead
/// of the transcript directly, and its residual claim is glued in as well.
fn prove_inner<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    claims: &[EvalClaim<T>],
    fs: &mut impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    config.validate()?;
    let vcfg = config.verifier_config();
    if poly.len() != 1 << vcfg.poly_log_size() || claims.iter().any(|c| c.index >= poly.len()) {
        return Err(crate::LigeritoError::InvalidProof);
    }

    vcfg.absorb_into(fs);
    let wtns_0 = ligero_commit(
        poly,
        config.initial_dims.0,
        config.initial_dims.1,
        &config.initial_reed_solomon,
    );
    let cm_0 = RecursiveLigeroCommitment {
        root: wtns_0.tree.get_root(),
    };
    fs.absorb_root(&cm_0.root);
    absorb_claims(fs, claims);

    let mut proof = LigeritoProof::<T, U>::new();
    proof.initial_ligero_cm = Some(cm_0);

    // Bind the initial_k column variables of v_0.
    let mut v: Vec<U> = poly.iter().map(|&x| U::from(x)).collect();
    let mut eval_rounds = Vec::new();
    let (rho_0, eval_residual) = if claims.is_empty() {
        let rho: Vec<U> = (0..config.initial_k).map(|_| fs.get_challenge()).collect();
        partial_eval_multilinear(&mut v, &rho);
        (rho, None)
    } else {
        let alphas: Vec<U> = (0..claims.len()).map(|_| fs.get_challenge()).collect();
        let mut q = compute_batched_eq::<T, U>(claims, &alphas, vcfg.poly_log_size());
        let mut rho = Vec::with_capacity(config.initial_k);
        for _ in 0..config.initial_k {
            let ((s0, s1, s2), r) = product_sumcheck_round(&mut v, &mut q, fs);
            eval_rounds.push(EvalSumcheckRound { s0, s1, s2 });
            rho.push(r);
        }
        (rho, Some(q))
    };

    // v_1 = v_0(ρ_0, ·)
    let wtns_1 = ligero_commit(
        &v,
        config.dims[0].0,
        config.dims[0].1,
        &config.reed_solomon_codes[0],
    );
    let cm_1 = RecursiveLigeroCommitment {
        root: wtns_1.tree.get_root(),
    };
    proof.recursive_commitments.push(cm_1.clone());
    fs.absorb_root(&cm_1.root);

    let queries = fs.get_distinct_queries(wtns_0.mat.len(), config.num_queries);
    let alpha = fs.get_challenge::<U>();
    let opened_rows: Vec<Vec<T>> = queries.iter().map(|&q| wtns_0.mat[q].clone()).collect();
    let sks_vks: Vec<T> = eval_sk_at_vks(config.initial_dims.0);
    let (b_0, s_0) = induce_sumcheck_poly_auto(
        vcfg.initial_dim,
        &sks_vks,
        &opened_rows,
        &rho_0,
        &queries,
        alpha,
    );
    fs.absorb_elem(s_0);
    proof.initial_ligero_proof = Some(RecursiveLigeroProof {
        opened_rows,
        merkle_proof: wtns_0.tree.prove(&queries),
    });

    let mut basis = match eval_residual {
        None => b_0,
        Some(q) => {
            let beta = fs.get_challenge::<U>();
            q.iter().zip(&b_0).map(|(qi, bi)| qi.add(&beta.mul(bi))).collect()
        }
    };

    let mut sumcheck_transcript = Vec::new();
    let mut wtns_prev = wtns_1;

    for i in 0..config.recursive_steps {
        let mut rs = Vec::with_capacity(config.ks[i]);
        for _ in 0..config.ks[i] {
            let (coeffs, r) = product_sumcheck_round(&mut v, &mut basis, fs);
            sumcheck_transcript.push(coeffs);
            rs.push(r);
        }

        let rows = wtns_prev.mat.len();

        if i == config.recursive_steps - 1 {
            fs.absorb_elems(&v);
            let queries = fs.get_distinct_queries(rows, config.num_queries);
            proof.final_ligero_proof = Some(FinalLigeroProof {
                yr: v,
                opened_rows: queries.iter().map(|&q| wtns_prev.mat[q].clone()).collect(),
                merkle_proof: wtns_prev.tree.prove(&queries),
            });
            proof.sumcheck_transcript = Some(SumcheckTranscript {
                transcript: sumcheck_transcript,
            });
            let mut proof = finalize(proof)?;
            proof.eval_rounds = eval_rounds;
            return Ok(proof);
        }

        let wtns_next = ligero_commit(
            &v,
            config.dims[i + 1].0,
            config.dims[i + 1].1,
            &config.reed_solomon_codes[i + 1],
        );
        let cm_next = RecursiveLigeroCommitment {
            root: wtns_next.tree.get_root(),
        };
        proof.recursive_commitments.push(cm_next.clone());
        fs.absorb_root(&cm_next.root);

        let queries = fs.get_distinct_queries(rows, config.num_queries);
        let alpha = fs.get_challenge::<U>();
        let opened_rows: Vec<Vec<U>> = queries.iter().map(|&q| wtns_prev.mat[q].clone()).collect();
        let sks_vks: Vec<U> = eval_sk_at_vks(config.dims[i].0);
        let (b, s) = induce_sumcheck_poly_auto(
            vcfg.log_dims[i],
            &sks_vks,
            &opened_rows,
            &rs,
            &queries,
            alpha,
        );
        fs.absorb_elem(s);
        proof.recursive_proofs.push(RecursiveLigeroProof {
            opened_rows,
            merkle_proof: wtns_prev.tree.prove(&queries),
        });

        let beta = fs.get_challenge::<U>();
        basis = basis.iter().zip(&b).map(|(x, y)| x.add(&beta.mul(y))).collect();
        wtns_prev = wtns_next;
    }

    unreachable!("recursive_steps >= 1 is enforced by validate()")
}

/// Prove with a caller-supplied transcript (no evaluation claims).
pub fn prove_with_transcript<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    mut fs: impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove_inner(config, poly, &[], &mut fs)
}

/// Prove the commitment together with evaluation claims `P(z_k) = v_k`.
///
/// The claims are bound to the committed polynomial: verify with
/// `verify_with_evaluations` using the same claims.
pub fn prove_with_evaluations<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    claims: &[EvalClaim<T>],
    mut fs: impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove_inner(config, poly, claims, &mut fs)
}

/// Main prover function using default Merlin transcript
pub fn prove<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    #[cfg(feature = "transcript-merlin")]
    let fs = FiatShamir::new_merlin();

    #[cfg(not(feature = "transcript-merlin"))]
    let fs = FiatShamir::new_sha256(0);

    prove_with_transcript(config, poly, fs)
}

/// Prover function using Julia-compatible SHA256 transcript
pub fn prove_sha256<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    let fs = FiatShamir::new_sha256(1234);
    prove_with_transcript(config, poly, fs)
}

/// Prover function using BLAKE2b transcript (for Substrate runtime compatibility)
///
/// Proofs generated with this function must be verified using `verify_blake2b()`
/// or `verify_complete_blake2b()`.
#[cfg(feature = "transcript-blake2b")]
pub fn prove_blake2b<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    let fs = FiatShamir::new_blake2b();
    prove_with_transcript(config, poly, fs)
}

/// Same as `prove` (kept for API compatibility).
pub fn prove_debug<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove(config, poly)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::hardcoded_config_12;
    use ligerito_binary_fields::{BinaryElem128, BinaryElem32};
    use std::marker::PhantomData;

    #[test]
    fn test_product_sumcheck_round() {
        use crate::sumcheck_polys::{eval_round_poly, inner_product};
        type U = BinaryElem128;
        let mut f: Vec<U> = (0..8).map(|i| U::from_bits(3 * i + 1)).collect();
        let mut g: Vec<U> = (0..8).map(|i| U::from_bits(5 * i + 2)).collect();
        let claim = inner_product(&f, &g);
        let mut fs = FiatShamir::new_sha256(0);
        let ((s0, s1, s2), r) = product_sumcheck_round(&mut f, &mut g, &mut fs);
        assert_eq!(s0.add(&s1), claim);
        assert_eq!(eval_round_poly((s0, s1, s2), r), inner_product(&f, &g));
    }

    #[test]
    fn test_simple_prove() {
        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);

        // Test with all ones polynomial
        let poly = vec![BinaryElem32::one(); 1 << 12];

        // This should not panic
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Simple proof generation should succeed");
    }

    #[test]
    fn test_sumcheck_consistency_in_prover() {
        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);

        let poly = vec![BinaryElem32::zero(); 1 << 12];
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Zero polynomial proof should succeed");

        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::one();
        poly[1] = BinaryElem32::from(2);
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Simple pattern proof should succeed");
    }

    #[test]
    fn test_prove_with_evaluations() {
        use crate::eval_proof::EvalClaim;

        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = crate::hardcoded_config_12_verifier();

        // polynomial with known values
        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::from(42);
        poly[7] = BinaryElem32::from(99);
        poly[100] = BinaryElem32::from(255);

        // claim: P(0) = 42, P(7) = 99, P(100) = 255
        let claims = vec![
            EvalClaim {
                index: 0,
                value: BinaryElem32::from(42),
            },
            EvalClaim {
                index: 7,
                value: BinaryElem32::from(99),
            },
            EvalClaim {
                index: 100,
                value: BinaryElem32::from(255),
            },
        ];

        // prove
        let fs = FiatShamir::new_sha256(0);
        let proof = prove_with_evaluations(&config, &poly, &claims, fs);
        assert!(proof.is_ok(), "prove_with_evaluations should succeed");
        let proof = proof.unwrap();
        assert_eq!(
            proof.eval_rounds.len(),
            verifier_config.initial_k,
            "eval sumcheck runs over the initial_k column variables"
        );

        // verify
        let fs = FiatShamir::new_sha256(0);
        let result = crate::verifier::verify_with_evaluations::<BinaryElem32, BinaryElem128>(
            &verifier_config,
            &proof,
            &claims,
            fs,
        );
        assert!(result.unwrap(), "honest evaluation proof should verify");
    }

    #[test]
    fn test_prove_with_evaluations_wrong_claim_fails() {
        use crate::eval_proof::EvalClaim;

        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = crate::hardcoded_config_12_verifier();

        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[5] = BinaryElem32::from(77);

        // wrong claim: P(5) = 88 (actual is 77)
        let claims = vec![EvalClaim {
            index: 5,
            value: BinaryElem32::from(88),
        }];

        // prove (prover uses actual polynomial, eval sumcheck computes from actual values)
        let fs = FiatShamir::new_sha256(0);
        let proof = prove_with_evaluations(&config, &poly, &claims, fs).unwrap();

        // verify with wrong claim — eval sumcheck should fail
        let fs = FiatShamir::new_sha256(0);
        let result = crate::verifier::verify_with_evaluations::<BinaryElem32, BinaryElem128>(
            &verifier_config,
            &proof,
            &claims,
            fs,
        )
        .unwrap();

        // target (α·88) ≠ actual sum (α·77)
        assert!(!result, "wrong eval claim should fail verification");
    }
}
