//! Soundness regression tests: honest proofs verify, tampered proofs are
//! rejected with `Ok(false)` (never a panic).

use ligerito::{
    eval_proof::EvalClaim, prove_with_evaluations, prove_with_transcript, verify,
    verify_with_evaluations, verify_with_transcript, FiatShamir, FinalizedLigeritoProof,
    ProverConfig, VerifierConfig,
};
use ligerito_binary_fields::{BinaryElem128, BinaryElem32, BinaryFieldElement};
use ligerito_reed_solomon::reed_solomon;

type T = BinaryElem32;
type U = BinaryElem128;
type Proof = FinalizedLigeritoProof<T, U>;

/// 2^12 polynomial: initial (2^8 × 2^4), then `steps` rounds folding 2 vars each.
fn config(steps: usize) -> (ProverConfig<T, U>, VerifierConfig) {
    let initial_dims = (1 << 8, 1 << 4);
    let dims: Vec<(usize, usize)> = (0..steps).map(|i| (1 << (6 - 2 * i), 1 << 2)).collect();
    let ks = vec![2; steps];
    let prover = ProverConfig {
        recursive_steps: steps,
        initial_dims,
        dims: dims.clone(),
        initial_k: 4,
        ks: ks.clone(),
        initial_reed_solomon: reed_solomon::<T>(initial_dims.0, initial_dims.0 * 4),
        reed_solomon_codes: dims.iter().map(|&(m, _)| reed_solomon::<U>(m, m * 4)).collect(),
        num_queries: 148,
    };
    let verifier = prover.verifier_config();
    (prover, verifier)
}

fn poly(seed: u64) -> Vec<T> {
    let mut x = seed | 1;
    (0..1 << 12)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            T::from(x as u32)
        })
        .collect()
}

fn prove(steps: usize, seed: u64) -> (Proof, VerifierConfig) {
    let (pc, vc) = config(steps);
    let proof = prove_with_transcript(&pc, &poly(seed), FiatShamir::new_sha256(0)).unwrap();
    (proof, vc)
}

fn accepts(vc: &VerifierConfig, proof: &Proof) -> bool {
    verify_with_transcript(vc, proof, FiatShamir::new_sha256(0)).expect("verifier must not error")
}

#[test]
fn honest_proofs_verify() {
    for steps in 1..=3 {
        let (proof, vc) = prove(steps, 7);
        assert!(accepts(&vc, &proof), "steps={steps}");
    }
}

#[test]
fn honest_proof_default_transcript() {
    let (pc, vc) = config(2);
    let proof = ligerito::prove(&pc, &poly(3)).unwrap();
    assert!(verify(&vc, &proof).unwrap());
}

/// Run `mutate` on an honest proof for each step count; it must be rejected.
fn rejects(name: &str, mutate: impl Fn(&mut Proof)) {
    for steps in 1..=3 {
        let (mut proof, vc) = prove(steps, 11);
        mutate(&mut proof);
        assert!(!accepts(&vc, &proof), "{name} accepted (steps={steps})");
    }
}

#[test]
fn rejects_modified_yr() {
    // the originally reported issue: a different final response for the
    // same commitment
    rejects("yr + e", |p| {
        p.final_ligero_proof.yr[0] = p.final_ligero_proof.yr[0].add(&U::one())
    });
    rejects("yr scaled", |p| {
        for y in &mut p.final_ligero_proof.yr {
            *y = y.mul(&U::from_bits(3));
        }
    });
}

#[test]
fn rejects_sum_preserving_round_tampering() {
    // shifting g(0) and g(1) by the same δ keeps g(0)+g(1) consistent;
    // the old verifier never constrained g(0)
    rejects("round shift", |p| {
        let d = U::from_bits(0x1234);
        let c = &mut p.sumcheck_transcript.transcript[0];
        *c = (c.0.add(&d), c.1.add(&d), c.2);
    });
    rejects("last round shift", |p| {
        let d = U::from_bits(0x99);
        let c = p.sumcheck_transcript.transcript.last_mut().unwrap();
        *c = (c.0.add(&d), c.1.add(&d), c.2);
    });
    rejects("x^2 coeff", |p| {
        let c = &mut p.sumcheck_transcript.transcript[0];
        c.2 = c.2.add(&U::one());
    });
}

#[test]
fn rejects_tampered_openings() {
    rejects("final row", |p| {
        let r = &mut p.final_ligero_proof.opened_rows[3];
        r[0] = r[0].add(&U::one());
    });
    rejects("initial row", |p| {
        let r = &mut p.initial_ligero_proof.opened_rows[0];
        r[1] = r[1].add(&T::one());
    });
    for steps in 2..=3 {
        let (mut proof, vc) = prove(steps, 5);
        let r = &mut proof.recursive_proofs[0].opened_rows[7];
        r[0] = r[0].add(&U::one());
        assert!(!accepts(&vc, &proof));
    }
}

#[test]
fn rejects_malformed_shapes_without_panicking() {
    rejects("truncated sumcheck", |p| {
        p.sumcheck_transcript.transcript.pop();
    });
    rejects("extra sumcheck round", |p| {
        let c = p.sumcheck_transcript.transcript[0];
        p.sumcheck_transcript.transcript.push(c);
    });
    rejects("short yr", |p| {
        p.final_ligero_proof.yr.pop();
    });
    rejects("missing final row", |p| {
        p.final_ligero_proof.opened_rows.pop();
    });
    rejects("short initial row", |p| {
        p.initial_ligero_proof.opened_rows[0].pop();
    });
    rejects("empty initial rows", |p| {
        p.initial_ligero_proof.opened_rows.clear();
    });
    rejects("missing commitment", |p| {
        p.recursive_commitments.pop();
    });
    rejects("spurious eval rounds", |p| {
        p.eval_rounds.push(ligerito::eval_proof::EvalSumcheckRound {
            s0: U::zero(),
            s1: U::zero(),
            s2: U::zero(),
        });
    });
}

#[test]
fn rejects_proof_for_different_polynomial_commitment() {
    // splice the initial commitment of one polynomial into another's proof
    let (mut a, vc) = prove(2, 1);
    let (b, _) = prove(2, 2);
    a.initial_ligero_cm = b.initial_ligero_cm.clone();
    a.initial_ligero_proof = b.initial_ligero_proof.clone();
    assert!(!accepts(&vc, &a));
}

#[test]
fn rejects_wrong_config() {
    let (proof, vc) = prove(2, 9);
    let mut other = vc.clone();
    other.num_queries = 200;
    assert!(!accepts(&other, &proof));
}

#[test]
fn evaluation_claims_bind_to_commitment() {
    for steps in 1..=3 {
        let (pc, vc) = config(steps);
        let p = poly(21);
        let claims: Vec<EvalClaim<T>> = [0usize, 7, 1000, 4095]
            .iter()
            .map(|&index| EvalClaim { index, value: p[index] })
            .collect();
        let proof = prove_with_evaluations(&pc, &p, &claims, FiatShamir::new_sha256(0)).unwrap();
        let check = |cl: &[EvalClaim<T>]| {
            verify_with_evaluations(&vc, &proof, cl, FiatShamir::new_sha256(0)).unwrap()
        };
        assert!(check(&claims), "honest claims (steps={steps})");

        let mut wrong_value = claims.clone();
        wrong_value[2].value = wrong_value[2].value.add(&T::one());
        assert!(!check(&wrong_value), "wrong value accepted");

        let mut wrong_index = claims.clone();
        wrong_index[1].index = 8;
        assert!(!check(&wrong_index), "wrong index accepted");

        let mut out_of_range = claims.clone();
        out_of_range[0].index = 1 << 12;
        assert!(!check(&out_of_range), "out-of-range index accepted");

        // a claims proof is not a plain proof, and vice versa
        assert!(!accepts(&vc, &proof));
    }
}

/// A prover that knows the value it claims but commits a *different*
/// polynomial, then runs the rest of the protocol honestly, must fail.
#[test]
fn rejects_false_claim_from_honest_prover() {
    let (pc, vc) = config(2);
    let p = poly(33);
    let lie = vec![EvalClaim { index: 5, value: p[5].add(&T::one()) }];
    let proof = prove_with_evaluations(&pc, &p, &lie, FiatShamir::new_sha256(0)).unwrap();
    assert!(!verify_with_evaluations(&vc, &proof, &lie, FiatShamir::new_sha256(0)).unwrap());
}
