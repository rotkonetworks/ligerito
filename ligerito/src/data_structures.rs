#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use binary_fields::BinaryFieldElement;
use merkle_tree::{BatchedMerkleProof, CompleteMerkleTree, MerkleRoot};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "prover")]
use reed_solomon::ReedSolomon;

/// Prover configuration (only with prover feature)
#[cfg(feature = "prover")]
pub struct ProverConfig<T: BinaryFieldElement, U: BinaryFieldElement> {
    pub recursive_steps: usize,
    pub initial_dims: (usize, usize),
    pub dims: Vec<(usize, usize)>,
    pub initial_k: usize,
    pub ks: Vec<usize>,
    pub initial_reed_solomon: ReedSolomon<T>,
    pub reed_solomon_codes: Vec<ReedSolomon<U>>,
    /// Number of query rows to open (security parameter |S|)
    ///
    /// This determines the soundness of the proof system. The minimum value is 148
    /// for 100-bit security when using Reed-Solomon codes with rate ρ=1/4 and ℓ≤8.
    ///
    /// Higher values provide more security:
    /// - 148: 100-bit security (minimum recommended)
    /// - 256: ~170-bit security
    /// - 512: ~340-bit security
    ///
    /// Note: If using RAA codes for G₁ (future WASM optimization), this would need
    /// to be 1060 for round 1 due to RAA's worse distance (0.19 vs 0.25).
    ///
    /// GPU backends can handle larger values efficiently, while increasing proof size
    /// linearly with the number of queries.
    pub num_queries: usize,
}

#[cfg(feature = "prover")]
impl<T: BinaryFieldElement, U: BinaryFieldElement> ProverConfig<T, U> {
    /// Validate configuration parameters
    pub fn validate(&self) -> crate::Result<()> {
        if self.num_queries < 148 {
            #[cfg(feature = "std")]
            return Err(crate::LigeritoError::InvalidConfig(format!(
                "num_queries must be >= 148 for 100-bit security, got {}",
                self.num_queries
            )));
            #[cfg(not(feature = "std"))]
            return Err(crate::LigeritoError::InvalidConfig);
        }
        let shape_ok = self.initial_dims.1 == 1 << self.initial_k
            && self.recursive_steps >= 1
            && self.dims.len() == self.recursive_steps
            && self.ks.len() == self.recursive_steps
            && self.reed_solomon_codes.len() == self.recursive_steps
            && self.initial_reed_solomon.message_length() == self.initial_dims.0
            && self.verifier_config().is_well_formed()
            && self
                .dims
                .iter()
                .zip(&self.ks)
                .zip(&self.reed_solomon_codes)
                .all(|((&(m, n), &k), rs)| n == 1 << k && rs.message_length() == m && m.is_power_of_two());
        if !shape_ok {
            #[cfg(feature = "std")]
            return Err(crate::LigeritoError::InvalidConfig(
                "inconsistent dims / ks / reed-solomon codes".into(),
            ));
            #[cfg(not(feature = "std"))]
            return Err(crate::LigeritoError::InvalidConfig);
        }
        Ok(())
    }

    /// The verifier configuration matching this prover configuration.
    pub fn verifier_config(&self) -> VerifierConfig {
        VerifierConfig {
            recursive_steps: self.recursive_steps,
            initial_dim: self.initial_dims.0.trailing_zeros() as usize,
            log_dims: self
                .dims
                .iter()
                .map(|&(m, _)| m.trailing_zeros() as usize)
                .collect(),
            initial_k: self.initial_k,
            ks: self.ks.clone(),
            num_queries: self.num_queries,
        }
    }
}

/// Verifier configuration
#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub recursive_steps: usize,
    pub initial_dim: usize,
    pub log_dims: Vec<usize>,
    pub initial_k: usize,
    pub ks: Vec<usize>,
    /// Number of query rows (must match prover)
    pub num_queries: usize,
}

impl VerifierConfig {
    /// Log₂ of the committed polynomial size: initial_dim + initial_k.
    /// This is the number of sumcheck rounds needed for evaluation proofs.
    pub fn poly_log_size(&self) -> usize {
        self.initial_dim + self.initial_k
    }

    /// Dimensions chain correctly: each round's message length equals the
    /// previous round's folded length.
    pub fn is_well_formed(&self) -> bool {
        self.recursive_steps >= 1
            && self.num_queries > 0
            && self.log_dims.len() == self.recursive_steps
            && self.ks.len() == self.recursive_steps
            && self.poly_log_size() < usize::BITS as usize - 2
            && (0..self.recursive_steps).all(|i| {
                let prev = if i == 0 {
                    self.initial_dim
                } else {
                    self.log_dims[i - 1]
                };
                prev == self.log_dims[i] + self.ks[i]
            })
    }

    /// Bind the public parameters into the Fiat-Shamir transcript.
    pub(crate) fn absorb_into(&self, fs: &mut impl crate::transcript::Transcript) {
        let mut header = Vec::with_capacity(4 * (4 + 2 * self.recursive_steps));
        for v in [
            self.initial_dim,
            self.initial_k,
            self.recursive_steps,
            self.num_queries,
        ]
        .into_iter()
        .chain(self.log_dims.iter().copied())
        .chain(self.ks.iter().copied())
        {
            header.extend_from_slice(&(v as u32).to_le_bytes());
        }
        fs.absorb_bytes(b"ligerito-v1-config", &header);
    }
}

/// Recursive Ligero witness (prover side only)
#[cfg(feature = "prover")]
pub struct RecursiveLigeroWitness<T: BinaryFieldElement> {
    pub mat: Vec<Vec<T>>, // Row-major matrix
    pub tree: CompleteMerkleTree,
}

/// Recursive Ligero commitment
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct RecursiveLigeroCommitment {
    pub root: MerkleRoot,
}

impl RecursiveLigeroCommitment {
    pub fn size_of(&self) -> usize {
        self.root.size_of()
    }
}

/// Recursive Ligero proof
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct RecursiveLigeroProof<T: BinaryFieldElement> {
    pub opened_rows: Vec<Vec<T>>,
    pub merkle_proof: BatchedMerkleProof,
}

impl<T: BinaryFieldElement> RecursiveLigeroProof<T> {
    pub fn size_of(&self) -> usize {
        self.opened_rows
            .iter()
            .map(|row| row.len() * core::mem::size_of::<T>())
            .sum::<usize>()
            + self.merkle_proof.size_of()
    }
}

/// Final Ligero proof
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct FinalLigeroProof<T: BinaryFieldElement> {
    pub yr: Vec<T>,
    pub opened_rows: Vec<Vec<T>>,
    pub merkle_proof: BatchedMerkleProof,
}

impl<T: BinaryFieldElement> FinalLigeroProof<T> {
    pub fn size_of(&self) -> usize {
        self.yr.len() * core::mem::size_of::<T>()
            + self
                .opened_rows
                .iter()
                .map(|row| row.len() * core::mem::size_of::<T>())
                .sum::<usize>()
            + self.merkle_proof.size_of()
    }
}

/// Sumcheck transcript
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct SumcheckTranscript<T: BinaryFieldElement> {
    pub transcript: Vec<(T, T, T)>, // Quadratic polynomial coefficients
}

impl<T: BinaryFieldElement> SumcheckTranscript<T> {
    pub fn size_of(&self) -> usize {
        self.transcript.len() * 3 * core::mem::size_of::<T>()
    }
}

/// Complete Ligerito proof (builder pattern)
pub struct LigeritoProof<T: BinaryFieldElement, U: BinaryFieldElement> {
    pub initial_ligero_cm: Option<RecursiveLigeroCommitment>,
    pub initial_ligero_proof: Option<RecursiveLigeroProof<T>>,
    pub recursive_commitments: Vec<RecursiveLigeroCommitment>,
    pub recursive_proofs: Vec<RecursiveLigeroProof<U>>,
    pub final_ligero_proof: Option<FinalLigeroProof<U>>,
    pub sumcheck_transcript: Option<SumcheckTranscript<U>>,
}

impl<T: BinaryFieldElement, U: BinaryFieldElement> Default for LigeritoProof<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BinaryFieldElement, U: BinaryFieldElement> LigeritoProof<T, U> {
    pub fn new() -> Self {
        Self {
            initial_ligero_cm: None,
            initial_ligero_proof: None,
            recursive_commitments: Vec::new(),
            recursive_proofs: Vec::new(),
            final_ligero_proof: None,
            sumcheck_transcript: None,
        }
    }
}

/// Finalized Ligerito proof
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct FinalizedLigeritoProof<T: BinaryFieldElement, U: BinaryFieldElement> {
    pub initial_ligero_cm: RecursiveLigeroCommitment,
    pub initial_ligero_proof: RecursiveLigeroProof<T>,
    pub recursive_commitments: Vec<RecursiveLigeroCommitment>,
    pub recursive_proofs: Vec<RecursiveLigeroProof<U>>,
    pub final_ligero_proof: FinalLigeroProof<U>,
    pub sumcheck_transcript: SumcheckTranscript<U>,
    /// Evaluation sumcheck rounds proving P(z_k) = v_k at specific positions.
    /// Empty when no evaluation claims are made (backwards compatible).
    #[cfg_attr(feature = "serde", serde(default))]
    pub eval_rounds: Vec<crate::eval_proof::EvalSumcheckRound<U>>,
}

impl<T: BinaryFieldElement, U: BinaryFieldElement> FinalizedLigeritoProof<T, U> {
    pub fn size_of(&self) -> usize {
        self.initial_ligero_cm.size_of()
            + self.initial_ligero_proof.size_of()
            + self
                .recursive_commitments
                .iter()
                .map(|c| c.size_of())
                .sum::<usize>()
            + self
                .recursive_proofs
                .iter()
                .map(|p| p.size_of())
                .sum::<usize>()
            + self.final_ligero_proof.size_of()
            + self.sumcheck_transcript.size_of()
            + self.eval_rounds.len() * 3 * core::mem::size_of::<U>()
    }
}

/// Finalize a proof builder into a complete proof
pub fn finalize<T: BinaryFieldElement, U: BinaryFieldElement>(
    proof: LigeritoProof<T, U>,
) -> crate::Result<FinalizedLigeritoProof<T, U>> {
    Ok(FinalizedLigeritoProof {
        initial_ligero_cm: proof
            .initial_ligero_cm
            .ok_or(crate::LigeritoError::InvalidProof)?,
        initial_ligero_proof: proof
            .initial_ligero_proof
            .ok_or(crate::LigeritoError::InvalidProof)?,
        recursive_commitments: proof.recursive_commitments,
        recursive_proofs: proof.recursive_proofs,
        final_ligero_proof: proof
            .final_ligero_proof
            .ok_or(crate::LigeritoError::InvalidProof)?,
        sumcheck_transcript: proof
            .sumcheck_transcript
            .ok_or(crate::LigeritoError::InvalidProof)?,
        eval_rounds: Vec::new(),
    })
}
