#[cfg(feature = "prover")]
use crate::data_structures::RecursiveLigeroWitness;
use binary_fields::BinaryFieldElement;
use merkle_tree::{build_merkle_tree, Hash};
#[cfg(all(feature = "prover", feature = "parallel"))]
use rayon::prelude::*;
#[cfg(feature = "prover")]
use reed_solomon::ReedSolomon;
use sha2::{Digest, Sha256};

/// Lay out `poly` as an `m × n` message matrix padded to `m · inv_rate` rows.
///
/// Row-contiguous: `mat[i][c] = poly[i * n + c]`. The column index `c` is the
/// low `log n` bits of the coefficient index, so combining a row with
/// `eq(ρ, ·)` equals `partial_eval_multilinear(poly, ρ)` (lsb-first binding).
#[cfg(feature = "prover")]
pub fn poly2mat<F: BinaryFieldElement>(
    poly: &[F],
    m: usize,
    n: usize,
    inv_rate: usize,
) -> Vec<Vec<F>> {
    assert_eq!(poly.len(), m * n, "poly length must equal m * n");
    let mut mat = vec![vec![F::zero(); n]; m * inv_rate];
    for (row, chunk) in mat.iter_mut().zip(poly.chunks_exact(n)) {
        row.copy_from_slice(chunk);
    }
    mat
}

/// Reed-Solomon encode every column of `poly_mat` (non-systematic encoding,
/// so the generator row for query `q` is `X_i(from_bits(q))`; see
/// `utils::evaluate_scaled_basis_inplace`).
#[cfg(feature = "prover")]
pub fn encode_cols<F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static>(
    poly_mat: &mut Vec<Vec<F>>,
    rs: &ReedSolomon<F>,
    parallel: bool,
) {
    let n = poly_mat[0].len();
    let encode_col = |j: usize| {
        let mut col: Vec<F> = poly_mat.iter().map(|row| row[j]).collect();
        // sequential FFT within each column task to avoid nested parallelism
        reed_solomon::encode_non_systematic_with_parallel(rs, &mut col, false);
        col
    };

    #[cfg(feature = "parallel")]
    let cols: Vec<Vec<F>> = if parallel {
        (0..n).into_par_iter().map(encode_col).collect()
    } else {
        (0..n).map(encode_col).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let cols: Vec<Vec<F>> = {
        let _ = parallel;
        (0..n).map(encode_col).collect()
    };

    for (i, row) in poly_mat.iter_mut().enumerate() {
        for (j, col) in cols.iter().enumerate() {
            row[j] = col[i];
        }
    }
}

/// Hash a row of field elements with deterministic serialization
/// Optimized: hashes entire row at once instead of element-by-element
#[inline(always)]
pub fn hash_row<F: BinaryFieldElement>(row: &[F]) -> Hash {
    let mut hasher = Sha256::new();

    // Hash row length for domain separation
    hasher.update((row.len() as u32).to_le_bytes());

    // Hash entire row at once using bytemuck for zero-copy byte slice
    // This is much faster than per-element hashing
    let row_bytes = unsafe {
        core::slice::from_raw_parts(row.as_ptr() as *const u8, core::mem::size_of_val(row))
    };
    hasher.update(row_bytes);

    hasher.finalize().into()
}

#[cfg(feature = "prover")]
pub fn ligero_commit<F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static>(
    poly: &[F],
    m: usize,
    n: usize,
    rs: &ReedSolomon<F>,
) -> RecursiveLigeroWitness<F> {
    let mut poly_mat = poly2mat(poly, m, n, 4);

    // Direct call - no backend abstraction overhead
    encode_cols(&mut poly_mat, rs, true);

    // Parallelize row hashing
    #[cfg(feature = "parallel")]
    let hashed_rows: Vec<Hash> = poly_mat.par_iter().map(|row| hash_row(row)).collect();

    #[cfg(not(feature = "parallel"))]
    let hashed_rows: Vec<Hash> = poly_mat.iter().map(|row| hash_row(row)).collect();

    let tree = build_merkle_tree(&hashed_rows);

    RecursiveLigeroWitness {
        mat: poly_mat,
        tree,
    }
}
