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
    #[cfg(feature = "op-count")]
    crate::op_count::bump_row_hash();
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

/// Commit to `poly` laid out as an `m × n` message matrix (see `poly2mat`):
/// RS-encode every column to length `4m` and Merkle-commit the rows.
#[cfg(feature = "prover")]
pub fn ligero_commit<F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static>(
    poly: &[F],
    m: usize,
    n: usize,
    rs: &ReedSolomon<F>,
) -> RecursiveLigeroWitness<F> {
    assert_eq!(poly.len(), m * n, "poly length must equal m * n");
    let block = rs.block_length();
    assert_eq!(rs.message_length(), m);

    // Column-major buffer; the parity region stays zero before encoding.
    // zeroed_vec gets zero pages from the allocator instead of writing the
    // whole buffer up front (F is Pod, so all-zero bits is F::zero()).
    let mut cols: Vec<F> = bytemuck::zeroed_vec(n * block);

    // Fill: tiles of message rows in parallel. Each tile reads its rows
    // contiguously and writes a disjoint range of every column.
    const ROWS_PER_TILE: usize = 4096;
    struct ColsPtr<F>(*mut F);
    // SAFETY: tiles write disjoint index sets (row range × all columns)
    unsafe impl<F> Send for ColsPtr<F> {}
    unsafe impl<F> Sync for ColsPtr<F> {}
    let dst = ColsPtr(cols.as_mut_ptr());
    let fill_tile = |(t, tile): (usize, &[F])| {
        let r0 = t * ROWS_PER_TILE;
        let p = &dst;
        for (i, row) in tile.chunks_exact(n).enumerate() {
            for (c, &v) in row.iter().enumerate() {
                // SAFETY: c * block + r0 + i < n * block, unique per (tile, i, c)
                unsafe { *p.0.add(c * block + r0 + i) = v };
            }
        }
    };
    #[cfg(feature = "parallel")]
    poly.par_chunks(ROWS_PER_TILE * n).enumerate().for_each(fill_tile);
    #[cfg(not(feature = "parallel"))]
    poly.chunks(ROWS_PER_TILE * n).enumerate().for_each(fill_tile);

    // Encode: one task per column; with fewer columns than threads, let each
    // column's FFT fork as well.
    #[cfg(feature = "parallel")]
    {
        let inner = n < rayon::current_num_threads();
        cols.par_chunks_mut(block)
            .for_each(|col| reed_solomon::encode_non_systematic_with_parallel(rs, col, inner));
    }
    #[cfg(not(feature = "parallel"))]
    for col in cols.chunks_mut(block) {
        reed_solomon::encode_non_systematic_with_parallel(rs, col, false);
    }

    RecursiveLigeroWitness::from_columns(cols, n)
}

#[cfg(feature = "prover")]
impl<F: BinaryFieldElement + Send + Sync + bytemuck::Pod> RecursiveLigeroWitness<F> {
    /// Merkle-commit the rows of a column-major matrix with `num_cols` columns.
    pub fn from_columns(cols: Vec<F>, num_cols: usize) -> Self {
        assert!(num_cols > 0 && cols.len() % num_cols == 0);
        let num_rows = cols.len() / num_cols;

        // Hash rows in tiles: gather a tile of rows into a small buffer
        // (contiguous reads from every column), then hash each row.
        const TILE: usize = 256;
        let hash_tile = |(t, out): (usize, &mut [Hash])| {
            let r0 = t * TILE;
            let rows = out.len();
            let mut buf = vec![F::zero(); rows * num_cols];
            for c in 0..num_cols {
                let col = &cols[c * num_rows + r0..c * num_rows + r0 + rows];
                for (i, &v) in col.iter().enumerate() {
                    buf[i * num_cols + c] = v;
                }
            }
            for (h, row) in out.iter_mut().zip(buf.chunks_exact(num_cols)) {
                *h = hash_row(row);
            }
        };
        let mut hashes = vec![Hash::default(); num_rows];
        #[cfg(feature = "parallel")]
        hashes.par_chunks_mut(TILE).enumerate().for_each(hash_tile);
        #[cfg(not(feature = "parallel"))]
        hashes.chunks_mut(TILE).enumerate().for_each(hash_tile);

        let tree = build_merkle_tree(&hashes);
        RecursiveLigeroWitness {
            cols,
            num_rows,
            num_cols,
            tree,
        }
    }
}
