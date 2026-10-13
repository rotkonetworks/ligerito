//! Soundness accounting for a [`VerifierConfig`].
//!
//! Every error term of the (interactive) protocol is listed per level with its
//! formula, so the claimed security level is auditable rather than implied by
//! a query constant. Terms are combined with a union bound. Fiat–Shamir
//! security of the non-interactive proof additionally needs round-by-round
//! soundness of each term in this table; that argument is not reproduced here.
//!
//! Analysis version: [`ANALYSIS_VERSION`]. It is a working accounting, not a
//! reviewed proof: where a term's constant is not pinned to a theorem it is
//! marked "constant to confirm" below.
//!
//! # Terms (level `t`, message length `m`, block length `n = 4m`, `2^k`
//! columns, `Q` opened rows, challenge field size `|F| = 2^128`)
//!
//! | term | error | source |
//! |---|---|---|
//! | query | `(1 − δ/2)^Q`, `δ = 1 − ρ` (0 if `Q ≥ n`) | Ligero row test in the unique-decoding regime (AHIV17; the 148-query choice of the Ligerito paper) |
//! | proximity gap | `k · n / |F|` | random tensor combination of the columns, unique-decoding regime (BCIKS20 / DP24); constant to confirm |
//! | row batching | `(Q − 1) / |F|` | the `Q` row checks are combined with powers of `α` (Schwartz–Zippel, degree `Q − 1`) |
//! | sumcheck | `2k / |F|` | `k` rounds of a degree-2 product sumcheck |
//! | glue | `1 / |F|` | claims merged with a random `β` (levels ≥ 1, and evaluation claims) |
//!
//! The initial challenges `ρ_0` are drawn from GF(2^128). Drawing them from
//! GF(2^32), as versions before 0.7 did, would make the level-0 proximity-gap
//! term `k · n / 2^32` — roughly 2^-12 at 2^20 — and the weakest line here.

#[cfg(feature = "std")]
use crate::VerifierConfig;

/// Identifies the accounting below; bump it whenever a term changes.
pub const ANALYSIS_VERSION: &str = "ligerito-udr-v1-draft";

/// Rows opened per level of every shipped configuration.
pub const DEFAULT_QUERIES: usize = 148;

/// Reed–Solomon `log2(1/ρ)` used by every level today.
pub const LOG_INV_RATE: usize = 2;

/// Bits of the field challenges are drawn from.
pub const CHALLENGE_FIELD_BITS: u32 = 128;

/// Smallest `Q` with `(1 − δ/2)^Q ≤ 2^-100` for `δ = 1 − ρ` (asymptotic
/// RS distance; the true distance is slightly larger, so this is
/// conservative). Indexed by `log2(1/ρ)`.
const QUERIES_100_BITS: [usize; 5] = [usize::MAX, 241, 148, 121, 110];

/// Queries needed for 100-bit soundness of one Ligero row test at rate
/// `2^-log_inv_rate` in the unique-decoding regime.
pub fn queries_for_100_bits(log_inv_rate: usize) -> Option<usize> {
    QUERIES_100_BITS.get(log_inv_rate).copied().filter(|&q| q != usize::MAX)
}

#[cfg(feature = "std")]
/// One level's error terms, each as `-log2(error)` (∞ when the term is 0).
#[derive(Clone, Debug)]
pub struct LevelSecurity {
    /// 0 is the initial commitment.
    pub level: usize,
    pub log_message_len: usize,
    pub log_columns: usize,
    pub queries: usize,
    pub query_bits: f64,
    pub proximity_gap_bits: f64,
    pub row_batching_bits: f64,
    pub sumcheck_bits: f64,
    pub glue_bits: f64,
}

#[cfg(feature = "std")]
impl LevelSecurity {
    /// Union bound over this level's terms.
    pub fn bits(&self) -> f64 {
        union_bits(&[
            self.query_bits,
            self.proximity_gap_bits,
            self.row_batching_bits,
            self.sumcheck_bits,
            self.glue_bits,
        ])
    }
}

#[cfg(feature = "std")]
/// Per-level accounting for a configuration.
#[derive(Clone, Debug)]
pub struct SecurityReport {
    pub analysis_version: &'static str,
    pub levels: Vec<LevelSecurity>,
}

#[cfg(feature = "std")]
impl SecurityReport {
    /// Union bound over all levels.
    pub fn total_bits(&self) -> f64 {
        union_bits(&self.levels.iter().map(|l| l.bits()).collect::<Vec<_>>())
    }

    /// The single weakest term and where it is.
    pub fn weakest(&self) -> (usize, &'static str, f64) {
        let mut w = (0, "", f64::INFINITY);
        for l in &self.levels {
            for (name, b) in [
                ("query", l.query_bits),
                ("proximity gap", l.proximity_gap_bits),
                ("row batching", l.row_batching_bits),
                ("sumcheck", l.sumcheck_bits),
                ("glue", l.glue_bits),
            ] {
                if b < w.2 {
                    w = (l.level, name, b);
                }
            }
        }
        w
    }
}

#[cfg(feature = "std")]
/// `-log2(Σ 2^-b_i)`.
fn union_bits(bits: &[f64]) -> f64 {
    let sum: f64 = bits.iter().map(|&b| (-b).exp2()).sum();
    if sum == 0.0 {
        f64::INFINITY
    } else {
        -sum.log2()
    }
}

#[cfg(feature = "std")]
/// `-log2(x / |F|)` for a challenge set of `2^field_bits` elements.
fn over_field(x: f64, field_bits: u32) -> f64 {
    if x <= 0.0 {
        f64::INFINITY
    } else {
        field_bits as f64 - x.log2()
    }
}

#[cfg(feature = "std")]
/// Accounting for `config` (assumed well formed).
pub fn analyze(config: &VerifierConfig) -> SecurityReport {
    analyze_with_challenge_bits(config, CHALLENGE_FIELD_BITS)
}

/// [`analyze`] with challenges drawn from a set of `2^bits` elements — e.g.
/// 64 to assess transcripts that only filled the low half of each GF(2^128)
/// challenge (SHA-256 and BLAKE2b transcripts before 0.8).
#[cfg(feature = "std")]
pub fn analyze_with_challenge_bits(config: &VerifierConfig, bits: u32) -> SecurityReport {
    let rho = (-(LOG_INV_RATE as f64)).exp2();
    let delta = 1.0 - rho;
    let per_query = -(1.0 - delta / 2.0).log2();

    // level t commits a message of length 2^log_m in 2^k columns; the
    // sumcheck after it folds the next level's k columns
    let mut levels = Vec::with_capacity(config.recursive_steps + 1);
    for t in 0..=config.recursive_steps {
        let (log_m, k) = if t == 0 {
            (config.initial_dim, config.initial_k)
        } else {
            (config.log_dims[t - 1], config.ks[t - 1])
        };
        let n = (1u64 << (log_m + LOG_INV_RATE)) as f64;
        let q = config.num_queries.min(n as usize);
        let sumcheck_rounds = config.ks.get(t).copied().unwrap_or(0);
        levels.push(LevelSecurity {
            level: t,
            log_message_len: log_m,
            log_columns: k,
            queries: q,
            query_bits: if q as f64 >= n { f64::INFINITY } else { q as f64 * per_query },
            proximity_gap_bits: over_field(k as f64 * n, bits),
            row_batching_bits: over_field(q as f64 - 1.0, bits),
            sumcheck_bits: over_field(2.0 * sumcheck_rounds as f64, bits),
            glue_bits: if t == 0 { f64::INFINITY } else { over_field(1.0, bits) },
        });
    }
    SecurityReport {
        analysis_version: ANALYSIS_VERSION,
        levels,
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for SecurityReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "analysis {}", self.analysis_version)?;
        writeln!(f, "level  log_m  k    Q    query  prox-gap  batch  sumcheck  glue   level-bits")?;
        for l in &self.levels {
            writeln!(
                f,
                "{:<6} {:<6} {:<4} {:<4} {:>6.1} {:>8.1} {:>6.1} {:>8.1} {:>6.1} {:>10.2}",
                l.level,
                l.log_message_len,
                l.log_columns,
                l.queries,
                l.query_bits,
                l.proximity_gap_bits,
                l.row_batching_bits,
                l.sumcheck_bits,
                l.glue_bits,
                l.bits()
            )?;
        }
        let (lvl, term, bits) = self.weakest();
        write!(
            f,
            "total {:.2} bits (union bound); weakest: level {lvl} {term} at {bits:.1} bits",
            self.total_bits()
        )
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn query_table_matches_formula() {
        // the Ligerito paper's 148 at rate 1/4, bolt-rs's 241 at rate 1/2
        for (log_inv_rate, q) in [(1, 241), (2, 148), (3, 121), (4, 110)] {
            let rho = (-(log_inv_rate as f64)).exp2();
            let per = -(1.0 - (1.0 - rho) / 2.0).log2();
            assert_eq!((100.0 / per).ceil() as usize, q, "log_inv_rate {log_inv_rate}");
            assert_eq!(queries_for_100_bits(log_inv_rate), Some(q));
        }
        assert_eq!(queries_for_100_bits(0), None);
    }

    /// The pre-0.8 SHA-256/BLAKE2b transcripts drew 64-bit challenges.
    #[test]
    fn half_width_challenges_report() {
        for (name, cfg) in [
            ("20", crate::hardcoded_config_20_verifier()),
            ("24", crate::hardcoded_config_24_verifier()),
            ("30", crate::hardcoded_config_30_verifier()),
        ] {
            let r = analyze_with_challenge_bits(&cfg, 64);
            std::println!("config {name}, 64-bit challenges\n{r}\n");
            assert!(r.total_bits() < 50.0);
        }
    }

    #[test]
    fn reports_shipped_configs() {
        for (name, cfg) in [
            ("12", crate::hardcoded_config_12_verifier()),
            ("20", crate::hardcoded_config_20_verifier()),
            ("24", crate::hardcoded_config_24_verifier()),
            ("30", crate::hardcoded_config_30_verifier()),
        ] {
            let r = analyze(&cfg);
            std::println!("config {name}\n{r}\n");
            // one 148-query row test per level is ~100.35 bits; levels add
            // under the union bound, so the total sits slightly below 100
            for l in &r.levels {
                assert!(l.query_bits > 100.0, "config {name} level {}", l.level);
                assert!(l.bits() > 99.0, "config {name} level {}", l.level);
            }
            assert!(r.total_bits() > 97.5, "config {name}: {}", r.total_bits());
        }
    }
}
