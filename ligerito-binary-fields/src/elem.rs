use crate::poly::{BinaryPoly128, BinaryPoly16, BinaryPoly32, BinaryPoly64};
use crate::{BinaryFieldElement, BinaryPolynomial};

// Irreducible polynomials (matching Julia implementation)
const IRREDUCIBLE_16: u32 = 0x1002D; // x^16 + x^5 + x^3 + x^2 + 1 (need to store in larger type)
const IRREDUCIBLE_32: u64 = (1u64 << 32) | 0b11001 | (1 << 7) | (1 << 9) | (1 << 15); // x^32 + Conway polynomial

macro_rules! impl_binary_elem {
    ($name:ident, $poly_type:ident, $poly_double:ident, $value_type:ty, $value_double:ty, $irreducible:expr, $bitsize:expr) => {
        #[repr(transparent)]
        #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(
            feature = "scale",
            derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
        )]
        pub struct $name($poly_type);

        // SAFETY: $name is repr(transparent) over $poly_type which wraps $value_type (a primitive)
        unsafe impl bytemuck::Pod for $name {}
        unsafe impl bytemuck::Zeroable for $name {}

        impl $name {
            pub const fn from_value(val: $value_type) -> Self {
                Self($poly_type::new(val))
            }

            fn mod_irreducible_wide(poly: $poly_double) -> Self {
                // julia-like reduction using leading_zeros (lzcnt)
                let mut p = poly.value();
                let irr = $irreducible;
                let n = $bitsize;

                // use leading_zeros instead of loop - julia uses this (binarypoly.jl:146)
                let total_bits = core::mem::size_of::<$value_double>() * 8;

                loop {
                    if p == 0 {
                        break; // avoid underflow when p is zero
                    }

                    let lz = p.leading_zeros() as usize;
                    let high_bit = total_bits - lz - 1;

                    if high_bit < n {
                        break;
                    }

                    p ^= irr << (high_bit - n);
                }

                Self($poly_type::new(p as $value_type))
            }
        }

        impl BinaryFieldElement for $name {
            type Poly = $poly_type;

            fn zero() -> Self {
                Self($poly_type::zero())
            }

            fn one() -> Self {
                Self($poly_type::one())
            }

            fn from_poly(poly: Self::Poly) -> Self {
                // For from_poly, we assume the polynomial is already reduced
                Self(poly)
            }

            fn poly(&self) -> Self::Poly {
                self.0
            }

            fn add(&self, other: &Self) -> Self {
                Self(self.0.add(&other.0))
            }

            fn mul(&self, other: &Self) -> Self {
                // Perform full multiplication using double-width type
                let a_wide = $poly_double::from_value(self.0.value() as u64);
                let b_wide = $poly_double::from_value(other.0.value() as u64);
                let prod_wide = a_wide.mul(&b_wide);

                // Reduce modulo irreducible polynomial
                Self::mod_irreducible_wide(prod_wide)
            }

            fn inv(&self) -> Self {
                assert_ne!(self.0.value(), 0, "Cannot invert zero");

                // For binary fields, we can use Fermat's little theorem efficiently
                // a^(2^n - 2) = a^(-1) in GF(2^n)

                // For small fields, use direct exponentiation
                if $bitsize <= 16 {
                    let exp = (1u64 << $bitsize) - 2;
                    return self.pow(exp);
                }

                // For larger fields, use the addition chain method
                // 2^n - 2 = 2 + 4 + 8 + ... + 2^(n-1)

                // Start with a^2
                let mut acc = self.mul(self);
                let mut result = acc; // a^2

                // Compute a^4, a^8, ..., a^(2^(n-1)) and multiply them all
                for _ in 2..$bitsize {
                    acc = acc.mul(&acc); // Square to get next power of 2
                    result = result.mul(&acc);
                }

                result
            }

            fn pow(&self, mut exp: u64) -> Self {
                if *self == Self::zero() {
                    return Self::zero();
                }

                let mut result = Self::one();
                let mut base = *self;

                while exp > 0 {
                    if exp & 1 == 1 {
                        result = result.mul(&base);
                    }
                    base = base.mul(&base);
                    exp >>= 1;
                }

                result
            }
        }

        impl From<$value_type> for $name {
            fn from(val: $value_type) -> Self {
                Self::from_value(val)
            }
        }

        #[cfg(feature = "rand")]
        impl rand::distributions::Distribution<$name> for rand::distributions::Standard {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $name {
                $name::from_value(rng.gen())
            }
        }
    };
}

impl_binary_elem!(
    BinaryElem16,
    BinaryPoly16,
    BinaryPoly32,
    u16,
    u32,
    IRREDUCIBLE_16,
    16
);
impl_binary_elem!(
    BinaryElem32,
    BinaryPoly32,
    BinaryPoly64,
    u32,
    u64,
    IRREDUCIBLE_32,
    32
);

// BinaryElem128 needs special handling since we don't have BinaryPoly256
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct BinaryElem128(BinaryPoly128);

// SAFETY: BinaryElem128 is repr(transparent) over BinaryPoly128 which wraps u128 (a primitive)
unsafe impl bytemuck::Pod for BinaryElem128 {}
unsafe impl bytemuck::Zeroable for BinaryElem128 {}

impl BinaryElem128 {
    pub const fn from_value(val: u128) -> Self {
        Self(BinaryPoly128::new(val))
    }
}

impl BinaryFieldElement for BinaryElem128 {
    type Poly = BinaryPoly128;

    fn zero() -> Self {
        Self(BinaryPoly128::zero())
    }

    fn one() -> Self {
        Self(BinaryPoly128::one())
    }

    fn from_poly(poly: Self::Poly) -> Self {
        Self(poly)
    }

    fn poly(&self) -> Self::Poly {
        self.0
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    fn mul(&self, other: &Self) -> Self {
        // Use SIMD carryless multiplication + reduction for performance
        use crate::simd::{carryless_mul_128_full, reduce_gf128};

        let product = carryless_mul_128_full(self.0, other.0);
        let reduced = reduce_gf128(product);

        Self(reduced)
    }

    fn inv(&self) -> Self {
        assert_ne!(self.0.value(), 0, "Cannot invert zero");

        // Use Itoh-Tsujii fast inversion with precomputed nibble tables
        // Reduces from ~127 multiplications to ~9
        let result = crate::fast_inverse::invert_gf128(self.0.value());
        Self(BinaryPoly128::new(result))
    }

    fn pow(&self, mut exp: u64) -> Self {
        if *self == Self::zero() {
            return Self::zero();
        }

        let mut result = Self::one();
        let mut base = *self;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        result
    }
}

impl BinaryElem128 {
    /// Multiply by x (field element 2) - very fast special case
    ///
    /// In GF(2^128) with irreducible x^128 + x^7 + x^2 + x + 1,
    /// multiplying by x is just a left shift with conditional reduction.
    /// This is ~10x faster than general multiplication.
    #[inline]
    pub fn mul_by_x(&self) -> Self {
        let val = self.0.value();

        // Shift left by 1 (multiply by x in polynomial ring)
        let shifted = val << 1;

        // If bit 128 would be set (overflow), reduce by the irreducible polynomial
        // x^128 = x^7 + x^2 + x + 1 (mod irreducible)
        // So we add 0x87 if the high bit was set
        let overflow = (val >> 127) & 1;
        let reduced = shifted ^ (overflow * 0x87);

        Self(BinaryPoly128::new(reduced))
    }
}

impl From<u128> for BinaryElem128 {
    fn from(val: u128) -> Self {
        Self::from_value(val)
    }
}

#[cfg(feature = "rand")]
impl rand::distributions::Distribution<BinaryElem128> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BinaryElem128 {
        BinaryElem128::from_value(rng.gen())
    }
}

// BinaryElem64 needs special handling
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "scale",
    derive(codec::Encode, codec::Decode, scale_info::TypeInfo)
)]
pub struct BinaryElem64(BinaryPoly64);

// SAFETY: BinaryElem64 is repr(transparent) over BinaryPoly64 which wraps u64 (a primitive)
unsafe impl bytemuck::Pod for BinaryElem64 {}
unsafe impl bytemuck::Zeroable for BinaryElem64 {}

impl BinaryElem64 {
    pub const fn from_value(val: u64) -> Self {
        Self(BinaryPoly64::new(val))
    }
}

impl BinaryFieldElement for BinaryElem64 {
    type Poly = BinaryPoly64;

    fn zero() -> Self {
        Self(BinaryPoly64::zero())
    }

    fn one() -> Self {
        Self(BinaryPoly64::one())
    }

    fn from_poly(poly: Self::Poly) -> Self {
        // For now, no reduction for 64-bit field
        Self(poly)
    }

    fn poly(&self) -> Self::Poly {
        self.0
    }

    fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    fn mul(&self, other: &Self) -> Self {
        Self(self.0.mul(&other.0))
    }

    fn inv(&self) -> Self {
        assert_ne!(self.0.value(), 0, "Cannot invert zero");
        // Fermat's little theorem: a^(2^64 - 2) = a^(-1)
        self.pow(0xFFFFFFFFFFFFFFFE)
    }

    fn pow(&self, mut exp: u64) -> Self {
        if *self == Self::zero() {
            return Self::zero();
        }

        let mut result = Self::one();
        let mut base = *self;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }

        result
    }
}

// Subfield embeddings.
//
// The fields here use unrelated polynomial bases, so copying bits from a
// smaller field into a larger one is NOT a ring homomorphism. Each
// `From` below is the genuine embedding φ(Σ aᵢxⁱ) = Σ aᵢγⁱ, where γ is a root
// of the smaller field's modulus inside the larger field (found by
// Cantor–Zassenhaus; `test_embeddings_are_homomorphisms` checks them).
// φ₁₆→₁₂₈ = φ₃₂→₁₂₈ ∘ φ₁₆→₃₂, so conversions commute.

const EMBED_16_IN_32_BASIS: [u32; 16] = [
    0x1,
    0xdcbf24b2,
    0xf491ccf4,
    0x54da6400,
    0x61a14d8e,
    0xecb96800,
    0x18ebb52f,
    0xb5c4f88b,
    0x4164c846,
    0x831a6455,
    0x188bc04c,
    0x4f33263e,
    0x50a6847a,
    0x792a7847,
    0x7a8bf7a3,
    0x6ae6d059,
];

const EMBED_32_IN_128_BASIS: [u128; 32] = [
    0x1,
    0xc2ad7d2a889a2e625dafe1c0f5fa0b0f,
    0xa36eb16896346ffdb4f0319a37e37a16,
    0x2ded3bb0e67b64c08fe95a442f57c746,
    0x9b81128f5e8c0ca5487bc77f5244124a,
    0x453b06a6bbba9cae098fcead3a21411f,
    0x7549d174a93c4b3ba233d944344ca11a,
    0x106977258f9fcf5780fb672275932851,
    0x743ad5c2d528fe76cd89f9a02897d4d4,
    0xf39b97d42ba79c45ea4f46f71a6ffbe5,
    0x78b4f80e5a1b921e7d3c7cfc7e92f2c1,
    0x7dad48d6664292573077d59f33e220e5,
    0xa7fb546b66ccf343f855a2e81758bc29,
    0xef1668428c4409562407596b1d41c294,
    0xc70a1808f5fc10d2fe1b2d079f63ecea,
    0x942926f037c3d59f3da95fa76909463,
    0xb323f8b46a9ff6353bb57cfe51b6b54a,
    0x87b2f19ab2f583d4cc57c2ddc6e46f14,
    0x7dfe742f5dd66a673e3cc29f3bfd39b7,
    0xaa21950a5d4bd0f3ca22bd3e6b742a76,
    0xdfb2560b1f907ef4c188e57e1018f99b,
    0xb7a85c215d1e57ef28e769bc080fd32f,
    0xc714e02a00f87c33636126394114accc,
    0x6c95b3decd5217e0972c9fba299fc505,
    0x83801942bb7e1b11675088db6cc367b3,
    0x28b3450906ee81627f22db82fde50bca,
    0x833b8d735c896678e8e96871714ba92d,
    0xaefd6cc1a76c85aec2d594b4d5e7b9cd,
    0xefbf2299a29121fe628916b5d3797507,
    0xe2ff39c69274e0af2ec56f0a9e3b0544,
    0x744a23790407126f178583b2f32fc242,
    0x54391c6205f7edcab410f02f4eee7a8d,
];

const EMBED_16_IN_128_BASIS: [u128; 16] = [
    0x1,
    0x9e1723a53bc326755304e1b0b798bb58,
    0x7da98df94ab991f5fd67f3d95323e2d0,
    0x11c665b8e43898730975d6f878ae17e7,
    0x9745e973bc195b0e42fa7e6313a877c1,
    0xa7760bb9cf936f32c2d393193ddcd37d,
    0x87efaf7735c02fabb132feb567dfbd8e,
    0x8b9c7bf0d9ad907a0ecd5861326456e6,
    0x5de72d989819fdbd73b5e20fc14e6bca,
    0x102588650a6ef99392071edacf1bbff5,
    0x86e5b069627d7b2f9e5b14c1162c9d24,
    0x589979c7388d8569af70fe63b577608b,
    0x8fc1820ca275a4f887cbe1577259af68,
    0x288c04d554f43b7d898cfa38f21b9882,
    0xe32e3eebd9303f281360e106561e8caf,
    0x5c089f50c9e1de153a432d996f7f29f3,
];

/// Byte-sliced lookup tables for a GF(2)-linear map given by basis images.
const fn byte_tables_u128<const B: usize, const N: usize>(basis: &[u128; N]) -> [[u128; 256]; B] {
    let mut out = [[0u128; 256]; B];
    let mut byte = 0;
    while byte < B {
        let mut v: usize = 1;
        while v < 256 {
            // v = lowbit + rest, both already filled
            let low = v & v.wrapping_neg();
            let bit = low.trailing_zeros() as usize;
            out[byte][v] = out[byte][v ^ low] ^ basis[byte * 8 + bit];
            v += 1;
        }
        byte += 1;
    }
    out
}

const fn byte_tables_u32<const B: usize, const N: usize>(basis: &[u32; N]) -> [[u32; 256]; B] {
    let mut out = [[0u32; 256]; B];
    let mut byte = 0;
    while byte < B {
        let mut v: usize = 1;
        while v < 256 {
            let low = v & v.wrapping_neg();
            let bit = low.trailing_zeros() as usize;
            out[byte][v] = out[byte][v ^ low] ^ basis[byte * 8 + bit];
            v += 1;
        }
        byte += 1;
    }
    out
}

static EMBED_32_IN_128: [[u128; 256]; 4] = byte_tables_u128(&EMBED_32_IN_128_BASIS);
static EMBED_16_IN_128: [[u128; 256]; 2] = byte_tables_u128(&EMBED_16_IN_128_BASIS);
static EMBED_16_IN_32: [[u32; 256]; 2] = byte_tables_u32(&EMBED_16_IN_32_BASIS);

impl From<BinaryElem16> for BinaryElem32 {
    #[inline]
    fn from(elem: BinaryElem16) -> Self {
        let v = elem.0.value();
        BinaryElem32::from(
            EMBED_16_IN_32[0][(v & 0xff) as usize] ^ EMBED_16_IN_32[1][(v >> 8) as usize],
        )
    }
}

/// Bit copy, NOT a field embedding: `BinaryElem64` multiplication is a
/// truncated carry-less product, so it is not a field.
impl From<BinaryElem16> for BinaryElem64 {
    fn from(elem: BinaryElem16) -> Self {
        BinaryElem64(BinaryPoly64::new(elem.0.value() as u64))
    }
}

impl From<BinaryElem16> for BinaryElem128 {
    #[inline]
    fn from(elem: BinaryElem16) -> Self {
        let v = elem.0.value();
        BinaryElem128::from_value(
            EMBED_16_IN_128[0][(v & 0xff) as usize] ^ EMBED_16_IN_128[1][(v >> 8) as usize],
        )
    }
}

/// Bit copy, NOT a field embedding (see `From<BinaryElem16> for BinaryElem64`).
impl From<BinaryElem32> for BinaryElem64 {
    fn from(elem: BinaryElem32) -> Self {
        BinaryElem64(BinaryPoly64::new(elem.0.value() as u64))
    }
}

impl From<BinaryElem32> for BinaryElem128 {
    #[inline]
    fn from(elem: BinaryElem32) -> Self {
        let v = elem.0.value();
        BinaryElem128::from_value(
            EMBED_32_IN_128[0][(v & 0xff) as usize]
                ^ EMBED_32_IN_128[1][((v >> 8) & 0xff) as usize]
                ^ EMBED_32_IN_128[2][((v >> 16) & 0xff) as usize]
                ^ EMBED_32_IN_128[3][(v >> 24) as usize],
        )
    }
}

/// Bit copy, NOT a field embedding (see `From<BinaryElem16> for BinaryElem64`).
impl From<BinaryElem64> for BinaryElem128 {
    fn from(elem: BinaryElem64) -> Self {
        BinaryElem128::from(elem.0.value() as u128)
    }
}
