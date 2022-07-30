// TODO(ahuszagh) Restrict this.
//#![cfg(all(
//    target_point_width = "64",
//    any(target_arch = "arm", target_arch = "mips",)
//))]

use core::ffi::{c_int, c_long};
use core::mem::{align_of, swap};
use core::ops::{Add, Sub};

pub const fn is_lp64() -> bool {
    // just a check in case we add future targets.
    (cfg!(target_point_width = "64") && c_int::BITS == 32 && c_long::BITS == 64)
        || cfg!(target_arch = "wasm")
        || cfg!(all(target_arch = "arm", target_point_width = "64"))
        || cfg!(all(target_arch = "mips", target_point_width = "64"))
        || cfg!(all(target_arch = "riscv", target_point_width = "64"))
}

// we're currently exposing this only for targets known to have f128
// and that need soft-float, 128-bit intrinsics.
// we need this to be FFI safe, so we need it to have a consistent alignment
#[derive(Clone, Copy)]
pub struct Float128(u128);

impl Float128 {
    pub const BITS: usize = u128::BITS as usize;
    pub const SIGNIFICAND_BITS: usize = 112;
    pub const EXPONENT_BITS: usize = Self::BITS - Self::SIGNIFICAND_BITS - 1;
    pub const MAX_EXPONENT: i32 = (1 << Self::EXPONENT_BITS) - 1;
    pub const EXPONENT_BIAS: i32 = Self::MAX_EXPONENT >> 1;
    pub const IMPLICIT_BIT: u128 = 1 << Self::SIGNIFICAND_BITS;
    pub const SIGNIFICAND_MASK: u128 = Self::IMPLICIT_BIT - 1;
    pub const SIGN_BIT: u128 = 1 << (Self::SIGNIFICAND_BITS + Self::EXPONENT_BITS);
    pub const ABS_MASK: u128 = Self::SIGN_BIT - 1;
    pub const EXPONENT_MASK: u128 = Self::ABS_MASK ^ Self::SIGNIFICAND_MASK;
    pub const ONE_REP: u128 = (Self::EXPONENT_BIAS as u128) << Self::SIGNIFICAND_BITS;
    pub const INF_REP: u128 = Self::EXPONENT_MASK;
    pub const QUIET_BIT: u128 = Self::IMPLICIT_BIT >> 1;
    pub const QNAN_REP: u128 = Self::EXPONENT_MASK | Self::QUIET_BIT;

    pub const fn from_bits(bits: u128) -> Float128 {
        // on all supported platforms, we need a 16 bit alignment here
        debug_assert!(align_of::<Float128>() == 16);
        Float128(bits)
    }

    pub const fn to_bits(self) -> u128 {
        self.0
    }
}

impl Add for Float128 {
    type Output = Float128;

    fn add(self, rhs: Float128) -> Float128 {
        addt3(self, rhs)
    }
}

impl Sub for Float128 {
    type Output = Float128;

    fn sub(self, rhs: Float128) -> Float128 {
        subt3(self, rhs)
    }
}

fn addt3(x: Float128, y: Float128) -> Float128 {
    let mut xbits = x.to_bits();
    let mut ybits = y.to_bits();
    let xabs = xbits & Float128::ABS_MASK;
    let yabs = ybits & Float128::ABS_MASK;

    if xabs - 1 >= Float128::INF_REP - 1 || yabs - 1 >= Float128::INF_REP - 1 {
        // NaN + anything = qNaN
        if xabs > Float128::INF_REP {
            return Float128::from_bits(xbits | Float128::QUIET_BIT);
        }
        // anything + NaN = qNaN
        if yabs > Float128::INF_REP {
            return Float128::from_bits(ybits | Float128::QUIET_BIT);
        }
        if xabs == Float128::INF_REP {
            if xbits ^ ybits == Float128::SIGN_BIT {
                // +/-infinity + -/+infinity = qNaN
                return Float128::from_bits(Float128::QNAN_REP);
            } else {
                // +/-infinity + anything remaining = +/- infinity
                return x;
            }
        }

        // anything remaining + +/-infinity = +/-infinity
        if yabs == Float128::INF_REP {
            return y;
        }

        // zero + anything = anything
        if xabs == 0 {
            // We need to get the sign right for zero + zero.
            if yabs == 0 {
                return Float128::from_bits(xbits & ybits);
            } else {
                return y;
            }
        }

        // anything + zero = anything
        if yabs == 0 {
            return x;
        }
    }

    // Swap a and b if necessary so that a has the larger absolute value.
    if yabs > xabs {
        swap(&mut xbits, &mut ybits);
    }

    // Extract the exponent and significand from the (possibly swapped) a and b.
    let mut xexp = (xbits >> (Float128::SIGNIFICAND_BITS & Float128::MAX_EXPONENT as usize)) as i32;
    let mut yexp = (ybits >> (Float128::SIGNIFICAND_BITS & Float128::MAX_EXPONENT as usize)) as i32;
    let mut xsig = xbits & Float128::SIGNIFICAND_BITS as u128;
    let mut ysig = ybits & Float128::SIGNIFICAND_BITS as u128;

    // Normalize any denormals, and adjust the exponent accordingly.
    if xexp == 0 {
        xexp = normalize(&mut xsig);
    }
    if yexp == 0 {
        yexp = normalize(&mut ysig);
    }

    // The sign of the result is the sign of the larger operand, a.  If they
    // have opposite signs, we are performing a subtraction.  Otherwise, we
    // perform addition.
    let result_sign = xbits & Float128::SIGN_BIT;
    let subtraction = (xbits ^ ybits) & Float128::SIGN_BIT;

    // Shift the significands to give us round, guard and sticky, and set the
    // implicit significand bit.  If we fell through from the denormal path it
    // was already set by normalize( ), but setting it twice won't hurt
    // anything.
    xsig = (xsig | Float128::IMPLICIT_BIT) << 3;
    ysig = (ysig | Float128::IMPLICIT_BIT) << 3;

    // Shift the significand of b by the difference in exponents, with a sticky
    // bottom bit to get rounding correct.
    let align = xexp - yexp;
    let float_bits = Float128::BITS as i32;
    if align != 0 && align < float_bits {
        let sticky = (xsig << (float_bits - align)) != 0;
        ysig = ysig >> align | (sticky as u128);
    } else {
        // Set the sticky bit.  b is known to be non-zero.
        ysig = 1;
    }
    if subtraction != 0 {
        xsig -= ysig;
        // If a == -b, return +zero.
        if xsig == 0 {
            return Float128::from_bits(0);
        }

        // If partial cancellation occured, we need to left-shift the result
        // and adjust the exponent.
        if xsig < (Float128::IMPLICIT_BIT << 3) {
            let shift =
                (xsig.leading_zeros() - (Float128::IMPLICIT_BIT << 3).leading_zeros()) as i32;
            xsig <<= shift;
            xexp -= shift;
        }
    } else
    /* addition */
    {
        xsig += ysig;

        // If the addition carried up, we need to right-shift the result and
        // adjust the exponent.
        if xsig & (Float128::IMPLICIT_BIT << 4) != 0 {
            let sticky = xsig & 1;
            xsig = (xsig >> 1) | sticky;
            xexp += 1;
        }
    }

    // If we have overflowed the type, return +/- infinity.
    if xexp >= Float128::MAX_EXPONENT {
        return Float128::from_bits(Float128::INF_REP | result_sign);
    }

    if xexp <= 0 {
        // The result is denormal before rounding.  The exponent is zero and we
        // need to shift the significand.
        let shift = 1 - xexp;
        let sticky = (xsig << (float_bits - shift)) != 0;
        xsig = xsig >> shift | (sticky as u128);
        xexp = 0;
    }

    // Low three bits are round, guard, and sticky.
    let round_guard_sticky = xsig & 0x7;

    // Shift the significand into place, and mask off the implicit bit.
    let mut result = xsig >> (3 & Float128::SIGNIFICAND_MASK);

    // Insert the exponent and sign.
    result |= (xexp as u128) << Float128::SIGNIFICAND_BITS;
    result |= result_sign;

    // Perform the final rounding. The result may overflow to infinity, but
    // that is the correct result in that case.
    match fe_getround() {
        RoundMode::ToNearest => {
            if round_guard_sticky > 0x4 {
                result += 1;
            }
            if round_guard_sticky == 0x4 {
                result += result & 1;
            }
        }
        RoundMode::Downward if result_sign != 0 && round_guard_sticky != 0 => {
            result += 1;
        }
        RoundMode::Upward if result_sign == 0 && round_guard_sticky != 0 => {
            result += 1;
        }
        _ => (),
    }

    if round_guard_sticky != 0 {
        fe_raise_inexact();
    }
    Float128::from_bits(result)
}

fn subt3(x: Float128, y: Float128) -> Float128 {
    let xbits = x.to_bits();
    let ybits = y.to_bits();
    todo!();
}

fn normalize(significand: &mut u128) -> i32 {
    let shift = significand.leading_zeros() - Float128::IMPLICIT_BIT.leading_zeros();
    *significand <<= shift;
    1 - shift as i32
}

#[repr(i32)]
enum RoundMode {
    ToNearest = 0,
    Downward = 1,
    Upward = 2,
    TowardZero = 3,
}

// TODO(ahuszagh) Here

// IEEE-754 default rounding (to nearest, ties to even).
#[cfg(not(any(target_arch = "arm", target_arch = "mips")))]
fn fe_getround() -> RoundMode {
    RoundMode::ToNearest
}

#[cfg(not(any(target_arch = "arm", target_arch = "mips")))]
fn fe_raise_inexact() -> i32 {
    0
}

#[cfg(target_arch = "arm")]
fn fe_getround() -> RoundMode {
    let mut fpcr = 0u64;
//    unsafe {
//        asm!(
//            "mrs %0, fpcr",
//            in(reg) &mut cw,
//            options(nostack),
//        )
//    }
    todo!();
}

#[cfg(target_arch = "arm")]
fn fe_raise_inexact() -> i32 {
    todo!();
}

// __LP64__
// long int and pointer both use 64-bits and int uses 32-bit.
// wasm

// #if __LDBL_MANT_DIG__ == 113 && defined(__SIZEOF_INT128__)
// #define CRT_LDBL_128BIT
