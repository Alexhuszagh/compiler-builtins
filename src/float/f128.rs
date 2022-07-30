#![cfg(all(
    target_point_width = "64",
    any(target_arch = "arm", target_arch = "mips",)
))]

// we're currently exposing this only for targets known to have f128
// and that need soft-float, 128-bit intrinsics.

pub struct f128([u8; 16]);

// __LP64__
// long int and pointer both use 64-bits and int uses 32-bit.
// wasm

// #if __LDBL_MANT_DIG__ == 113 && defined(__SIZEOF_INT128__)
// #define CRT_LDBL_128BIT
