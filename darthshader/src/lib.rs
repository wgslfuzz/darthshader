#![feature(variant_count)]
#![deny(
    clippy::correctness,
    clippy::cast_possible_wrap,
    unused_lifetimes,
    unused_unsafe,
    single_use_lifetimes,
    missing_debug_implementations
)]
#![recursion_limit = "256"]

extern crate link_cplusplus;

// `libafl` re-exports the derive macros from `libafl_derive`, so `libafl_derive`
// has to be resolvable whenever `libafl`'s metadata is loaded. Build systems
// that pass every dependency to rustc explicitly (rather than letting it search
// a directory) only hand a proc macro to its direct dependents, so name it here
// to make it one of ours.
extern crate libafl_derive;

pub mod ast;
pub mod dictionary;
pub mod exit;
pub mod generator;
pub mod ir;
pub mod ladder;
pub mod layeredinput;
pub mod minimizer;
pub mod randomext;
