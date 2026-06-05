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

pub mod ast;
pub mod exit;
pub mod dictionary;
pub mod generator;
pub mod ir;
pub mod ladder;
pub mod layeredinput;
pub mod minimizer;
pub mod randomext;
