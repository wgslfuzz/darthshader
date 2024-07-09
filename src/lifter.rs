#![feature(lazy_cell)]
#![feature(variant_count)]

extern crate link_cplusplus;

use std::{fs, path::PathBuf};

use clap::{Arg, Command};
use layeredinput::LayeredInput;
use libafl::prelude::HasTargetBytes;
use libafl_bolts::AsSlice;

mod ast;
mod generator;
mod ir; 
mod layeredinput;
mod randomext;

pub fn main() {
    let res = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author("Lukas Bernhard")
        .arg(
            Arg::new("ron")
                .short('f')
                .long("file")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf))
                .help("The ron file to lift to wgsl"),
        )
        .get_matches();

    let mut file_path: PathBuf = res.get_one("ron").cloned().unwrap();
    let input = fs::read(&file_path).expect("failed to read file");

    let mut opts = ron::Options::default();
    opts.recursion_limit = None;
    let l: LayeredInput = opts.from_bytes(&input).expect("Failed to deserialize");
    file_path.set_extension("wgsl");
    fs::write(file_path, l.target_bytes().as_slice()).expect("failed to write output file");
}
