use std::path::PathBuf;

fn build_wgsl() {
    let dir: PathBuf = ["tree-sitter-wgsl", "src"].iter().collect();

    println!("cargo:rerun-if-changed={}", dir.to_str().unwrap());

    cc::Build::new()
        .include(&dir)
        .file(dir.join("parser.c"))
        .file(dir.join("scanner.cc"))
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .compile("tree-sitter-wgsl");
}

fn main() {
    build_wgsl();
}
