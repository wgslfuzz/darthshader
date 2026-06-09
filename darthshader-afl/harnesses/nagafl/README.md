The naga library is fuzzed by wrapping it with a separate harness:

```bash
cargo install --version 0.15.4 cargo-afl
cargo afl build --release
```
