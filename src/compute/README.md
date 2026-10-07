# Compute Layer

Rust crate that trains the multilayer perceptron and writes training artifacts. It is split into a library (`src/lib.rs`) and a thin `compute` binary (`src/main.rs`) that provides the CLI.

The Rust version is pinned in `rust-toolchain.toml`.

## Commands

```bash
cargo build --release                        # build
cargo run --release -- --help                # show the CLI help
cargo test                                   # run all tests
cargo test <test_name>                       # run a single test
cargo clippy --all-targets -- -D warnings    # lint (as in CI)
cargo fmt                                    # format
```
