set shell := ["cmd", "/c"]

run *args:
    cargo run --no-default-features --features llvm {{ args }}

test *args:
    cargo test {{ args }}
