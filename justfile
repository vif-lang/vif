set windows-shell := ["cmd.exe", "/c"]

# default task, runs linting, formatting, and tests
all: lint fmt test

# check for linting issues
lint:
	cargo clippy --workspace --all-targets --all-features --locked -- -Dwarnings

# check formatting
fmt:
	cargo fmt --all -- --check

# run tests
test *args:
	cargo nextest run --workspace --all-features --locked --profile ci {{ args }}

# automatically fix lint and formatting issues
fix:
	cargo fmt --all
	cargo clippy --workspace --all-targets --all-features --locked --fix --allow-dirty --allow-no-vcs

# run vifc in debug
run *args:
	cargo run --no-default-features --features llvm {{ args }}

# build a release binary
release:
	cargo build --release --no-default-features --features llvm --config .cargo/release.toml
