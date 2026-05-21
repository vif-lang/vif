# Vif

Vif is a low-level, data-oriented programming language aimed at game development.

## Requirements

- Rust, using the toolchain pinned in `rust-toolchain.toml`
- LLVM 21.1
- Windows SDK and Visual Studio 2022 Build Tools on Windows

Set `LLVM_SYS_211_PREFIX` to your LLVM install directory:

```sh
export LLVM_SYS_211_PREFIX=/path/to/llvm
```

On Windows, set the same variable in your environment.

## Build

```sh
cargo build
```

Run the compiler with LLVM enabled:

```sh
just run -- <args>
```

Run tests:

```sh
just test
```

## License

Vif uses split licensing:

- The compiler and repository files outside `std/` are licensed under GPL-3.0-only.
  See [`LICENSE`](LICENSE).
- The standard library in `std/` is licensed under the zlib License. See
  [`std/LICENSE`](std/LICENSE).

Unless a file states otherwise, files under `std/` use the standard library license
and all other files use the compiler license.
