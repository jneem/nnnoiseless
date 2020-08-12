# nnnoiseless
[![Rust](https://github.com/jneem/nnnoiseless/workflows/Rust/badge.svg)](https://github.com/jneem/nnnoiseless/actions?query=workflow%3ARust)
[![docs]( https://docs.rs/nnnoiseless/badge.svg)](https://docs.rs/nnnoiseless)

`nnnoiseless` is a rust crate for suppressing audio noise. It is a (safe) rust port of
the [`RNNoise`][1] C library, and is based on a recurrent
neural network.

While `nnnoiseless` is meant to be used as a library, a simple command-line
tool is provided as an example. It operates on WAV files or RAW 16-bit little-endian
PCM files. It can be used as:

```
cargo run --release --example nnnoiseless INPUT.wav OUTPUT.wav
```

## C-API

It is possible to install `nnnoiseless` as C-API library, with a [`RNNoise`][1]-compatible header.

``` sh
# cargo install cargo-c
# cargo cinstall --destdir /tmp/staging-nnnoiseless
# sudo cp -a /tmp/staging-nnnoiseless/* /
```

[1]: https://github.com/xiph/rnnoise
