# nnnoiseless
[![Rust](https://github.com/jneem/nnnoiseless/workflows/Rust/badge.svg)](https://github.com/jneem/nnnoiseless/actions?query=workflow%3ARust)
[![docs]( https://docs.rs/nnnoiseless/badge.svg)](https://docs.rs/nnnoiseless)

`nnnoiseless` is a rust crate for suppressing audio noise. It is a rust port of
the [`RNNoise`][1] C library, and is based on a recurrent
neural network.

While `nnnoiseless` is meant to be used as a library, a simple command-line
tool is provided as an example. It operates on WAV files or RAW 16-bit little-endian
PCM files. It can be used as:

```
cargo run --release --example nnnoiseless INPUT.wav OUTPUT.wav
```

## Safety

Except for the C API described below, `nnnoiseless` is mostly written in safe
rust. It currently uses `unsafe` in two places, to cast arrays of `f32`s to
arrays of `Complex<f32>`s with half the length; this delivers a small but
measurable performance improvement. If a future version of
[`RustFFT`](https://github.com/awelkie/RustFFT) has built-in support for
real-only FFTs, this unsafe code will be removed.

## C-API

It is possible to install `nnnoiseless` as C-API library, with a [`RNNoise`][1]-compatible header.

``` sh
$ cargo install cargo-c
$ mkdir staging-nnnoiseless
$ cargo cinstall --destdir staging-nnnoiseless
$ sudo cp -a staging-nnnoiseless/* /
```

[1]: https://github.com/xiph/rnnoise
