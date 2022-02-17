# nnnoiseless
[![Rust](https://github.com/jneem/nnnoiseless/workflows/Rust/badge.svg)](https://github.com/jneem/nnnoiseless/actions?query=workflow%3ARust)
[![docs]( https://docs.rs/nnnoiseless/badge.svg)](https://docs.rs/nnnoiseless)

`nnnoiseless` is a rust crate for suppressing audio noise. It is a rust port of
the [`RNNoise`][1] C library, and is based on a recurrent
neural network.

While `nnnoiseless` is meant to be used as a library, a simple command-line
tool is provided as an example. It operates on WAV files or RAW PCM files.
Run

```
cargo install nnnoiseless
```

to install it (you might need to install [rust](https://www.rustlang.org) first).
Once `nnnoiseless` is installed, you can run it like

```
nnnoiseless input.wav output.wav
```

or, for more advanced usage, try

```
nnnoiseless --help
```

## Safety

Except for the C API described below, `nnnoiseless` is mostly written in safe
rust. It currently uses `unsafe` in two places, to cast arrays of `f32`s to
arrays of `Complex<f32>`s with half the length; this delivers a small but
measurable performance improvement. If a future version of
[`RustFFT`](https://github.com/awelkie/RustFFT) has built-in support for
real-only FFTs, this unsafe code will be removed.

## C API

It is possible to install `nnnoiseless` as a library usable from `C`, with an
[`RNNoise`][1]-compatible header.

``` sh
$ cargo install cargo-c
$ mkdir staging-nnnoiseless
$ cargo cinstall --destdir staging-nnnoiseless
$ sudo cp -a staging-nnnoiseless/* /
```

# Custom models

`nnnoiseless` is based on a neural network. There's one built in, but you can
also swap out the built-in network for your own. (This might be useful, for
example, if you have a particular kind of noise that you want to filter out and
`nnnoiseless`'s built-in network doesn't do a good enough job.)

## Loading a `nnnoiseless` network

Let's suppose that you've already trained (or downloaded from somewhere) your
neural network weights, and that they are in the file `weights.rnn`. You can use
these weights for the `nnnoiseless` binary by passing in the `--model` option:

```
nnnoiseless --model=weights.rnn input.wav output.wav
```

On the other hand, if you're using `nnnoiseless` as a library, you can load your
neural network weights using [`RnnModel::from_bytes`] or [`RnnModel::from_static_bytes`].

## Converting an `RNNoise` network

Some people have already made their own neural network weights for `RNNoise`
(for example, [here](https://github.com/GregorR/rnnoise-models)). These
weights can be used in `nnnoiseless` also, but you'll need to first convert them
from the (text-based) `RNNoise` format to the (binary) `nnnoiseless` format. There
is a script in the `train` directory that can do this for you: just run

```
python train/convert_rnnoise.py input_file.txt output_file.rnn
```

## Training your own weights

This is a little involved, but at least it's documented now. See `train/README.md` for
more information.

[1]: https://github.com/xiph/rnnoise
[`RnnModel::from_bytes`]: https://docs.rs/nnnoiseless/latest/nnnoiseless/struct.RnnModel.html#method.from_bytes
[`RnnModel::from_static_bytes`]: https://docs.rs/nnnoiseless/latest/nnnoiseless/struct.RnnModel.html#method.from_static_bytes
