# How to train a new model

In order to train a new model, you must have `python` installed, along with the
`keras` module. In Arch Linux, you can install `python-keras` and
`python-tensorflow` (or one of its variants) from the AUR. Then you need to do
the following:


## Create your speech and noise samples

You need to create two collections files: one should consist of samples of speech (or
whatever kind of sounds you want noise reduction for), and the other should
consist of noise samples. The various `info.txt` files in
[rnnoise-models](https://github.com/GregorR/rnnoise-models) contain some
sources of data. Note that the link to `rnnoise_contributions.tar.gz` there is
old; a better link is
`https://media.xiph.org/rnnoise/rnnoise_contributions.tar.gz`.
Another source for speech data is the [DNS Challenge](https://github.com/microsoft/DNS-Challenge).

Our speech and noise files should be 16-bit, little-endian, 48khz, 1-channel WAV files. If
you have files in other formats, you can convert them to the required format by
installing `ffmpeg` and running
```
ffmpeg -i $file -f s16le -ac 1 -ar 48000 output.wav
```

## Generate the training data

Once you have your speech and noise files, you need to use `nnnoiseless` to
generate training features:
```
cargo run --features=train --bin=train --release -- --count=<COUNT> --signal-glob=</PATH/TO/SPEECH/*.wav> --noise-glob=<PATH/TO/NOISE/*.wav> -o training.h5
```
where `<COUNT>` is the number of frames of training data that you want to
generate. (I don't know what the optimal number is, but 10 million seems to be
plenty.) The output file needs to be called `training.h5`, because that's what
the training script expects.
If you have multiple sources of signal and/or noise, you can invoke the `--signal-glob` option (or the `--noise-glob` option) multiple times.

## Train the model

With the `training.h5` script from the previous step in your current directory,
run the `train/rnn_train.py` script. For example, if you are in `nnnoiseless`'s
root directory, run
```
python train/rnn_train.py
```
This will take some time, maybe even a few days, depending on your hardware.
When it's done, there will be two output files created:

- `weights.hdf5` will contain a description of the learned model. This model can be loaded into keras if you want to fiddle with it, but probably it won't be useful to you.
- `weights.rnn` will contain the description that can be loaded into `nnnoiseless`.

Finally, you can run `nnnoiseless` with your newly learned model, by running
```
cargo run --release -- --model weights.rnn <INPUT> <OUTPUT>
```
