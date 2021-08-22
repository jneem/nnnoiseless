# How to train a new model

In order to train a new model, you must have `python` installed, along with the
`keras` module. In Arch Linux, you can install `python-keras` and
`python-tensorflow` (or one of its variants) from the AUR. Then you need to do
the following:


## Create your speech and noise samples

You need to create two files: one should consist of samples of speech (or
whatever kind of sounds you want noise reduction for), and the other should
consist of noise samples. The various `info.txt` files in
[rnnoise-models](https://github.com/GregorR/rnnoise-models) contain some
sources of data. Note that the link to `rnnoise_contributions.tar.gz` there is
old; a better link is
`https://media.xiph.org/rnnoise/rnnoise_contributions.tar.gz`.

Our speech and noise files should be 16-bit, little-endian raw PCM audio. If
you have files in other formats, you can convert them to the required format by
installing `ffmpeg` and running
```
ffmpeg -i $file -f s16le -ac 1 -ar 48000 output.raw
```
Then you need to combine all your speech samples into a single file, and all
your noise samples into a single file. You can do this using `cat`, or the
[rnnoise-models](https://github.com/GregorR/rnnoise-models) contains some
utilities for dividing up the samples into chunks and interleaving them into a
single file. I'm not sure what the benefit of this is.

## Generate the training data

Once you have your speech and noise files, you need to use `nnnoiseless` to
generate training features:
```
cargo run --features=train --bin=train <SIGNAL> <NOISE> <COUNT> -o training.h5
```
where `<SIGNAL>` is the file containing your speech data, `<NOISE>` is the file
containing your noise data, and `<COUNT>` is the number of frames of training
data that you want to generate. (TODO: give some guidance on a good number.
`rnnoise-models` seems to use about 10M.) The output file needs to be called
`training.h5`, because that's what the training script expects.

## Train the model

With the `training.h5` script from the previous step in your current directory,
run the `train/rnn_train.py` script. For example, if you are in `nnnoiseless`'s
root directory, run
```
python train/rnn_train.py
```
This will take some time, maybe even a few days, depending on your hardware.
The output of this step is a file called `weights.hdf5`, in your working
directory.

## Convert the output

To convert the `weights.hdf5` file from the previous step into a description
that `nnnoiseless` can use, run the `train/dump_rnn.py` script. For example,
if you've been working in `nnnoiseless`'s root directory this whole time, type
```
python train/dump_rnn.py weights.hdf5 /dev/null nn-model.txt blah
```
(The `/dev/null` and `blah` arguments above are remnants of code that was
copied from `RNNoise`, and they don't do anything interesting here. They will
probably be removed eventually.)

Finally, you can run `nnnoiseless` with your newly learned model, by running
```
cargo run --release -- --model nn-model.txt <INPUT> <OUTPUT>
```
