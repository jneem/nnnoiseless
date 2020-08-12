use std::borrow::Cow;

use crate::util::{inner_p_bytes as inner_p, relu, sigmoid_approx, tansig_approx};

const MAX_NEURONS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
}

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

#[derive(Clone)]
pub struct DenseLayer {
    /// An array of length `nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct GruLayer {
    /// An array of length `3 * nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `3 * nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    /// An array of length `3 * nb_neurons^2`.
    pub recurrent_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

/// An `RnnModel` contains all the model parameters for the denoising algorithm.
/// `nnnoiseless` has a built-in model that should work for most purposes, but if you have
/// specific needs then you might benefit from training a custom model. Scripts for model
/// training are available as part of [`RNNoise`]; once the model is trained, you can load it
/// here.
///
/// [`RNNoise`]: https://github.com/xiph/rnnoise
#[derive(Clone)]
pub struct RnnModel {
    pub(crate) input_dense: DenseLayer,
    pub(crate) vad_gru: GruLayer,
    pub(crate) noise_gru: GruLayer,
    pub(crate) denoise_gru: GruLayer,
    pub(crate) denoise_output: DenseLayer,
    pub(crate) vad_output: DenseLayer,
}

#[derive(Clone)]
pub struct RnnState<'model> {
    model: Cow<'model, RnnModel>,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
}

#[derive(Debug)]
pub enum ReadModelError {
    IntParse(std::num::ParseIntError),
    Io(std::io::Error),
    CorruptFile,
}

impl From<std::num::ParseIntError> for ReadModelError {
    fn from(e: std::num::ParseIntError) -> ReadModelError {
        ReadModelError::IntParse(e)
    }
}

impl From<std::io::Error> for ReadModelError {
    fn from(e: std::io::Error) -> ReadModelError {
        ReadModelError::Io(e)
    }
}

impl std::fmt::Display for ReadModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ReadModelError::IntParse(e) => write!(f, "error parsing i8: {}", e),
            ReadModelError::Io(e) => write!(f, "{}", e),
            ReadModelError::CorruptFile => write!(f, "model file corrupted"),
        }
    }
}

impl std::error::Error for ReadModelError {}

impl RnnModel {
    /// Reads an `RnnModel` from a `std::io::Read`.
    ///
    /// The file format of an `RnnModel` is not specified anywhere; it should have been generated
    /// from the `dump_rnn.py` script in the [`RNNoise` repository].
    ///
    /// [`RNNoise`]: https://github.com/xiph/rnnoise
    pub fn from_read<R: std::io::Read>(mut r: R) -> Result<RnnModel, ReadModelError> {
        let mut data = String::new();
        r.read_to_string(&mut data)?;

        let header = "rnnoise-nu model file version 1";
        if !data.starts_with(header) {
            return Err(ReadModelError::CorruptFile);
        }

        let data = &data[header.len()..];

        // After the header, the model file consists of a giant list of whitespace-separated
        // integers. This list consists of the data for the various layers, one after the other.
        // The format for a single dense layer is:
        // <nb_neurons> <nb_inputs> <activation>
        // <weights...>
        // <bias...>
        // where each of the <?> terms represents a single integer, and each of the <?...> terms
        // represents an array of integers of the appropriate length (as determined by nb_neurons
        // and nb_inputs).
        //
        // The format for GRU layer is similar, but with some extra recurrent weights.
        let mut ints = data
            .split_whitespace()
            .map(|s| s.parse::<i8>().map_err(|e| e.into()));

        let mut read_int = || {
            ints.next()
                .ok_or(ReadModelError::CorruptFile)
                .and_then(|x| x)
        };

        fn read_array(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
            rows: usize,
            cols: usize,
        ) -> Result<Vec<i8>, ReadModelError> {
            // The array is stored in column-major format, and we want to convert it to row-major.
            let mut ret = vec![0; rows * cols];
            for j in 0..cols {
                for i in 0..rows {
                    ret[i * cols + j] = int()?;
                }
            }
            Ok(ret)
        }

        fn act(x: i8) -> Result<Activation, ReadModelError> {
            match x {
                0 => Ok(Activation::Tanh),
                1 => Ok(Activation::Sigmoid),
                2 => Ok(Activation::Relu),
                _ => Err(ReadModelError::CorruptFile),
            }
        }

        fn unsigned(x: i8) -> Result<usize, ReadModelError> {
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err(ReadModelError::CorruptFile)
            }
        }

        fn read_dense(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
        ) -> Result<DenseLayer, ReadModelError> {
            let nb_inputs = unsigned(int()?)?;
            let nb_neurons = unsigned(int()?)?;
            let activation = act(int()?)?;
            let input_weights = read_array(int, nb_neurons, nb_inputs)?;
            let bias = read_array(int, 1, nb_neurons)?;

            Ok(DenseLayer {
                nb_inputs,
                nb_neurons,
                input_weights: input_weights.into(),
                bias: bias.into(),
                activation,
            })
        }

        fn read_gru(
            int: &mut impl FnMut() -> Result<i8, ReadModelError>,
        ) -> Result<GruLayer, ReadModelError> {
            let nb_inputs = unsigned(int()?)?;
            let nb_neurons = unsigned(int()?)?;
            let activation = act(int()?)?;
            let input_weights = read_array(int, 3 * nb_neurons, nb_inputs)?;
            let recurrent_weights = read_array(int, 3 * nb_neurons, nb_neurons)?;
            let bias = read_array(int, 1, 3 * nb_neurons)?;

            Ok(GruLayer {
                nb_inputs,
                nb_neurons,
                input_weights: input_weights.into(),
                recurrent_weights: recurrent_weights.into(),
                bias: bias.into(),
                activation,
            })
        }

        Ok(RnnModel {
            input_dense: read_dense(&mut read_int)?,
            vad_gru: read_gru(&mut read_int)?,
            noise_gru: read_gru(&mut read_int)?,
            denoise_gru: read_gru(&mut read_int)?,
            denoise_output: read_dense(&mut read_int)?,
            vad_output: read_dense(&mut read_int)?,
        })
    }
}

impl Default for RnnModel {
    fn default() -> RnnModel {
        use crate::model::*;

        let input_dense = DenseLayer {
            bias: Cow::Borrowed(&INPUT_DENSE_BIAS[..]),
            input_weights: Cow::Borrowed(&INPUT_DENSE_WEIGHTS[..]),
            nb_inputs: 42,
            nb_neurons: 24,
            activation: Activation::Tanh,
        };

        let vad_gru = GruLayer {
            bias: Cow::Borrowed(&VAD_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&VAD_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&VAD_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 24,
            nb_neurons: 24,
            activation: Activation::Relu,
        };

        let noise_gru = GruLayer {
            bias: Cow::Borrowed(&NOISE_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&NOISE_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&NOISE_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 90,
            nb_neurons: 48,
            activation: Activation::Relu,
        };

        let denoise_gru = GruLayer {
            bias: Cow::Borrowed(&DENOISE_GRU_BIAS[..]),
            input_weights: Cow::Borrowed(&DENOISE_GRU_WEIGHTS[..]),
            recurrent_weights: Cow::Borrowed(&DENOISE_GRU_RECURRENT_WEIGHTS[..]),
            nb_inputs: 114,
            nb_neurons: 96,
            activation: Activation::Relu,
        };

        let denoise_output = DenseLayer {
            bias: Cow::Borrowed(&DENOISE_OUTPUT_BIAS[..]),
            input_weights: Cow::Borrowed(&DENOISE_OUTPUT_WEIGHTS[..]),
            nb_inputs: 96,
            nb_neurons: 22,
            activation: Activation::Sigmoid,
        };

        let vad_output = DenseLayer {
            bias: Cow::Borrowed(&VAD_OUTPUT_BIAS[..]),
            input_weights: Cow::Borrowed(&VAD_OUTPUT_WEIGHTS[..]),
            nb_inputs: 24,
            nb_neurons: 1,
            activation: Activation::Sigmoid,
        };

        RnnModel {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        }
    }
}

impl DenseLayer {
    fn compute(&self, output: &mut [f32], input: &[f32]) {
        let m = self.nb_inputs;
        let n = self.nb_neurons;

        for i in 0..n {
            // Compute update gate.
            let sum =
                self.bias[i] as f32 + inner_p(&self.input_weights[(i * m)..((i + 1) * m)], input);
            output[i] = WEIGHTS_SCALE * sum;
        }
        match self.activation {
            Activation::Sigmoid => {
                for i in 0..n {
                    output[i] = sigmoid_approx(output[i]);
                }
            }
            Activation::Tanh => {
                for i in 0..n {
                    output[i] = tansig_approx(output[i]);
                }
            }
            Activation::Relu => {
                for i in 0..n {
                    output[i] = relu(output[i]);
                }
            }
        }
    }
}

impl GruLayer {
    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let mut z = [0.0; MAX_NEURONS];
        let mut r = [0.0; MAX_NEURONS];
        let m = self.nb_inputs;
        let n = self.nb_neurons;

        for i in 0..n {
            // Compute update gate.
            let sum = self.bias[i] as f32
                + inner_p(&self.input_weights[(i * m)..((i + 1) * m)], input)
                + inner_p(&self.recurrent_weights[(i * n)..((i + 1) * n)], state);
            z[i] = sigmoid_approx(WEIGHTS_SCALE * sum);
        }
        for i in 0..n {
            // Compute reset gate.
            let sum = self.bias[n + i] as f32
                + inner_p(&self.input_weights[((i + n) * m)..((i + n + 1) * m)], input)
                + inner_p(
                    &self.recurrent_weights[((i + n) * n)..((i + n + 1) * n)],
                    state,
                );
            // NOTE: our r[i] differs from the one in rnnoise because we're premultiplying it by
            // state[i].
            r[i] = state[i] * sigmoid_approx(WEIGHTS_SCALE * sum);
        }
        for i in 0..n {
            // Compute output.
            let sum = self.bias[2 * n + i] as f32
                + inner_p(
                    &self.input_weights[((i + 2 * n) * m)..((i + 2 * n + 1) * m)],
                    input,
                )
                + inner_p(
                    &self.recurrent_weights[((i + 2 * n) * n)..((i + 2 * n + 1) * n)],
                    &r[..],
                );
            let sum = match self.activation {
                Activation::Sigmoid => sigmoid_approx(WEIGHTS_SCALE * sum),
                Activation::Tanh => tansig_approx(WEIGHTS_SCALE * sum),
                Activation::Relu => relu(WEIGHTS_SCALE * sum),
            };
            state[i] = z[i] * state[i] + (1.0 - z[i]) * sum;
        }
    }
}

impl<'model> RnnState<'model> {
    pub(crate) fn new(model: Cow<'model, RnnModel>) -> RnnState<'model> {
        let vad_gru_state = vec![0.0f32; model.vad_gru.nb_neurons];
        let noise_gru_state = vec![0.0f32; model.noise_gru.nb_neurons];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru.nb_neurons];
        RnnState {
            model,
            vad_gru_state,
            noise_gru_state,
            denoise_gru_state,
        }
    }

    pub fn compute(&mut self, gains: &mut [f32], vad: &mut [f32], input: &[f32]) {
        assert_eq!(input.len(), INPUT_SIZE);

        let mut buf = [0.0; MAX_NEURONS * 3];
        let mut denoise_buf = [0.0; MAX_NEURONS * 3];
        let model = &self.model;

        let vad_gru_state = &mut self.vad_gru_state[..];
        let noise_gru_state = &mut self.noise_gru_state[..];
        let denoise_gru_state = &mut self.denoise_gru_state[..];
        model
            .input_dense
            .compute(&mut buf[0..model.input_dense.nb_neurons], input);
        model
            .vad_gru
            .compute(vad_gru_state, &buf[0..model.input_dense.nb_neurons]);
        model.vad_output.compute(vad, vad_gru_state);

        copy(&mut buf[model.input_dense.nb_neurons..], vad_gru_state);
        copy(
            &mut buf[(model.input_dense.nb_neurons + model.vad_gru.nb_neurons)..],
            input,
        );
        model.noise_gru.compute(noise_gru_state, &buf);

        copy(&mut denoise_buf, vad_gru_state);
        copy(
            &mut denoise_buf[model.vad_gru.nb_neurons..],
            noise_gru_state,
        );
        copy(
            &mut denoise_buf[(model.vad_gru.nb_neurons + model.noise_gru.nb_neurons)..],
            input,
        );
        model.denoise_gru.compute(denoise_gru_state, &denoise_buf);
        model.denoise_output.compute(gains, denoise_gru_state);
    }
}

const INPUT_SIZE: usize = 42;

fn copy(dst: &mut [f32], src: &[f32]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y;
    }
}
