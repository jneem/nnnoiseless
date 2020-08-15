use std::borrow::Cow;

use crate::util::{relu, sigmoid_approx, tansig_approx, zip3};

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
        let n = self.nb_neurons;

        for (out, bias) in output.iter_mut().zip(&self.bias[..]) {
            *out = *bias as f32;
        }

        for (col, y) in self.input_weights.chunks_exact(n).zip(input) {
            for (&x, out) in col.iter().zip(output.iter_mut()) {
                *out += x as f32 * y;
            }
        }

        match self.activation {
            Activation::Sigmoid => {
                for out in output.iter_mut() {
                    *out = sigmoid_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Tanh => {
                for out in output.iter_mut() {
                    *out = tansig_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Relu => {
                for out in output.iter_mut() {
                    *out = relu(*out * WEIGHTS_SCALE);
                }
            }
        }
    }
}

impl GruLayer {
    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let mut z = [0.0; MAX_NEURONS];
        let mut r = [0.0; MAX_NEURONS];
        let mut h = [0.0; MAX_NEURONS];
        let n = self.nb_neurons;
        let stride = n * 3;

        // Compute update gate.
        for (z, bias) in z.iter_mut().zip(&self.bias[0..n]) {
            *z = *bias as f32;
        }
        for (input_col, y) in self.input_weights.chunks_exact(stride).zip(input) {
            for (&x, z) in input_col[0..n].iter().zip(z.iter_mut()) {
                *z += x as f32 * y;
            }
        }
        for (rec_col, y) in self.recurrent_weights.chunks_exact(stride).zip(&state[..]) {
            for (&x, z) in rec_col[0..n].iter().zip(z.iter_mut()) {
                *z += x as f32 * y;
            }
        }
        for z in z[0..n].iter_mut() {
            *z = sigmoid_approx(WEIGHTS_SCALE * *z);
        }

        // Compute reset gate.
        for (dst, src) in r.iter_mut().zip(&self.bias[n..(2 * n)]) {
            *dst = *src as f32;
        }
        for (input_col, y) in self.input_weights.chunks_exact(stride).zip(input) {
            for (&x, out) in input_col[n..(2 * n)].iter().zip(r.iter_mut()) {
                *out += x as f32 * y;
            }
        }
        for (rec_col, y) in self.recurrent_weights.chunks_exact(stride).zip(&state[..]) {
            for (&x, out) in rec_col[n..(2 * n)].iter().zip(r.iter_mut()) {
                *out += x as f32 * y;
            }
        }
        for (out, &s) in r[0..n].iter_mut().zip(&state[..]) {
            *out = s * sigmoid_approx(WEIGHTS_SCALE * *out);
        }

        // Compute output.
        for (dst, src) in h.iter_mut().zip(&self.bias[(2 * n)..]) {
            *dst = *src as f32;
        }
        for (input_col, y) in self.input_weights.chunks_exact(stride).zip(input) {
            for (&x, out) in input_col[(2 * n)..].iter().zip(h.iter_mut()) {
                *out += x as f32 * y;
            }
        }
        for (rec_col, y) in self.recurrent_weights.chunks_exact(stride).zip(&r[0..n]) {
            for (&x, out) in rec_col[(2 * n)..].iter().zip(h.iter_mut()) {
                *out += x as f32 * y;
            }
        }

        for (s, &z, &h) in zip3(state, &z[0..n], &h[0..n]) {
            let h = match self.activation {
                Activation::Sigmoid => sigmoid_approx(WEIGHTS_SCALE * h),
                Activation::Tanh => tansig_approx(WEIGHTS_SCALE * h),
                Activation::Relu => relu(WEIGHTS_SCALE * h),
            };
            *s = z * *s + (1.0 - z) * h;
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
