use std::borrow::Cow;

use crate::util::{relu, sigmoid_approx, tansig_approx, zip3};

const MAX_NEURONS: usize = 128;

// It's annoying to expose a public API with `i8`s, because `include_bytes` works with `u8`s only.
// So we do conversions from `&[i8]` to `&[u8]` internally. Hopefully at some point rust will have
// a safe API for this...
fn to_i8(x: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const i8, x.len()) }
}

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

impl RnnModel {
    /// Reads an `RnnModel` from an array of bytes, in the format produced by the
    /// `nnnoiseless` training scripts.
    pub fn from_bytes(bytes: &[u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(to_i8(bytes), |xs| Cow::Owned(xs.to_owned()))
    }

    /// Reads an `RnnModel` from a static array of bytes, in the format produced by the
    /// `nnnoiseless` training scripts.
    ///
    /// This differs from [`RnnModel::from_bytes`] in that the returned model doesn't need to
    /// allocate its own byte buffers; it will just store references to the provided `bytes` array.
    ///
    /// For example, if you have your neural network weights available at compile-time then the
    /// following code will embed them into your binary and initialize a model without allocation:
    ///
    /// ```ignore
    /// let weight_data: &'static [u8] = include_bytes!("/path/to/model/weights.rnn");
    /// let model = RnnModel::from_static_bytes(weight_data).expect("Corrupted model file");
    /// ```
    pub fn from_static_bytes(bytes: &'static [u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(to_i8(bytes), Cow::Borrowed)
    }

    /// Reads an `RnnModel` from an array of bytes, in our new nnnoiseless format.
    ///
    /// The format is simple: each NN layer is represented by an array of signed `i8`'s,
    /// and these layers as simply concatenated.
    ///
    /// The format for a dense layer is
    /// <nb_neurons> <nb_inputs> <activation>
    /// <weights...>
    /// <bias...>
    /// where each of the <?> terms represents a single integer, and each of the <?...> terms
    /// represents an array of integers of the appropriate length (`weights` has length
    /// `nb_neurons * nb_inputs` and `bias` has length `nb_neurons`).
    ///
    /// The format for a GRU layer is
    /// <nb_neurons> <nb_inputs> <activation>
    /// <input_weights...>
    /// <recurrent_weights...>
    /// <bias...>
    /// where `input_weights` and `recurrent_weights` have length `3 * nb_inputs * nb_neurons` each,
    /// and `bias` has length `3 * nb_neurons`.
    fn from_bytes_impl<'a>(
        bytes: &'a [i8],
        moo: fn(&'a [i8]) -> Cow<'static, [i8]>,
    ) -> Option<RnnModel> {
        let read_array = |bytes: &'a [i8], len: usize| -> Option<(Cow<'static, [i8]>, &[i8])> {
            if bytes.len() >= len {
                Some((moo(&bytes[..len]), &bytes[len..]))
            } else {
                None
            }
        };

        fn unsigned(b: i8) -> Option<usize> {
            if b >= 0 {
                Some(b as usize)
            } else {
                None
            }
        }

        fn act(x: i8) -> Option<Activation> {
            match x {
                0 => Some(Activation::Tanh),
                1 => Some(Activation::Sigmoid),
                2 => Some(Activation::Relu),
                _ => None,
            }
        }

        let read_dense = |bytes: &'a [i8]| -> Option<(DenseLayer, &[i8])> {
            if bytes.len() < 3 {
                return None;
            }

            let nb_inputs = unsigned(bytes[0])?;
            let nb_neurons = unsigned(bytes[1])?;
            let activation = act(bytes[2])?;
            let (input_weights, bytes) = read_array(&bytes[3..], nb_neurons * nb_inputs)?;
            let (bias, bytes) = read_array(bytes, nb_neurons)?;

            let layer = DenseLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                bias,
                activation,
            };
            Some((layer, bytes))
        };

        let read_gru = |bytes: &'a [i8]| -> Option<(GruLayer, &[i8])> {
            if bytes.len() < 3 {
                return None;
            }

            let nb_inputs = unsigned(bytes[0])?;
            let nb_neurons = unsigned(bytes[1])?;
            let activation = act(bytes[2])?;
            let (input_weights, bytes) = read_array(&bytes[3..], 3 * nb_neurons * nb_inputs)?;
            let (recurrent_weights, bytes) = read_array(bytes, 3 * nb_neurons * nb_neurons)?;
            let (bias, bytes) = read_array(bytes, 3 * nb_neurons)?;

            let layer = GruLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                recurrent_weights,
                bias,
                activation,
            };
            Some((layer, bytes))
        };

        let (input_dense, bytes) = read_dense(bytes)?;
        let (vad_gru, bytes) = read_gru(bytes)?;
        let (noise_gru, bytes) = read_gru(bytes)?;
        let (denoise_gru, bytes) = read_gru(bytes)?;
        let (denoise_output, bytes) = read_dense(bytes)?;
        let (vad_output, bytes) = read_dense(bytes)?;

        if !bytes.is_empty() {
            return None;
        }

        // The input to the first layer must be of size 42, because that's how many features
        // there are. The denoise output must be of size 22, and the vad output must be of size 1.
        // Other than that, the output of one layer must match with the inputs of the following
        // layer.
        if input_dense.nb_inputs != 42
            || denoise_output.nb_neurons != 22
            || vad_output.nb_neurons != 1
        {
            return None;
        }
        if input_dense.nb_neurons != vad_gru.nb_inputs || vad_gru.nb_neurons != vad_output.nb_inputs
        {
            return None;
        }
        if 42 + input_dense.nb_neurons + vad_gru.nb_neurons != noise_gru.nb_inputs {
            return None;
        }
        if 42 + vad_gru.nb_neurons + noise_gru.nb_neurons != denoise_gru.nb_inputs {
            return None;
        }
        if denoise_gru.nb_neurons != denoise_output.nb_inputs {
            return None;
        }

        Some(RnnModel {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        })
    }
}

impl Default for RnnModel {
    fn default() -> RnnModel {
        let bytes: &'static [u8] = include_bytes!("weights.rnn");
        RnnModel::from_static_bytes(bytes).unwrap()
    }
}

impl DenseLayer {
    fn matrix(&self) -> SubMatrix {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons,
            offset: 0,
        }
    }

    fn compute(&self, output: &mut [f32], input: &[f32]) {
        copy_i8(output, &self.bias[..]);
        self.matrix().mul_add(output, input);

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
    fn input_submatrix(&self, offset: usize) -> SubMatrix {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn rec_submatrix(&self, offset: usize) -> SubMatrix {
        SubMatrix {
            data: self.recurrent_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let mut z = [0.0; MAX_NEURONS];
        let mut r = [0.0; MAX_NEURONS];
        let mut h = [0.0; MAX_NEURONS];
        let n = self.nb_neurons;

        // Compute update gate.
        copy_i8(&mut z[0..n], &self.bias[0..n]);
        self.input_submatrix(0).mul_add(&mut z[0..n], input);
        self.rec_submatrix(0).mul_add(&mut z[0..n], &state[..]);
        for z in z[0..n].iter_mut() {
            *z = sigmoid_approx(WEIGHTS_SCALE * *z);
        }

        // Compute reset gate.
        copy_i8(&mut r[0..n], &self.bias[n..(2 * n)]);
        self.input_submatrix(n).mul_add(&mut r[0..n], input);
        self.rec_submatrix(n).mul_add(&mut r[0..n], &state[..]);
        for (out, &s) in r[0..n].iter_mut().zip(&state[..]) {
            *out = s * sigmoid_approx(WEIGHTS_SCALE * *out);
        }

        // Compute output.
        copy_i8(&mut h[0..n], &self.bias[(2 * n)..]);
        self.input_submatrix(2 * n).mul_add(&mut h[0..n], input);
        self.rec_submatrix(2 * n).mul_add(&mut h[0..n], &r[0..n]);

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

fn copy_i8(dst: &mut [f32], src: &[i8]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y as f32;
    }
}

struct SubMatrix<'a> {
    data: &'a [i8],
    stride: usize,
    offset: usize,
}

impl<'a> SubMatrix<'a> {
    fn mul_add(&self, output: &mut [f32], input: &[f32]) {
        for (col, input) in self.data.chunks_exact(self.stride).zip(input) {
            for (&x, out) in col[self.offset..].iter().zip(&mut *output) {
                *out += x as f32 * input;
            }
        }
    }
}
