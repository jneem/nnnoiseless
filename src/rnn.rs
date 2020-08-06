use crate::util::{inner_p_bytes as inner_p, relu, sigmoid_approx, tansig_approx};

const MAX_NEURONS: usize = 128;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
}

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

#[derive(Copy, Clone)]
pub struct DenseLayer {
    /// An array of length `nb_neurons`.
    pub bias: &'static [i8],
    /// An array of length `nb_inputs * nb_neurons`.
    pub input_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Copy, Clone)]
pub struct GruLayer {
    /// An array of length `3 * nb_neurons`.
    pub bias: &'static [i8],
    /// An array of length `3 * nb_inputs * nb_neurons`.
    pub input_weights: &'static [i8],
    /// An array of length `3 * nb_neurons^2`.
    pub recurrent_weights: &'static [i8],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct RnnModel {
    pub input_dense_size: usize,
    pub input_dense: DenseLayer,
    pub vad_gru_size: usize,
    pub vad_gru: GruLayer,
    pub noise_gru_size: usize,
    pub noise_gru: GruLayer,
    pub denoise_gru_size: usize,
    pub denoise_gru: GruLayer,
    pub denoise_output_size: usize,
    pub denoise_output: DenseLayer,
    pub vad_output_size: usize,
    pub vad_output: DenseLayer,
}

#[derive(Clone)]
pub struct RnnState {
    model: &'static RnnModel,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
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

impl RnnState {
    pub fn new() -> RnnState {
        let model = &crate::model::MODEL;
        let vad_gru_state = vec![0.0f32; model.vad_gru_size];
        let noise_gru_state = vec![0.0f32; model.noise_gru_size];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru_size];
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
            .compute(&mut buf[0..model.input_dense_size], input);
        model
            .vad_gru
            .compute(vad_gru_state, &buf[0..model.input_dense_size]);
        model.vad_output.compute(vad, vad_gru_state);

        copy(&mut buf[model.input_dense_size..], vad_gru_state);
        copy(
            &mut buf[(model.input_dense_size + model.vad_gru_size)..],
            input,
        );
        model.noise_gru.compute(noise_gru_state, &buf);

        copy(&mut denoise_buf, vad_gru_state);
        copy(&mut denoise_buf[model.vad_gru_size..], noise_gru_state);
        copy(
            &mut denoise_buf[(model.vad_gru_size + model.noise_gru_size)..],
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
